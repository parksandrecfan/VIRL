# TO DO:
# save validation best only
# patience
# lora

# test on test.yaml
import yaml
import torch
# import torch
import math
import numpy as np    
import matplotlib.pyplot as plt
# decode
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

import pickle
import json
import os
import random
import torch_geometric as tg
import pytorch_lightning as pl
import argparse
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
import shutil

import sys
sys.path.append('automatemain')
# print(os.getcwd())
from automate.sbgcn import LinearBlock, BipartiteResMRConv 
sys.path.remove('automatemain')

# os.chdir("automatemain")
# from automate.sbgcn import  LinearBlock, BipartiteResMRConv 
# os.chdir("../")
from hybridbrep.implicit import implicit_part_to_data, ImplicitDecoder

def mse_loss(pred, label, method = "sum"):
    if method=="mean":
        loss = torch.mean((pred-label)**2)
    if method=="sum":
        loss = torch.sum((pred-label)**2)
    return loss

def mae_loss(pred, label, method = "sum"):
    # print((pred*label).shape)
    if method=="mean":
        loss = torch.mean(torch.abs(pred-label))
    if method=="sum":
        loss = torch.sum(torch.abs(pred-label))
    return loss

def mape_loss(pred, label, method = "sum"):
    if method=="mean":
        loss = torch.mean(torch.abs((pred-label)/(label+eps)))
    if method=="sum":
        loss = torch.sum(torch.abs((pred-label)/(label+eps)))
    return loss

def huber_loss(pred, label, method = "sum"):
    return torch.nn.functional.huber_loss(pred, label, reduction=method)

def get_filelist(path, include=None, exclude=None):
    Filelist = []
    for home, dirs, files in os.walk(path):
        for filename in files:
            include_criteria=True
            exclude_criteria=True
            if isinstance(include, str):
                if include not in filename:
                    include_criteria=False
            elif isinstance(include, list):
                for i in include:
                    if i not in filename:
                        include_criteria=False
                        continue
            if isinstance(exclude, str):
                if exclude in filename:
                    exclude_criteria=False
            elif isinstance(exclude, list):
                for e in exclude:
                    if e in filename:
                        exclude_criteria=False
                        continue
            if include_criteria==True and exclude_criteria==True:
                Filelist.append(home+'/'+filename)
    return Filelist
    
def run(args):
    # Specify the path to your YAML file
    # yaml_file_path = 'aft-area-mae1.yaml'
    yaml_file_path = args.yaml
    task_name = yaml_file_path.split('/')[-1].split('.')[0] #ex: exp79
    # Open the YAML file
    with open(yaml_file_path, 'r') as file:
        # Load the YAML content
        config = yaml.safe_load(file)
    
    # Now, yaml_content is a Python dictionary containing the data from the YAML file
    # print(yaml_content)
    
    # yaml_content['seq_len']
    # baseline = config['baseline']
    # adapt_lin = config['adapt_lin']
    splits = config['splits']
    add_size = config['add_size']
    patience = 3
    
    range_root = config['range_root'] if 'range_root' in config else None
    range_log = config['range_log']
    norm_const = config['norm_const'] if 'norm_const' in config else None

    # range_root = "project/latent/range"
    label_root = config['label_root']
    label_ext = config['label_ext']
    label_log = config["label_log"]
    # print(label_log==True)
    
    # label_root = "project/areas"
    train_batch = config['train_batch']
    # train_batch=64
    test_batch = config['test_batch']
    # test_batch=64
    hidden = config["hidden"]
    latent_w = config["latent_w"] if "latent_w" in config else 64
    device = config['device']
    # device = 'cuda:2' if torch.cuda.is_available() else 'cpu'
    # print("Using device:", device)
    
    # iteration=10000
    iteration = config['iteration']
    shots = config["shots"] #10
    # shots = int(shots*20/19)
    seed = config["seed"] #0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # iteration=30000
    val_every = config['val_every']
    val_pct=20 # 5 before 1764
    # val_every=50
    save_dir = config['save_dir']
    # save_dir = "partial_finetune/aft/mae"
    # task_name = config['task_name']
    # task_name ="area_regress"
    
    # eps = config['eps']
    eps = 1e-20
    lr = config['lr']
    # lr=0.001
    lr_step = config['lr_step']
    # lr_step = 50
    gamma = config['gamma']
    # gamma = 0.99
    if config['loss_func']=="mae_loss":
        loss_func=mae_loss
    if config['loss_func']=="mse_loss":
        loss_func=mse_loss
    if config['loss_func']=="mape_loss":
        loss_func=mape_loss
    if config['loss_func']=="huber_loss":
        loss_func=huber_loss
    
    pretrain = config['pretrain'] if 'pretrain' in config else "None"
    # pretrain = "area"
    pre_path = config['pre_path'] if 'pre_path' in config else "None"
    # pre_path = "project/models/BRepFaceAutoencoder_64_1024_4/BRepFaceAutoencoder_64_1024_4.ckpt"
    partial = config['partial'] if 'partial' in config else "None"
    if partial=='lora':
        intral_ab = config['intral_ab'] if 'intral_ab' in config else True
        inter_ab = config['inter_ab'] if 'inter_ab' in config else True
    # partial = "finetune"
    lr_ratio = config['lr_ratio']
    regress_norm = config['regress_norm']
    regress_tdi = config['regress_tdi']
    
    rank = config['rank'] if 'rank' in config else "None"
    if seed ==1:
        shutil.copy(yaml_file_path, save_dir+"/"+task_name+"/config.yaml")
        for key, value in config.items():
            print(f'{key}: {value}')    
        print('end lr:', lr*gamma**int(iteration/lr_step))
    
    # these are just for loading torch models, not for forward pass, so only init is included
    
    class BRepFaceAutoencoder(pl.LightningModule):
        def __init__(self, code_size=64, hidden_size=1024, decoder_layers=4):
            super().__init__()
            self.encoder = BRepFaceEncoder(code_size, hidden_size)
            self.decoder = ImplicitDecoder(64+2, 4, hidden_size, decoder_layers, use_tanh=False)

    
    class deepsdf(nn.Module):
        def __init__(self, code_size):
            super(deepsdf, self).__init__()
            self.lin1 = nn.utils.weight_norm(nn.Linear(code_size+5+3,1024)) #5(xyz and 123combo)+64+3(UVW)
            self.lin2 = nn.utils.weight_norm(nn.Linear(code_size+3+1024,1024)) # from this point just 64+uvw
            self.lin3 = nn.utils.weight_norm(nn.Linear(code_size+3+1024,1024))
            self.lin4 = nn.utils.weight_norm(nn.Linear(code_size+3+1024,1))
            # self.act = nn.ReLU()
            self.act = nn.LeakyReLU(negative_slope=0.001)
            
    class deepbox(nn.Module):
        def __init__(self, code_size):
            super(deepbox, self).__init__()
            self.lin1 = nn.utils.weight_norm(nn.Linear(code_size+5,64)) #64+3(UVW)
            self.lin2 = nn.utils.weight_norm(nn.Linear(code_size+64,64))
            self.lin3 = nn.utils.weight_norm(nn.Linear(code_size+64,3))
            # self.act = nn.ReLU()
            self.act = nn.LeakyReLU(negative_slope=0.001)
            
    class BRepVolAutoencoder(nn.Module):
        def __init__(self, code_size=64, hidden_size=1024, device=0):
            super().__init__()
            self.encoder = BRepFaceEncoder(code_size, hidden_size)
            self.decoder1 = deepbox(code_size)
            self.decoder2 = deepsdf(code_size)
    
    # import dataset, from Ben's
    def load_item(path):
        open_file = open(path, "rb")
        this_item=pickle.load(open_file)
        open_file.close()
        return this_item
        
    def dump_item(this_item, path):
        open_file = open(path, "wb")
        pickle.dump(this_item, open_file)
        open_file.close()

    def create_subset(data, seed, size):
        random.seed(seed)
        return random.sample(data, size)
    
    class BRep_regress(torch.utils.data.Dataset):
        def __init__(
            self, 
            splits, 
            data_root, 
            range_root, 
            mode='train', 
            preshuffle=False,
            label_root = None,
            label_ext = 'pt',
            seed = None,
            size = None):
            
            split = mode
            if mode=='validate':
                split = 'val'
    
            with open(splits, 'r') as f:
                ids = json.load(f)[split]
    
            if preshuffle and mode in ['train','val']: #never preshuffle, but always shooting, so randomness is here
                random.seed(seed)
                random.shuffle(ids)
            
            if mode=='train':
                ids = create_subset(ids, seed, shots)
            # elif mode=='validate':
            #     ids = ids[size:]
            
            self.all_paths = [os.path.join(data_root, f'{id}.pt') for id in ids]
            self.all_ranges = [os.path.join(range_root, f'{id}.pt') for id in ids]
            self.all_names = ids
            self.label_ext = label_ext
            self.all_labels = [os.path.join(label_root, f'{id}.{self.label_ext}') for id in ids] if label_root and label_ext else None
    
                
        def __getitem__(self, idx):
            data = torch.load(self.all_paths[idx])        
            data['face_ct'] = int(data['num_faces'])
            del data['largest_face']
            del data['sdf_max']
            del data['curve_max']
            del data['surf_max']
    
            data_range = torch.load(self.all_ranges[idx])   
            if self.label_ext == 'pt':
                label = torch.load(self.all_labels[idx]) 
            if self.label_ext == 'pkl':
                label = load_item(self.all_labels[idx])
            if label_log == True:
                label = max(math.log10(label+eps), 0)
                if norm_const!=None:
                    label = label*norm_const
            if range_log == True:
                data_range = torch.log10(data_range+eps)
            return data, data_range, float(label)
    
        def __len__(self):
            return len(self.all_paths)

    class lora_model(nn.Module):
        def __init__(self, hidden_size, rank):
            super().__init__()
            # print("here")
            self.v_in_a = nn.Linear(hidden_size, rank, bias=False)
            self.v_in_b = nn.Linear(rank, hidden_size, bias=False)
            self.v_in_b.weight.data.fill_(0)
            self.e_in_a = nn.Linear(hidden_size, rank, bias=False)
            self.e_in_b = nn.Linear(rank, hidden_size, bias=False)
            self.e_in_b.weight.data.fill_(0)
            self.l_in_a = nn.Linear(hidden_size, rank, bias=False)
            self.l_in_b = nn.Linear(rank, hidden_size, bias=False)
            self.l_in_b.weight.data.fill_(0)
            self.f_in_a = nn.Linear(hidden_size, rank, bias=False)
            self.f_in_b = nn.Linear(rank, hidden_size, bias=False)
            self.f_in_b.weight.data.fill_(0)
            self.ve_a = nn.Linear(hidden_size, rank, bias=False)
            self.ve_b = nn.Linear(rank, hidden_size, bias=False)
            self.ve_b.weight.data.fill_(0)
            self.el_a = nn.Linear(hidden_size, rank, bias=False)
            self.el_b = nn.Linear(rank, hidden_size, bias=False)
            self.el_b.weight.data.fill_(0)
            self.lf_a = nn.Linear(hidden_size, rank, bias=False)
            self.lf_b = nn.Linear(rank, hidden_size, bias=False)
            self.lf_b.weight.data.fill_(0)
        # def forward(x):
        #     pass
    
    class BRepFaceEncoder(nn.Module):
        def __init__(self, code_size, hidden_size):
            super().__init__()
            self.v_in = LinearBlock(3, hidden_size)
            self.e_in = LinearBlock(15, hidden_size)
            self.l_in = LinearBlock(10, hidden_size)
            self.f_in = LinearBlock(17, hidden_size)
            self.v_to_e = BipartiteResMRConv(hidden_size)
            self.e_to_l = BipartiteResMRConv(hidden_size)
            self.l_to_f = BipartiteResMRConv(hidden_size)
            self.hidden = hidden_size
            if partial=="lora":
                self.lora = lora_model(hidden_size, rank)
                
            self.proj = LinearBlock(hidden_size, code_size)
            
        def forward(self, data):
            if partial=="lora":
                # print(self.lora.v_in_b.weight[0,0])
                v = self.v_in(data.vertex_positions)
                
                e = torch.cat([data.edge_curves, data.edge_curve_parameters, data.edge_curve_flipped.reshape((-1,1))], dim=1)
                e = self.e_in(e)
                
                l = self.l_in(data.loop_types.float())
                
                f = torch.cat([data.face_surfaces, data.face_surface_parameters, data.face_surface_flipped.reshape((-1,1))], dim=1)
                f = self.f_in(f)

                if intral_ab==True:
                    v_lora = self.lora.v_in_b(self.lora.v_in_a(v))
                    v = v + v_lora
                    e_lora = self.lora.e_in_b(self.lora.e_in_a(e))
                    e = e + e_lora
                    l_lora = self.lora.l_in_b(self.lora.l_in_a(l))
                    l = l + l_lora
                    f_lora = self.lora.f_in_b(self.lora.f_in_a(f))
                    f = f + f_lora
                
                # TODO - incorporate edge-loop data and vert-edge data
                # Potential TODO - heterogenous input of edges and curves based on function type
                if inter_ab==True:
                    e = self.v_to_e(v, e, data.edge_to_vertex[[1,0]])
                    
                    e_lora = self.lora.ve_b(self.lora.ve_a(e))
                    e = e+e_lora
                    
                    l = self.e_to_l(e, l, data.loop_to_edge[[1,0]])
                    l_lora = self.lora.el_b(self.lora.el_a(l))
                    l=l+l_lora
                    
                    f = self.l_to_f(l, f, data.face_to_loop[[1,0]])
                    f_lora = self.lora.lf_b(self.lora.lf_a(f))
                    f=f+f_lora
                    
                else:
                    e = self.v_to_e(v, e, data.edge_to_vertex[[1,0]])
                    l = self.e_to_l(e, l, data.loop_to_edge[[1,0]])
                    f = self.l_to_f(l, f, data.face_to_loop[[1,0]])
                    
            
            else:
                v = self.v_in(data.vertex_positions)
                e = torch.cat([data.edge_curves, data.edge_curve_parameters, data.edge_curve_flipped.reshape((-1,1))], dim=1)
                e = self.e_in(e)
                l = self.l_in(data.loop_types.float())
                f = torch.cat([data.face_surfaces, data.face_surface_parameters, data.face_surface_flipped.reshape((-1,1))], dim=1)
                f = self.f_in(f)
                # TODO - incorporate edge-loop data and vert-edge data
                # Potential TODO - heterogenous input of edges and curves based on function type
                e = self.v_to_e(v, e, data.edge_to_vertex[[1,0]])
                l = self.e_to_l(e, l, data.loop_to_edge[[1,0]])
                f = self.l_to_f(l, f, data.face_to_loop[[1,0]])

            
            if self.hidden!=64:
                return self.proj(f)
            else:
                return f
            
    class BRepVolRegress(nn.Module):
        def __init__(self, code_size=64, hidden_size=1024, device=0):
            super().__init__()
            self.encoder = BRepFaceEncoder(code_size, hidden_size)
            # regress_model = torch.nn.Linear(69,1, bias=True) #64+range5=69
            # regress_model.weight = torch.nn.Parameter(torch.ones((1,69)))
            # regress_model.bias = torch.nn.Parameter(torch.ones(1))
            regress_model = nn.Sequential(
            nn.Linear(add_size+code_size, 64),
            # nn.LeakyReLU(),
            nn.Sigmoid(),
            nn.Linear(64, 1)
        )
            
            self.regress = regress_model
            # if adapt_lin == True:
            #     self.adapt_lin = nn.Linear(add_size,1)
            #     self.adapt_lin.weight.data.fill_(1)
            #     self.adapt_lin.bias.data.fill_(0)
                
        def forward(self, data, data_range):
            codes = self.encoder(data)
            
            # Create an empty tensor to store the pooled segments
            pooled_data = torch.zeros(len(data.face_ct), codes.size(1), dtype=codes.dtype, device=codes.device)
            
            # Calculate the cumulative sums for the segments
            cumulative_sums = torch.cumsum(data.face_ct, dim=0)
            segment_starts = torch.cat([torch.zeros(1, device=codes.device), cumulative_sums[:-1]])
            
            # Loop through the segments and perform differentiable max pooling
            for i, start in enumerate(segment_starts):
                end = cumulative_sums[i]
                segment = codes[int(start):int(end)]
                pooled_segment = torch.max(segment, 0)[0]
                # pooled_segment = F.adaptive_max_pool1d(segment.permute(1, 0).unsqueeze(0), (1,)).squeeze(0).permute(1, 0)
                pooled_data[i] = pooled_segment
            
            # pred = torch.abs(self.regress(pooled_data)) ###ReLU because volume is positive
            # return pred[:,0]
            # data_range = torch.rand(data_range.shape).to(codes.device)
            # print(data_range)
            # print(pooled_data.shape, data_range.shape)
            if add_size != 0:
                codes_cat = torch.cat([pooled_data, data_range],axis=1)
            else:
                codes_cat = pooled_data
            pred = torch.abs(self.regress(codes_cat)) ###ReLU because volume is positive
            if regress_norm == True:
                # if adapt_lin ==True:
                #     adapted = self.adapt_lin(data_range)[:,0]
                #     # return adapted # for sanity check, when lr=0
                #     # print(pred[:,0].shape, adapted.shape)
                #     return pred[:,0] * adapted
                # else:
                    # print(data_range[:,-1].shape)
                    # print(pred.shape, data_range.shape)
                    # print(pred[:,0].mean())
                return pred[:,0] * data_range[:,-1]
            else:
                return pred[:,0]
    
    # labels are volumes instead of SDF
    

    
    # range_root = "project/latent/range"
    # label_root = "project/volumes"
    
    ds = BRep_regress(splits=splits, data_root='inputs/data/simple_preprocessed', \
                      range_root = range_root, mode='train', preshuffle=False, label_root=label_root,\
                      label_ext = label_ext, seed=seed, size=shots)
    
    ds_val = BRep_regress(splits=splits, data_root='inputs/data/simple_preprocessed', \
                      range_root = range_root, mode='validate', preshuffle=False, label_root=label_root,\
                         label_ext = label_ext)

    ds_test = BRep_regress(splits=splits, data_root='inputs/data/simple_preprocessed', \
                      range_root = range_root, mode='test', preshuffle=False, label_root=label_root,\
                          label_ext = label_ext)
    
    print(f'Train Set size = {len(ds)}')
    # ds_val = BRepDS('project/data/simple_train_test.json', 'project/data/simple_preprocessed', 'validate')
    print(f'Val Set Size = {len(ds_val)}')
    print(f'test Set Size = {len(ds_test)}')
    # set_val = set(ds_val.all_names)
    set_train = set(ds.all_names)

    # sanity check that no overlapping between sets
    # with open("inputs/data/simple_train_val_test.json", 'r') as f:
    #     loaded = json.load(f)
    #     orig_train = set(loaded['train'])
    #     orig_val = set(loaded['val'])
    #     orig_test = set(loaded['test'])
    #     print(orig_train.intersection(set_val))
    #     print(orig_val.intersection(set_train))
    #     print(orig_test.intersection(set_train))
    #     print(orig_test.intersection(set_val))

    
    # print(ds_val.all_names)
    # print(ddgsd)
    # train_batch=64
    # test_batch=64
    dl = tg.loader.DataLoader(ds, batch_size=train_batch, shuffle=True, num_workers=8, persistent_workers=True)
    val_batch = train_batch
    dl_val = tg.loader.DataLoader(ds_val, batch_size=val_batch, shuffle=False, num_workers=8, persistent_workers=True)
    dl_test = tg.loader.DataLoader(ds_test, batch_size=test_batch, shuffle=False, num_workers=8, persistent_workers=True)
    # create the linear regression model, based on train + val set
    if regress_tdi==True and add_size!=0:
        xs = torch.ones(0)
        ys = torch.ones(0)
        for j, (train_data, train_data_range,train_label) in enumerate(dl):
            x = train_data_range
            y = train_label
            xs=torch.cat((xs,x))
            ys=torch.cat((ys,y))
        # print(xs.mean(), ys.mean())
    
        # for j, (val_data, val_data_range,val_label) in enumerate(dl_val):
        #     x = val_data_range
        #     y = val_label
        #     xs=torch.cat((xs,x))
        #     ys=torch.cat((ys,y))

        reg = LinearRegression().fit(xs,ys)
        # check if the r2 score makes sense
        sklearn_pred = reg.predict(xs)
        lin_train_r2 = r2_score(ys, sklearn_pred)
        print("train R2 score: ", lin_train_r2)
        
        plt.figure(figsize=(5,5))
        plt.title("train set linear regression: %.3f"%lin_train_r2)
        plt.scatter(ys, sklearn_pred)
        minr = min(min(ys), min(sklearn_pred))
        maxr = max(max(ys), max(sklearn_pred))
        plt.plot([minr, maxr],[minr, maxr],"r--")
        plt.savefig("%s/%s/train_baseline_r2.jpg"%(save_dir, task_name))

        
        xs = torch.ones(0)
        ys = torch.ones(0)
        for j, (test_data, test_data_range,test_label) in enumerate(dl_test):
            x = test_data_range
            y = test_label
            xs=torch.cat((xs,x))
            ys=torch.cat((ys,y))
        
        sklearn_pred = reg.predict(xs)
        lin_test_r2 = r2_score(ys, sklearn_pred)
        print("test R2 score: ", lin_test_r2)

        plt.figure(figsize=(5,5))
        plt.title("test set linear regression: %.3f"%lin_test_r2)
        plt.scatter(ys, sklearn_pred)
        minr = min(min(ys), min(sklearn_pred))
        maxr = max(max(ys), max(sklearn_pred))
        plt.plot([minr, maxr],[minr, maxr],"r--")
        plt.savefig("%s/%s/test_baseline_r2.jpg"%(save_dir, task_name))
    # print(data_range.shape, label.shape)
    
    # chekc the two sets are completely different
    # set1 = set(ds.all_names)
    # set2 = set(ds_val.all_names)
    # set1.intersection(set2)
    
    model = BRepVolRegress(latent_w, hidden)
    # print(model.regress[0].bias.sum())
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    print(f"encoder learnable parameters: {encoder_params}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"total learnable parameters: {total_params}")
    
    # model = model.to(device)
    if pretrain== "None":
        pass
    elif pretrain=="area":
        pre_model = BRepFaceAutoencoder(latent_w, hidden)
        print("pretrained path", pre_path)
        # a=torch.load(pre_path)['state_dict']
        a=torch.load(pre_path, map_location=torch.device('cpu'))
        # print(a)
        # pre_model.load_state_dict(a)
        pre_model.load_state_dict(a, strict=False)
        pre_model.eval()
        model.encoder = pre_model.encoder
        # print(model.encoder)
        # print(model.device)
    
    elif pretrain=="volume":
        pre_model = BRepVolAutoencoder(latent_w, hidden)
        print("pretrained path", pre_path)
        a=torch.load(pre_path, map_location=torch.device('cpu'))
        # print(a.keys())
        # pre_model.load_state_dict(a)
        pre_model.load_state_dict(a, strict=False)
        pre_model.eval()
        model.encoder = pre_model.encoder
        # print(model.encoder)

        # model.encoder.drop_layer.p=0 # dropout parameters are not carried in model.state_dict()
        # set dropout to 0
        
        # print(model.encoder.proj.f[0].bias)
    
    model = model.to(device)
    
    if partial== "None":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif partial=="lin":
        optimizer = optim.Adam(model.regress.parameters(), lr=lr)
    elif partial=="finetune":
        optimizer = optim.Adam([{'params': model.encoder.parameters(), 'lr': lr_ratio * lr},
                                {'params': model.regress.parameters(), 'lr': lr}], lr=lr)
    elif partial=="lora":
        optimizer = optim.Adam([{'params': model.encoder.lora.parameters(), 'lr': lr_ratio*lr},
                                {'params': model.regress.parameters(), 'lr': lr}], lr=lr)
    
    
    scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=gamma) # after training lr->8e-5
    
    train_loss_log=[]
    train_loss_log_avg=[]
    val_loss_log=[]
    test_loss_log=[]
    
    
    itering = iter(dl)
    batch_ct = 1
    # train_loss_doc = 0
    for i in range(iteration):
            
        model.train()
        optimizer.zero_grad()
            
        data, data_range, label  = next(itering)
        # print(type(data_range), data_range.shape)
        if regress_tdi==True and add_size!=0:
            data_range = torch.tensor(reg.predict(data_range)).unsqueeze(1)
        # print(type(data_range), data_range.shape)
        # print(data_range.sum())
        
        if (data_range.shape[0]<train_batch) == True:
            # print("%s batch done"%batch_ct)
            itering = iter(dl)
            batch_ct+=1
        
        data=data.to(device)
        data_range=data_range.to(device) #128x5
        label=label.to(device)
        # train on normalized label?
        # range5 = torch.tensor((xn, yn, zn, xyz_min, xyz_max))
        # if regress_norm==True:
        #     label = label/data_range[:,-1]
            # xyz_min = data_range[:,3] #128
            # xyz_max = data_range[:,4] #128
            # xr = xyz_min+data_range[:,0]*(xyz_max-xyz_min) #128
            # yr = xyz_min+data_range[:,1]*(xyz_max-xyz_min) #128
            # zr = xyz_min+data_range[:,2]*(xyz_max-xyz_min) #128
            # box_v = xr*yr*zr+eps #128
            # label = label/box_v
        
        # print(label.shape, label)
        # print(data.shape)
        # print(data_range.shape)
        
        pred = model(data, data_range)
        # print("pred", pred.shape, pred.dtype)
        # print("label", label.shape, label.dtype)
        train_loss=loss_func(pred, label, method="mean")
        # print(data_range, label)
        # print(reaewrwer)
        # train_loss_doc += train_loss.item()
        train_loss.backward()
        optimizer.step()
        scheduler.step()
        train_loss_log.append(train_loss.item())
        if i<val_every:
            train_loss_log_avg.append(sum(train_loss_log)/len(train_loss_log))
        else:
            train_loss_log_avg.append((sum(train_loss_log[-val_every:]))/val_every)
        # print(train_loss.item())
        dump_item(train_loss_log_avg, "%s/%s/train_loss_log.pkl"%(save_dir, task_name))
        
        if i%val_every==0:
            with torch.no_grad():
                # get validation loss
                model.eval()
                
                val_loss_sum=0
                for j, (val_data, val_data_range,val_label) in enumerate(dl_val):
                    val_data=val_data.to(device)
                    if regress_tdi==True and add_size!=0:
                        val_data_range = torch.tensor(reg.predict(val_data_range)).unsqueeze(1)
                    val_data_range=val_data_range.to(device)
                    val_label=val_label.to(device)

                    # if regress_norm==True:
                    #     val_label = val_label/val_data_range[:,-1]
                        # xyz_min = val_data_range[:,3] #128
                        # xyz_max = val_data_range[:,4] #128
                        # xr = xyz_min+val_data_range[:,0]*(xyz_max-xyz_min) #128
                        # yr = xyz_min+val_data_range[:,1]*(xyz_max-xyz_min) #128
                        # zr = xyz_min+val_data_range[:,2]*(xyz_max-xyz_min) #128
                        # box_v = xr*yr*zr+eps #128
                        # val_label = val_label/box_v
    
                    val_pred = model(val_data, val_data_range)
                    val_batch_loss=loss_func(val_pred, val_label, method="sum")
                    val_loss_sum+=val_batch_loss.item()
                    
                val_loss=val_loss_sum/len(ds_val.all_names)
                val_loss_log.append(val_loss)

                test_loss_sum=0
                for j, (test_data, test_data_range,test_label) in enumerate(dl_test):
                    test_data=test_data.to(device)
                    if regress_tdi==True and add_size!=0:
                        test_data_range = torch.tensor(reg.predict(test_data_range)).unsqueeze(1)
                    test_data_range=test_data_range.to(device)
                    test_label=test_label.to(device)

                    # if regress_norm==True:
                    #     test_label = test_label/test_data_range[:,-1]
                        # xyz_min = test_data_range[:,3] #128
                        # xyz_max = test_data_range[:,4] #128
                        # xr = xyz_min+test_data_range[:,0]*(xyz_max-xyz_min) #128
                        # yr = xyz_min+test_data_range[:,1]*(xyz_max-xyz_min) #128
                        # zr = xyz_min+test_data_range[:,2]*(xyz_max-xyz_min) #128
                        # box_v = xr*yr*zr+eps #128
                        # test_label = test_label/box_v
    
                    test_pred = model(test_data, test_data_range)
                    test_batch_loss=loss_func(test_pred, test_label, method="sum")
                    test_loss_sum+=test_batch_loss.item()
                    
                test_loss=test_loss_sum/len(ds_test.all_names)
                test_loss_log.append(test_loss)                
                 
                print("iter: %s, train: %.3f, val: %.3f, test: %.3f"%(i, train_loss_log_avg[-1], val_loss, test_loss))
                # to save some storage
                if len(val_loss_log)<=2 or (len(val_loss_log)>2 and min(val_loss_log[:-1])-val_loss>0):
                    # delete all other models
                    del_paths = get_filelist("%s/%s"%(save_dir, task_name), include=".pt", exclude=[".yaml", ".pkl", ".jpg"])
                    # print(del_paths)
                    for del_path in del_paths:
                        os.remove(del_path)
                    
                    # print("save best val")
                    torch.save(model.state_dict(),"%s/%s/%s_iter_val_%.3f.pt"%(save_dir, task_name, i, val_loss))
                dump_item(val_loss_log, "%s/%s/val_loss_log.pkl"%(save_dir, task_name))
                dump_item(test_loss_log, "%s/%s/test_loss_log.pkl"%(save_dir, task_name))
                if i//val_every - np.argmin(val_loss_log) ==patience: #not getting updated
                    break
                
    dump_item(train_loss_log_avg, "%s/%s/train_loss_log.pkl"%(save_dir, task_name))
    dump_item(val_loss_log, "%s/%s/val_loss_log.pkl"%(save_dir, task_name))
    dump_item(test_loss_log, "%s/%s/test_loss_log.pkl"%(save_dir, task_name))
    # if i>int(iteration/2): #save storage
    #     torch.save(model.state_dict(),"%s/%s/%s_iter.pt"%(save_dir, task_name, i))
    
    # print("test loss:", test_loss_log[-1])
    for key, value in config.items():
        if key=="task_name":
            print(f'{key}: {value}')   
    train_iter_min = np.argmin(np.array(train_loss_log_avg))
    # print(task_name,"best train iter:", train_iter_min)
    # print(task_name,"best train loss:", train_loss_log_avg[train_iter_min])
    val_iter_min = np.argmin(np.array(val_loss_log))
    # print(task_name,"best val iter:", val_iter_min * val_every)
    # print(task_name,"best val loss:", val_loss_log[val_iter_min])
    # print(task_name,"best val test loss:", test_loss_log[val_iter_min])

    plt.plot(train_loss_log_avg)
    # val_every=20
    val_loss_log2 = [i for i in val_loss_log for j in range(val_every)]
    plt.yscale('log')
    plt.plot(val_loss_log2)
    test_loss_log2 = [i for i in test_loss_log for j in range(val_every)]
    plt.plot(test_loss_log2)
    plt.savefig("%s/%s/loss_plot.jpg"%(save_dir, task_name))

    # TO DO: find the best test iteration, plot R2 and get test accuracy
    val_pred_whole = []
    val_gt_whole = []

    
    r2_folder = "%s/%s"%(save_dir, task_name)
    r2_path = get_filelist(r2_folder, include = "%s_iter"%(val_iter_min * val_every))[0]
    print(r2_path)
    a=torch.load(r2_path)
    model.cpu()
    model.load_state_dict(a)
    model.to(device)
    with torch.no_grad():
        model.eval()
        for j, (val_data, val_data_range,val_label) in enumerate(dl_val):
            val_data=val_data.to(device)
            
            if regress_tdi==True and add_size!=0:
                val_data_range = torch.tensor(reg.predict(val_data_range)).unsqueeze(1)
            
            val_data_range=val_data_range.to(device)
            val_label=val_label.to(device)
            
            val_pred = model(val_data, val_data_range)

            # if regress_norm==True:
            #     val_pred = val_data_range[:,-1]*val_pred
                # xyz_min = val_data_range[:,3] #128
                # xyz_max = val_data_range[:,4] #128
                # xr = xyz_min+val_data_range[:,0]*(xyz_max-xyz_min) #128
                # yr = xyz_min+val_data_range[:,1]*(xyz_max-xyz_min) #128
                # zr = xyz_min+val_data_range[:,2]*(xyz_max-xyz_min) #128
                # box_v = xr*yr*zr+eps #128
                # val_pred = box_v*val_pred
            # if label_log ==True:
            #     # print(val_pred)
            #     val_pred = torch.pow(10, val_pred)
            #     val_label = torch.pow(10, val_label)
            val_pred_whole = val_pred_whole + val_pred.tolist()
            val_gt_whole = val_gt_whole + val_label.tolist()
    minr = min(min(val_gt_whole), min(val_pred_whole))
    maxr = max(max(val_gt_whole), max(val_pred_whole))
    # print(maxr)
    # print(val_pred_whole)
    val_r2 = r2_score(val_gt_whole, val_pred_whole)
    print(task_name,"val r2:", val_r2)
    plt.figure(figsize=(5,5))
    plt.title("MSE loss val set R2: %.3f"%val_r2)
    plt.plot([minr, maxr],[minr, maxr],"r--")
    plt.xlabel('ground truth')
    plt.ylabel('prediction')
    plt.scatter(val_gt_whole, val_pred_whole)
    plt.savefig("%s/%s/val_r2.jpg"%(save_dir, task_name))

    #test r2 plot and accuracy
    test_pred_whole = []
    test_gt_whole = []
    # test_ratio_pred_whole = []
    # test_ratio_gt_whole = []

    with torch.no_grad():
        model.eval()
        for j, (test_data, test_data_range,test_label) in enumerate(dl_test):
            test_data=test_data.to(device)
            
            if regress_tdi==True and add_size!=0:
                test_data_range = torch.tensor(reg.predict(test_data_range)).unsqueeze(1)
            
            test_data_range=test_data_range.to(device)
            test_label=test_label.to(device)
    
            test_pred = model(test_data, test_data_range)
            # test_ratio_pred_whole = test_ratio_pred_whole + test_pred.tolist()

            # if regress_norm==True:
                # test_ratio_gt_whole = test_ratio_gt_whole + (test_label/test_data_range[:,-1]).tolist()
                # test_pred = test_data_range[:,-1]*test_pred
                # xyz_min = test_data_range[:,3] #128
                # xyz_max = test_data_range[:,4] #128
                # xr = xyz_min+test_data_range[:,0]*(xyz_max-xyz_min) #128
                # yr = xyz_min+test_data_range[:,1]*(xyz_max-xyz_min) #128
                # zr = xyz_min+test_data_range[:,2]*(xyz_max-xyz_min) #128
                # box_v = xr*yr*zr+eps #128
                # test_ratio_gt_whole = test_ratio_gt_whole + (test_label/box_v).tolist()
                # test_pred = box_v*test_pred
            # if label_log ==True:
            #     test_pred = torch.pow(10, test_pred)
            #     test_label = torch.pow(10, test_label)
            test_pred_whole = test_pred_whole + test_pred.tolist()
            test_gt_whole = test_gt_whole + test_label.tolist()
    minr = min(min(test_gt_whole), min(test_pred_whole))
    maxr = max(max(test_gt_whole), max(test_pred_whole))
    # minr2 = min(min(test_ratio_gt_whole), min(test_ratio_pred_whole))
    # maxr2 = max(max(test_ratio_gt_whole), max(test_ratio_pred_whole))

    # problem_idx = 2670
    # print("problem_test", ds_test.all_names[problem_idx], test_gt_whole[problem_idx], test_pred_whole[problem_idx])
    # print(np.argmax(np.array(test_pred_whole)), np.max(np.array(test_pred_whole)))
    # max_idx = np.argmax(np.array(test_pred_whole))
    # print(ds_test.all_names[max_idx], test_gt_whole[max_idx])
    
    test_r2 = r2_score(test_gt_whole, test_pred_whole)
    # test_ratio_r2 = r2_score(test_ratio_gt_whole, test_ratio_pred_whole)
    print(task_name,"test r2:", test_r2)
    # print("test ratio r2:", test_ratio_r2)
    
    plt.figure(figsize=(5,5))
    plt.title("test set R2: %.3f"%test_r2)
    plt.plot([minr, maxr],[minr, maxr],"r--")
    plt.xlabel('ground truth')
    plt.ylabel('prediction')
    plt.scatter(test_gt_whole, test_pred_whole)
    # plt.scatter(test_gt_whole[problem_idx], test_pred_whole[problem_idx], c="r")
    plt.savefig("%s/%s/test_r2.jpg"%(save_dir, task_name))
    plt.close()

    # plt.figure(figsize=(5,5))
    # plt.title("ratio loss test set R2: %.3f"%test_ratio_r2)
    # plt.plot([minr2, maxr2],[minr2, maxr2],"r--")
    # plt.xlabel('ground truth')
    # plt.ylabel('prediction')
    # plt.scatter(test_ratio_gt_whole, test_ratio_pred_whole)
    # plt.scatter(test_ratio_gt_whole[problem_idx], test_ratio_pred_whole[problem_idx], c="r")
    # plt.savefig("%s/%s/test_ratio_r2.jpg"%(save_dir, task_name))
    # plt.close()
    
    # if baseline is true, plot the norm term (data range -1) linear regression, if given train and validation set
    # if baseline==False:
    #     pass
    # else:
    #     xs = torch.ones(0)
    #     ys = torch.ones(0)
    #     for j, (train_data, train_data_range,train_label) in enumerate(dl):
    #         x = train_data_range[:,-1]
    #         y = train_label
    #         xs=torch.cat((xs,x))
    #         ys=torch.cat((ys,y))
    #     # print(xs.mean(), ys.mean())
    
    #     for j, (val_data, val_data_range,val_label) in enumerate(dl_val):
    #         x = val_data_range[:,-1]
    #         y = val_label
    #         xs=torch.cat((xs,x))
    #         ys=torch.cat((ys,y))
            
    #     reg = LinearRegression().fit(xs.reshape(-1,1),ys)
    #     bias = float(reg.intercept_)
    #     weight = float(reg.coef_)
    #     sklearn_pred = reg.predict(xs.reshape(-1,1))
    #     lin_train_r2 = r2_score(ys, sklearn_pred)
    #     plt.figure(figsize=(5,5))
    #     plt.title("train+val set linear regression\non normalization term: %.3f"%lin_train_r2)
    #     plt.scatter(ys, sklearn_pred)
    #     minr = min(min(ys), min(sklearn_pred))
    #     maxr = max(max(ys), max(sklearn_pred))
    #     plt.plot([minr, maxr],[minr, maxr],"r--")
    #     plt.savefig("%s/%s/train_baseline_r2.jpg"%(save_dir, task_name))
        
    
    #     xs = torch.ones(0)
    #     ys = torch.ones(0)
    #     for j, (test_data, test_data_range,test_label) in enumerate(dl_test):
    #         x = test_data_range[:,-1]
    #         y = test_label
    #         xs=torch.cat((xs,x))
    #         ys=torch.cat((ys,y))
    #     sklearn_pred = reg.predict(xs.reshape(-1,1))
    #     lin_test_r2 = r2_score(ys, sklearn_pred)
    #     plt.figure(figsize=(5,5))
    #     plt.title("test set linear regression\non normalization term: %.3f"%lin_test_r2)
    #     plt.scatter(ys, sklearn_pred)
    #     minr = min(min(ys), min(sklearn_pred))
    #     maxr = max(max(ys), max(sklearn_pred))
    #     plt.plot([minr, maxr],[minr, maxr],"r--")
    #     plt.savefig("%s/%s/test_baseline_r2.jpg"%(save_dir, task_name))
    
    
    filename = "%s/%s/output.txt"%(save_dir, task_name)
    if os.path.exists(filename):
        os.remove(filename)
    file1 = open(filename,"a")
    
    file1.write(str(train_iter_min)+" \n")
    file1.write(str(train_loss_log_avg[train_iter_min])+" \n")
    file1.write(str(val_iter_min * val_every)+" \n")
    file1.write(str(val_loss_log[val_iter_min])+" \n")
    file1.write(str(val_r2)+" \n")
    file1.write(str(test_loss_log[val_iter_min])+" \n")
    file1.write(str(test_r2)+"\n")
    if regress_tdi==True and add_size!=0:
        file1.write(str(lin_train_r2)+"\n")
        file1.write(str(lin_test_r2)+"\n")
        
    #     print("train R2 score: ", lin_train_r2)
    #     print("test R2 score: ", lin_train_r2)
    # if baseline==True:
    #     file1.write("\n\n\n\n\n\n\n")
    #     file1.write(str(lin_train_r2)+"\n")
    #     file1.write(str(lin_test_r2)+"\n")
    #     file1.write(str(weight)+"\n")
    #     file1.write(str(bias)+"\n")
    # file1.write(str(test_ratio_r2))
    
    file1.close()


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml', type=str, default='config/scratch_exp/exp79.yaml')
    args = parser.parse_args()
    run(args)