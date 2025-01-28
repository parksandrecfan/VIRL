
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim

import pickle
import json
import os
import random
import torch_geometric as tg
import yaml
import argparse
import shutil
from sklearn.metrics import r2_score
import math


def run(args):
    yaml_file_path = args.yaml
    # Open the YAML file
    with open(yaml_file_path, 'r') as file:
        # Load the YAML content
        config = yaml.safe_load(file)


    train_batch=config["train_batch"] #64
    test_batch=config["test_batch"] #64
    hidden = config["hidden"] #32
    lr = config["lr"] #0.001
    num_warmup_steps = config['num_warmup_steps']
    end_lr =  config['end_lr']
    gpu = config["gpu"] #2
    iteration= config["iteration"] #50000tt
    val_every= config["val_every"] #20
    save_dir = config["save_dir"]+"/"+yaml_file_path.split('/')[-1].split('.')[0] #"pretrain_experiment"
    drop = config["drop_p"] #0.2
    drop_end = config['drop_end']
    samples = config["samples"] #5000 maybe a good start
    grid_res = config["grid_res"] # 20 or 40
    splits = config['splits']
    data_root = config['data_root']
    range_root = config['range_root']
    label_root = config['label_root']

    sel_sample_pct = 0.4
    sel_sample_num = int(samples*sel_sample_pct)
    rand_sample_num = samples-sel_sample_num
    
    
    shutil.copy(yaml_file_path, save_dir+"/config.yaml")
    max_grad_norm=0.5
    
    for key, value in config.items():
        print(f'{key}: {value}')
    
    def load_item(path):
        open_file = open(path, "rb")
        this_item=pickle.load(open_file)
        open_file.close()
        return this_item
        
    def dump_item(this_item, path):
        open_file = open(path, "wb")
        pickle.dump(this_item, open_file)
        open_file.close()
        
    class deepsdf(nn.Module):
        def __init__(self):
            super(deepsdf, self).__init__()
            self.lin1 = nn.utils.weight_norm(nn.Linear(72,1024))
            self.lin2 = nn.utils.weight_norm(nn.Linear(1091,1024))
            self.lin3 = nn.utils.weight_norm(nn.Linear(1091,1024))
            self.lin4 = nn.utils.weight_norm(nn.Linear(1091,1))
            self.act = nn.LeakyReLU(negative_slope=0.001)
            
        def forward(self, data_latent): #nx72
                   
            latent2skip = data_latent[:,:,5:] #nx67
            data_latent = self.act(self.lin1(data_latent))
            data_latent = torch.cat((data_latent, latent2skip), axis=2)
            data_latent = self.act(self.lin2(data_latent))
            data_latent = torch.cat((data_latent, latent2skip), axis=2)
            data_latent = self.act(self.lin3(data_latent))
            data_latent = torch.cat((data_latent, latent2skip), axis=2)
            data_latent = self.lin4(data_latent)
            
            return data_latent[:,:,0]
    
    class BRep_sdf_ds(torch.utils.data.Dataset):
    
        def flip48(self, data_range, dist, sel_ps, flip=[False,False,False], reverse = False, rollaxis=0):
            sdf3d = dist.reshape((40,40,40))
            for i in range(3):
                if flip[i]==True:
                    sdf3d = torch.flip(sdf3d, [i])
                    flip1 = data_range[i*2].clone()
                    flip2 = data_range[i*2+1].clone()
                    data_range[i*2] = -flip2
                    data_range[i*2+1] = -flip1
                    sel_ps[:,i] = 1-(sel_ps[:,i]).clone()
                    
            if reverse==False:
                if rollaxis==0: #0, 1, 2
                    pass
                else:
                    sel_ps0 = sel_ps[:,0].clone()
                    sel_ps1 = sel_ps[:,1].clone()
                    sel_ps2 = sel_ps[:,2].clone()
                    
                    if rollaxis==1: #2, 0, 1
                        sdf3d = torch.transpose(torch.transpose(sdf3d, 1,2),0,1)
                        data_range = torch.roll(data_range, 2)
                        sel_ps[:,0] = sel_ps2
                        sel_ps[:,1] = sel_ps0
                        sel_ps[:,2] = sel_ps1
                    elif rollaxis==2: #1, 2, 0
                        sdf3d = torch.transpose(torch.transpose(sdf3d, 0,1),1,2)
                        data_range = torch.roll(data_range, 4)
                        sel_ps[:,0] = sel_ps1
                        sel_ps[:,1] = sel_ps2
                        sel_ps[:,2] = sel_ps0
                    
            elif reverse==True:
                orig_x = data_range[0:2].clone()
                orig_z = data_range[4:6].clone()
                data_range[0:2] = orig_z
                data_range[4:6] = orig_x
                sel_ps = torch.flip(sel_ps, [1])
                
                if rollaxis==0: #2, 1, 0
                    sdf3d = torch.transpose(sdf3d, 0,2)
                else:
                    sel_ps0 = sel_ps[:,0].clone()
                    sel_ps1 = sel_ps[:,1].clone()
                    sel_ps2 = sel_ps[:,2].clone()
                    
                    if rollaxis==1: # 0, 2, 1
                        sdf3d = torch.transpose(sdf3d, 1,2)
                        data_range = torch.roll(data_range, 2)
                        sel_ps[:,0] = sel_ps2
                        sel_ps[:,1] = sel_ps0
                        sel_ps[:,2] = sel_ps1
                    elif rollaxis==2: # 1, 0, 2
                        sdf3d = torch.transpose(sdf3d, 0,1)
                        data_range = torch.roll(data_range, 4)
                        sel_ps[:,0] = sel_ps1
                        sel_ps[:,1] = sel_ps2
                        sel_ps[:,2] = sel_ps0
                    
            return data_range, sdf3d.flatten(), sel_ps
        
        def __init__(
            self, 
            splits, 
            data_root, 
            range_root, 
            mode='train', 
            validate_pct=5, 
            preshuffle=True,
            label_root = 'inputs/sdf_40/dic',
            label_ext = 'pt',
            sel_sample_num = 0,
            seed = None,
            size = None):
    
            split = mode
            if mode=='validate':
                split = 'train'
    
            with open(splits, 'r') as f:
                ids = json.load(f)[split]
            
            if seed and size:
                ids = create_subset(ids, seed, size)
    
            if preshuffle and mode in ['train','validate']:
                random.shuffle(ids)
    
            if mode=='train':
                ids = [x for i,x in enumerate(ids) if i % 100 <= (100 - validate_pct)]
                self.aug = True
            elif mode=='validate':
                ids = [x for i,x in enumerate(ids) if i % 100 > (100 - validate_pct)]
                self.aug=False
                
            self.all_paths = [os.path.join(data_root, f'{id}.pt') for id in ids]
            # self.all_dics = [os.path.join(dic_root, f'{id}.pt') for id in ids] #
            self.all_labels = [os.path.join(label_root, f'{id}.{label_ext}') for id in ids]
            self.all_ranges = [os.path.join(range_root, f'{id}.pt') for id in ids]
            self.all_names = ids
            self.sel_sample_num=sel_sample_num    
                
        def __getitem__(self, idx):
            data = torch.load(self.all_paths[idx])        
            data['face_ct'] = int(data['num_faces'])
            del data['largest_face']
            del data['sdf_max']
            del data['curve_max']
            del data['surf_max']
            data_range = torch.load(self.all_ranges[idx])  # 48 types
            dist_label = load_item(self.all_labels[idx])   
            
            dist = torch.tensor(dist_label["dist"]).float() #for output #48 types
            bound = torch.tensor(np.array(dist_label["bound"])).float()[0] #for output
            sel_idxs = torch.randint(low=0, high=bound.shape[0], size=(self.sel_sample_num,))
            sel_bds = bound[sel_idxs].int() # the actual index of bounds
            sel_sdfs = dist[sel_bds]
    
            z = sel_bds%grid_res
            y = (sel_bds//grid_res)%grid_res
            x = (sel_bds//(grid_res**2))%grid_res
            sel_ps = torch.stack((x,y,z)).T/(grid_res-1)

            if self.aug==True:
                flip = torch.tensor([bool(random.getrandbits(1)) for i in range(3)])
                reverse = torch.tensor(bool(random.getrandbits(1)))
                rollaxis=torch.tensor(random.randint(0,2))

            elif self.aug==False:
                flip = torch.tensor((False, False, False))
                reverse = torch.tensor(False)
                rollaxis = torch.tensor(0)
                
            data_range, dist, sel_ps = self.flip48(data_range, dist, sel_ps, flip, reverse, rollaxis)
            
            return data, data_range, dist, sel_ps, sel_sdfs, flip, reverse, rollaxis
    
        def __len__(self):
            return len(self.all_paths)
    
    class BRepFaceEncoder(nn.Module):
        def __init__(self, hidden_size):
            super().__init__()
            self.v_in = LinearBlock(3, hidden_size)
            self.e_in = LinearBlock(15, hidden_size)
            self.l_in = LinearBlock(10, hidden_size)
            self.f_in = LinearBlock(17, hidden_size)
            self.v_to_e = BipartiteResMRConv(hidden_size)
            self.e_to_l = BipartiteResMRConv(hidden_size)
            self.l_to_f = BipartiteResMRConv(hidden_size)
            #newly added
            self.drop = drop
            self.drop_layer = nn.Dropout(p=self.drop)
            self.proj = LinearBlock(hidden_size, 64)
        def forward(self, data):
            v = self.v_in(data.vertex_positions)
            e = torch.cat([data.edge_curves, data.edge_curve_parameters, data.edge_curve_flipped.reshape((-1,1))], dim=1)
            e = self.e_in(e)
            l = self.l_in(data.loop_types.float())
            f = torch.cat([data.face_surfaces, data.face_surface_parameters, data.face_surface_flipped.reshape((-1,1))], dim=1)
            f = self.f_in(f)

            e = self.v_to_e(v, e, data.edge_to_vertex[[1,0]])
            e = self.drop_layer(e)
            l = self.e_to_l(e, l, data.loop_to_edge[[1,0]])
            l = self.drop_layer(l)
            f = self.l_to_f(l, f, data.face_to_loop[[1,0]])
            f = self.drop_layer(f)
            return self.proj(f)

    class deepbox(nn.Module):
        def __init__(self):
            super(deepbox, self).__init__()
            self.lin1 = nn.utils.weight_norm(nn.Linear(69,64)) #64+3(UVW)
            self.lin2 = nn.utils.weight_norm(nn.Linear(128,64))
            self.lin3 = nn.utils.weight_norm(nn.Linear(128,3))
            self.act = nn.LeakyReLU(negative_slope=0.001)
            
        def forward(self, pooled_data):
            skip_data = pooled_data[:,5:]
            data_latent = pooled_data
            data_latent = self.act(self.lin1(data_latent))
            data_latent = torch.cat((data_latent, skip_data), axis=1)
            data_latent = self.act(self.lin2(data_latent))
            data_latent = torch.cat((data_latent, skip_data), axis=1)
            data_latent = self.act(self.lin3(data_latent))
            return data_latent
            
    class BRepVolAutoencoder(nn.Module):
        def __init__(self, code_size, device=0):
            super().__init__()
            self.encoder = BRepFaceEncoder(code_size)
            self.decoder1 = deepbox()
            self.decoder2 = deepsdf()
        
        def forward(self, data, uvw, aug):            
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
                pooled_data[i] = pooled_segment
            
            pooled_data_aug = torch.cat((aug, pooled_data), axis=1) #batchx64+batchx5
            box = self.decoder1(pooled_data_aug) #xr, yr, zr
            '''scale input here'''
            batch_aug = aug.unsqueeze(1).repeat(1,uvw.shape[1],1)
            scaled_uvw = torch.einsum('abc,ac->abc',uvw, box) #-0.5~0.5->-10~10
            batch_latent = pooled_data.unsqueeze(1).repeat(1,scaled_uvw.shape[1],1)
            data_latent = torch.cat((batch_aug, batch_latent, uvw), axis=2).float() #5+64+3
            
            pred = self.decoder2(data_latent)
            return pred, box
    
    def cosine_lr_schedule(
        num_warmup_steps: int,
        num_training_steps: int,
        min_lr: float,
        max_lr: float,
    ):
        def get_lr(t: int) -> float:
            # this function outputs a function with the parameters baked into it. 
            """Outputs the learning rate at step t under the cosine schedule.
    
            Args:
                t: the current step number
    
            Returns:
                lr: learning rate at step t
    
            Hint: Question 2.1
            """
    
            assert max_lr >= min_lr >= 0.0
            assert num_training_steps >= num_warmup_steps >= 0
    
            if t < num_warmup_steps:
                # Warm-up phase: Linearly increase learning rate from min_lr to max_lr
                lr = min_lr + (max_lr - min_lr) * t / num_warmup_steps
            elif t < num_training_steps:
                # Cosine annealing phase: Decrease learning rate using a cosine schedule
                progress = (t - num_warmup_steps) / (num_training_steps - num_warmup_steps)
                lr = max_lr - 0.5 * (1.0 - math.cos(progress * math.pi)) * (max_lr - min_lr)
            else:  # t >= num_training_steps
                lr = min_lr
    
            return lr
    
        return get_lr
    
    def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
        for g in optimizer.param_groups:
            g["lr"] = lr

    def get_truth(dist, data_range, samples, grid_res, device, mode): #dist, range already in device, others not
        batch = dist.shape[0]
        
        if mode=='random':
            
            dist_4d = dist.reshape((batch, grid_res, grid_res, grid_res))    
            nx = torch.randint(low=0, high=grid_res-1, size=(batch, samples)).to(device)
            rx = torch.rand((batch, samples)).to(device)
            ny = torch.randint(low=0, high=grid_res-1, size=(batch, samples)).to(device)
            ry = torch.rand((batch, samples)).to(device)
            nz = torch.randint(low=0, high=grid_res-1, size=(batch, samples)).to(device)
            rz = torch.rand((batch, samples)).to(device)
    
            # random 100 points
            ps = torch.stack((nx+rx, ny+ry, nz+rz)).permute(1, 2, 0) # 8x100x3
            
            p0s = torch.stack((nx, ny, nz)).permute(1, 2, 0) #8x100x3
            p1s= torch.stack((nx+1, ny, nz)).permute(1, 2, 0)
            p2s= torch.stack((nx, ny+1, nz)).permute(1, 2, 0)
            p3s= torch.stack((nx+1, ny+1, nz)).permute(1, 2, 0)
            p4s= torch.stack((nx, ny, nz+1)).permute(1, 2, 0)
            p5s= torch.stack((nx+1, ny, nz+1)).permute(1, 2, 0)
            p6s= torch.stack((nx, ny+1, nz+1)).permute(1, 2, 0)
            p7s= torch.stack((nx+1, ny+1, nz+1)).permute(1, 2, 0)
            
            v0s = torch.stack([dist_4d[i,:,:,:][p0s[i,:,0],p0s[i,:,1],p0s[i,:,2]] for i in range(dist_4d.shape[0])]) #8x100
            v1s = torch.stack([dist_4d[i,:,:,:][p1s[i,:,0],p1s[i,:,1],p1s[i,:,2]] for i in range(dist_4d.shape[0])]) #8x100
            v2s = torch.stack([dist_4d[i,:,:,:][p2s[i,:,0],p2s[i,:,1],p2s[i,:,2]] for i in range(dist_4d.shape[0])]) #8x100
            v3s = torch.stack([dist_4d[i,:,:,:][p3s[i,:,0],p3s[i,:,1],p3s[i,:,2]] for i in range(dist_4d.shape[0])]) #8x100
            v4s = torch.stack([dist_4d[i,:,:,:][p4s[i,:,0],p4s[i,:,1],p4s[i,:,2]] for i in range(dist_4d.shape[0])]) #8x100
            v5s = torch.stack([dist_4d[i,:,:,:][p5s[i,:,0],p5s[i,:,1],p5s[i,:,2]] for i in range(dist_4d.shape[0])]) #8x100
            v6s = torch.stack([dist_4d[i,:,:,:][p6s[i,:,0],p6s[i,:,1],p6s[i,:,2]] for i in range(dist_4d.shape[0])]) #8x100
            v7s = torch.stack([dist_4d[i,:,:,:][p7s[i,:,0],p7s[i,:,1],p7s[i,:,2]] for i in range(dist_4d.shape[0])]) #8x100
            
            # combine same xy diff z
            l0s = (1-rz)*v0s+rz*v4s #nx ny
            l1s = (1-rz)*v1s+rz*v5s #nx+1 ny
            l2s = (1-rz)*v2s+rz*v6s #nx ny+1
            l3s = (1-rz)*v3s+rz*v7s #nx+1 ny+1
            
            # combine same x diff y
            f0s = (1-ry)*l0s+ry*l2s
            f1s = (1-ry)*l1s+ry*l3s
            
            # values
            vs = (1-rx)*f0s+rx*f1s #8x100
        
        elif mode=='uniform': #ignore samples, just take the 20 points
            x = torch.arange(grid_res).float().to(device)
            y = torch.arange(grid_res).float().to(device)
            z = torch.arange(grid_res).float().to(device)
            x_grid, y_grid, z_grid = torch.meshgrid(x, y, z)
            nx = x_grid.flatten().repeat(batch, 1) #batchx8000
            ny = y_grid.flatten().repeat(batch, 1)
            nz = z_grid.flatten().repeat(batch, 1)
            ps = torch.stack((nx, ny, nz)).permute(1, 2, 0) # batch*8000*3
            vs = dist #batchx8000
    
    
        xmin = data_range[:,0]
        xmax = data_range[:,1]
        ymin = data_range[:,2]
        ymax = data_range[:,3]
        zmin = data_range[:,4]
        zmax = data_range[:,5]
        
        xr = xmax-xmin
        yr = ymax-ymin
        zr = zmax-zmin
    
        truth = vs
        hwl = torch.stack((xr, yr, zr)).T # only for plotting
        
        norm_ps = ps/(grid_res-1)
        return norm_ps, truth, hwl #ps is uvw, set to 0~19. norm_ps is 0~1
        
            
    import sys
    sys.path.append('automatemain')
    # print(os.getcwd())
    from automate.sbgcn import LinearBlock, BipartiteResMRConv 
    sys.path.remove('automatemain')
    
    ds = BRep_sdf_ds(splits=splits, data_root=data_root, \
                    range_root = range_root, mode='train', validate_pct=5, preshuffle=False,\
                    label_root = label_root, label_ext = 'pt',sel_sample_num=sel_sample_num)
    ds_val = BRep_sdf_ds(splits=splits, data_root=data_root, \
                         range_root = range_root, mode='validate', validate_pct=5, preshuffle=False,\
                         label_root = label_root, label_ext = 'pt', sel_sample_num=sel_sample_num)

    print(f'Train Set size = {len(ds)}')
    print(f'Val Set Size = {len(ds_val)}')
    dl = tg.loader.DataLoader(ds, batch_size=train_batch, shuffle=True, num_workers=8, persistent_workers=True)
    dl_val = tg.loader.DataLoader(ds_val, batch_size=test_batch, shuffle=False, num_workers=8, persistent_workers=True)
    
    device = 'cuda:%s'%gpu if torch.cuda.is_available() else 'cpu'
    print("Using device:", device)


    
    model = BRepVolAutoencoder(hidden, device=device).to(device)
    print(model)

    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    print(f"encoder learnable parameters: {encoder_params}")
    decoder_params = sum(p.numel() for p in model.decoder1.parameters())+sum(p.numel() for p in model.decoder2.parameters())
    print(f"decoder learnable parameters: {decoder_params}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"total learnable parameters: {total_params}")
    
    optimizer = optim.Adam(model.parameters(), lr=lr)

    
    lr_schedule = cosine_lr_schedule(num_warmup_steps, iteration, end_lr, lr)
    
    train_loss_log=[]
    train_loss_log_avg=[]
    test_loss_log=[]
    
    itering = iter(dl)
    batch_ct = 1
    
    for i in range(iteration):
        # warmup+cosine anneal
        lr = lr_schedule(i)
        set_lr(optimizer, lr)
        
        model.train()
        optimizer.zero_grad()
        data, data_range, dist, sel_ps, sel_sdfs, flip, reverse, rollaxis = next(itering)
        aug = torch.cat((flip, reverse.unsqueeze(1), rollaxis.unsqueeze(1)), dim=1)
        
        if (data_range.shape[0]<train_batch) == True:
            # print("%s batch done"%batch_ct)
            itering = iter(dl)
            batch_ct+=1
        
        data=data.to(device)
        data_range=data_range.to(device)
        dist=dist.to(device)
        sel_ps = sel_ps.to(device)
        sel_sdfs =  sel_sdfs.to(device)
        aug = aug.to(device)
        
        uvw, truth, hwl= get_truth(dist, data_range, rand_sample_num, grid_res, device, mode='random') # batch * points# * 4
        sel_truth=sel_sdfs
        
        truth_all = torch.cat((sel_truth, truth),1) # 4->1
        uvw_all = torch.cat((sel_ps, uvw),1)
        # print(uvw_all.shape, aug_batch.shape) batch x samples x 3 or 5
        pred, box = model(data, uvw_all, aug) #batch*points*4

        recon_loss=torch.mean((pred-truth_all)**2)
        box_loss=torch.mean((box-hwl)**2)
        train_loss=recon_loss+box_loss
        train_loss.backward()
        optimizer.step()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        if i<drop_end:
            new_drop=drop*(1-i/drop_end)
            model.encoder.drop_layer = nn.Dropout(p=new_drop)
        else:
            model.encoder.drop_layer = nn.Dropout(p=0)
        
        
        train_loss_log.append(train_loss.item())
        if i<val_every:
            train_loss_log_avg.append(sum(train_loss_log)/len(train_loss_log))
        else:
            train_loss_log_avg.append((sum(train_loss_log[-val_every:]))/val_every)
        dump_item(train_loss_log_avg, "%s/train_loss_log.pkl"%save_dir)
        
        if i%val_every==0:
            with torch.no_grad():
                # get validation loss
                model.eval()
                
                test_loss_recon=0
                test_loss_box=0
                
                for j, (val_data, val_data_range,val_dist, val_sel_ps, val_sel_sdfs, val_flip,val_reverse, val_rollaxis) in enumerate(dl_val):
                    
                    val_aug = torch.cat((val_flip, val_reverse.unsqueeze(1), val_rollaxis.unsqueeze(1)), dim=1)
        
                    val_data=val_data.to(device)
                    val_data_range=val_data_range.to(device)
                    val_dist=val_dist.to(device)
                    val_sel_ps = val_sel_ps.to(device)
                    val_sel_sdfs =  val_sel_sdfs.to(device)
                    val_aug = val_aug.to(device)
                    '''truth dont need to calculate real xyz'''
                    val_uvw, val_truth, val_hwl = get_truth(val_dist, val_data_range, rand_sample_num, grid_res, device, mode='random') # batch * points# * 4
                    val_sel_truth = val_sel_sdfs
                    val_truth_all = torch.cat((val_sel_truth, val_truth),1)
                    val_uvw_all = torch.cat((val_sel_ps, val_uvw),1)
                    
                    val_pred, val_box = model(val_data, val_uvw_all, val_aug)
                    val_recon_loss = torch.sum((val_pred-val_truth_all)**2)/samples
                    val_box_loss=torch.sum((val_box-val_hwl)**2)/3

                    test_loss_recon+=val_recon_loss.item()
                    test_loss_box+=val_box_loss.item()
                    
                test_loss_recon=test_loss_recon/len(ds_val.all_names)
                test_loss_box=test_loss_box/len(ds_val.all_names)
                test_loss=test_loss_box+test_loss_recon
                
                test_loss_log.append(test_loss)
                plt.figure(figsize=(8,6))
                plt.plot(train_loss_log_avg)
                val_loss_log2 = [i for i in test_loss_log for j in range(val_every)]
                plt.yscale('log')
                plt.plot(val_loss_log2)
                plt.savefig("%s/loss_plot.jpg"%save_dir)
                plt.close()
                
                print("iter: %s, train: %.2f, test: %.2f recon: %.2f box %.2f"%(i, train_loss_log_avg[-1], test_loss, test_loss_recon, test_loss_box))
                if test_loss==min(test_loss_log):
                    print('min')
                    torch.save(model.state_dict(),"%s/%s_iter.pt"%(save_dir, i))
                dump_item(test_loss_log, "%s/test_loss_log.pkl"%save_dir)
    
    torch.save(model.state_dict(),"%s/%s_iter.pt"%(save_dir, i))

    print("test loss:", test_loss_log[-1])
    train_iter_min = np.argmin(np.array(train_loss_log_avg))
    print("best train iter:", train_iter_min)
    print("best train loss:", train_loss_log_avg[train_iter_min])
    test_iter_min = np.argmin(np.array(test_loss_log))
    print("best test iter:", test_iter_min * val_every)
    print("best test loss:", test_loss_log[test_iter_min])
    
    plt.figure(figsize=(8,6))
    plt.plot(train_loss_log_avg)
    # val_every=20
    val_loss_log2 = [i for i in test_loss_log for j in range(val_every)]
    plt.yscale('log')
    plt.plot(val_loss_log2)
    plt.savefig("%s/loss_plot.jpg"%save_dir)
    plt.close()


    filename = "%s/output.txt"%(save_dir)
    if os.path.exists(filename):
        os.remove(filename)
    file1 = open(filename,"a")

    file1.write(str(train_iter_min)+" \n")
    file1.write(str(train_loss_log_avg[train_iter_min])+" \n")
    file1.write(str(test_iter_min * val_every) + "\n")
    file1.write(str(test_loss_log[test_iter_min]) + "\n")    
    file1.close()

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml', type=str, default='pretrain.yaml')
    args = parser.parse_args()
    run(args)