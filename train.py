import torch
import torch.nn as nn
import glob
import os
import argparse
import sys
import torchvision
from torchvision.utils import make_grid
from torch.optim import Adam, AdamW
from PIL import Image
from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

import numpy as np
import pickle
from multiprocessing import Pool, cpu_count

from utilities.model import *
from utilities.noise_scheduling import *

import json


torch.cuda.empty_cache()
torch.cuda.synchronize()

def get_args():
    parser = argparse.ArgumentParser(description="Run FUN Trainig Pipeline")

    parser.add_argument(
        "--train_data",
        type=str,
        default=None,
        help="Path to the training data (.npy arrays)"
    )

    parser.add_argument(
        '--config_file',
        type=str,
        default='config.json',
        help='Config file to determine all the training parameters'
    )

    return parser.parse_args()



def get_time_embedding(time_steps, t_emb_dim):
    factor = 10000 ** ((torch.arange(
        start = 0, end = t_emb_dim // 2, device = time_steps.device) /  (t_emb_dim // 2)
                      ))
    t_emb = time_steps[:, None].repeat(1, t_emb_dim // 2) / factor
    t_emb = torch.cat([torch.sin(t_emb), torch.cos(t_emb)], dim=-1)

    return t_emb


def wrap_angle(x):
    pi = torch.tensor(np.pi, dtype=x.dtype, device=x.device)
    return ((x+pi) % (2*pi)) - pi


def masked_smooth_l1(pred, target, mask, beta=0.1):
    diff = pred - target
    abs_diff = torch.abs(diff)
    loss = torch.where(abs_diff < beta, 0.5 * (diff ** 2) / beta, abs_diff - 0.5 * beta)
    loss = loss * mask
    return loss.sum() / (mask.sum() + 1e-8)


def load_file(file):
    arr = np.load(file).astype(np.float32)  # [64, 6]
    arr = np.transpose(arr, (1,0))         # [6,64]
    # arr is in radians in [-pi, pi]
    arr = arr / np.pi                       # now in [-1,1]
    mask = ~np.isnan(arr)
    arr = np.nan_to_num(arr, nan=0.0)
    return torch.from_numpy(arr), torch.from_numpy(mask.astype(np.float32))


class ProteinAngleDataset(Dataset):
    def __init__(self, file_list, num_workers=None):
        if num_workers is None:
            num_workers = min(cpu_count(), 4)  # Limit to 4 for laptops
        print(f"Preloading {len(file_list)} files using {num_workers} workers...")
        with Pool(num_workers) as pool:
            results = list(tqdm(pool.imap(load_file, file_list), total=len(file_list), desc="Preloading data"))
        self.data, self.masks = zip(*results)
        print(f"Preloaded {len(self.data)} samples into memory.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.masks[idx]

if __name__ == '__main__':
    device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using Device :: {device}')


    args = get_args()

    print('='*20)
    print('DDPM TRAINING PIPELINE')
    print('='*20)

    train_data_folder = args.train_data
    npy_files = glob.glob(os.path.join(train_data_folder,'*.npy'))
    
    dataset = ProteinAngleDataset(npy_files,num_workers=8)
    print('Created Dataset')
    print('Continuing to create loader object....')
    protein_loader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True)


    with open(args.config_file, "r") as f:
        config = json.load(f)


    diffusion_config = config['diffusion_params']
    dataset_config = config['dataset_params']
    model_config = config['model_params']
    train_config = config['train_params']

    scheduler = NoiseScheduler(num_timesteps=diffusion_config['num_timesteps'],
                                     schedule='linear',
                                    beta_start=diffusion_config['beta_start'],
                                    beta_end=diffusion_config['beta_end'],
                                    device='cuda')



    # Instantiate the model
    model = UNet(model_config).to(device)
    model.train()

    # Create output directory
    if not os.path.exists(train_config['task_name']):
        os.mkdir(train_config['task_name'])

    #Try to load checkpoint
    ckpt_path = os.path.join(train_config['task_name'], train_config['ckpt_name'])
    if os.path.exists(ckpt_path):
        print('Loading checkpoint as found one')
        model.load_state_dict(torch.load(ckpt_path, map_location=device))

    # Specify training parameters
    num_epochs = train_config['num_epochs']
    optimizer = AdamW(model.parameters(), lr=train_config['lr'],weight_decay=1e-4)

    # Main training loop
    for epoch_idx in range(num_epochs):
        losses = []
        pbar = tqdm(protein_loader)  # loader yields [batch, 6, 64] angle tensors
        for angles, mask in pbar:
            pbar.set_description('Preparing batch')
            optimizer.zero_grad()
            #print(angles)
            #print(mask)
            #break
            angles = angles.float().to(device)
            mask = mask.float().to(device)
            # Ensure angles are in [-pi, pi)
            angles = wrap_angle(angles)

            # Sample random noise
            noise = torch.randn_like(angles)
            # Sample timestep
            t = torch.randint(0, diffusion_config['num_timesteps'], (angles.shape[0],), device=angles.device)

            # Add noise using scheduler (noise and wrap are handled inside scheduler)
            noisy_angles = scheduler.add_noise(angles, noise, t)
            # Predict the noise
            noise_pred = model(noisy_angles, t)
            # Wrapped loss for angular data!
            loss = masked_smooth_l1(noise_pred, noise, mask)
            losses.append(loss.item())
            pbar.set_description('Backprop')
            #print(next(model.parameters()).device)
            loss.backward()
            optimizer.step()
            pbar.set_description('done_bp')
        print('Finished epoch:{} | Loss : {:.4f}'.format(
            epoch_idx + 1,
            np.mean(losses),
        ))
        torch.save(model.state_dict(), ckpt_path)

    print('Done Training ...')


            


