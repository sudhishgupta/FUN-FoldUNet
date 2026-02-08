import torch
import torch.nn as nn
import glob
import os
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


torch.cuda.empty_cache()
torch.cuda.synchronize()



# class LinearNoiseScheduler:
#     def __init__(self, num_timesteps, beta_start, beta_end, device='cuda'):
#         self.device = torch.device(device)
#         self.num_timesteps = num_timesteps
#         self.beta_start = beta_start
#         self.beta_end = beta_end


#         self.betas = torch.linspace(beta_start, beta_end, num_timesteps, device=self.device)            #array having [b1 ..... bn]
#         self.alphas = 1. - self.betas                                               #array having [a1 ..... an]
#         self.alpha_cum_prod = torch.cumprod(self.alphas, dim=0)                     #cumulative product of alpha_i's
#         self.sqrt_alpha_cum_prod = torch.sqrt(self.alpha_cum_prod)                  #sqrt of cumulatuve product
#         self.sqrt_one_minus_alpha_cum_prod = torch.sqrt(1. - self.alpha_cum_prod)   #sqrt of (1-cumulative product of alpha_i's)


#     # FORWARD DIFFUSION PROCESS

#     def add_noise(self, original, noise, t):

#         t = t.to(self.device)
#         original_shape = original.shape                                               #shape of image  B X C X H X W
#         batch_size = original_shape[0]                                               #batch-size

#         sqrt_alpha_cum_prod = self.sqrt_alpha_cum_prod[t].reshape(batch_size)                           #reshaping to match batch-size
#         sqrt_one_minus_alpha_cum_prod = self.sqrt_one_minus_alpha_cum_prod[t].reshape(batch_size)       #reshaping to match batch-size

#         for _ in range(len(original_shape)-1):
#             sqrt_alpha_cum_prod = sqrt_alpha_cum_prod.unsqueeze(-1)                                    #unsqueezing the batch-dim to make 2d tensor
#             sqrt_one_minus_alpha_cum_prod = sqrt_one_minus_alpha_cum_prod.unsqueeze(-1)                #unsqueezing the batch-dim to make 2d tensor

#         xt = sqrt_alpha_cum_prod*original + sqrt_one_minus_alpha_cum_prod*noise
#         return xt
        

#     # REVERSE DIFFUSION PROCESS

#     # def sample_prev_timestep(self, xt, noise_pred, t):

#     #     t = torch.tensor(t, device=self.device) if not torch.is_tensor(t) else t.to(self.device)

#     #     #original image saving for visualization
#     #     x0 = (xt - (self.sqrt_one_minus_alpha_cum_prod[t]*noise_pred)) / self.sqrt_alpha_cum_prod[t]
#     #     x0 = torch.clamp(x0, -1., 1.)

#     #     #computing mean at timestep t for sampling

#     #     mean = xt - ((self.betas[t] * noise_pred) / (self.sqrt_one_minus_alpha_cum_prod[t]))
#     #     mean = mean / torch.sqrt(self.alphas[t])


#     #     # at timestep 0 noise is not added
#     #     if t == 0:
#     #         mean_wrapped = wrap_angle(mean)
#     #         x0_wrapped = wrap_angle(x0)
#     #         return mean_wrapped,x0_wrapped
#     #     else:
#     #         variance = (1. - self.alpha_cum_prod[t-1]) / (1. - self.alpha_cum_prod[t])
#     #         variance = variance * self.betas[t]
#     #         sigma = variance ** 0.5
#     #         z = torch.randn(xt.shape).to(xt.device)

#     #         sample = mean + sigma * z
            
#     #         sample_wrapped = wrap_angle(sample)
#     #         x0_wrapped = wrap_angle(x0)

#     #         return sample_wrapped, x0_wrapped
#     def sample_prev_timestep(self, xt, noise_pred, t):
#         t = torch.tensor(t, device=self.device) if not torch.is_tensor(t) else t.to(self.device)
#         # predict x0
#         x0_pred = (xt - (self.sqrt_one_minus_alpha_cum_prod[t] * noise_pred)) / self.sqrt_alpha_cum_prod[t]
#         # DO NOT clamp x0_pred here (avoid bias)
#         # compute mean
#         mean = xt - ((self.betas[t] * noise_pred) / (self.sqrt_one_minus_alpha_cum_prod[t]))
#         mean = mean / torch.sqrt(self.alphas[t])

#         if t == 0:
#             return mean, x0_pred
#         else:
#             variance = (1. - self.alpha_cum_prod[t-1]) / (1. - self.alpha_cum_prod[t])
#             variance = variance * self.betas[t]
#             sigma = torch.sqrt(variance)
#             z = torch.randn(xt.shape, device=xt.device)
#             sample = mean + sigma * z
#             return sample, x0_pred



def get_time_embedding(time_steps, t_emb_dim):
    factor = 10000 ** ((torch.arange(
        start = 0, end = t_emb_dim // 2, device = time_steps.device) /  (t_emb_dim // 2)
                      ))
    t_emb = time_steps[:, None].repeat(1, t_emb_dim // 2) / factor
    t_emb = torch.cat([torch.sin(t_emb), torch.cos(t_emb)], dim=-1)

    return t_emb


class LinearNoiseScheduler:
    """
    Noise scheduler supporting 'linear' and 'cosine' schedules.

    Usage:
        scheduler = LinearNoiseScheduler(
            num_timesteps=1000,
            beta_start=0.0001,      # only used for linear schedule
            beta_end=0.02,          # only used for linear schedule
            schedule='cosine',      # or 'linear'
            s=0.008,                # small offset for cosine schedule
            device='cuda'
        )
    """
    def __init__(self, num_timesteps, beta_start=1e-4, beta_end=2e-2,
                 schedule='linear', s=0.008, device='cuda'):
        self.device = torch.device(device)
        self.num_timesteps = int(num_timesteps)
        self.schedule = schedule.lower()
        self.s = float(s)

        if self.schedule == 'linear':
            # original linear schedule (kept for compatibility)
            betas = torch.linspace(beta_start, beta_end, self.num_timesteps, device=self.device)
        elif self.schedule == 'cosine':
            betas = self._cosine_betas(self.num_timesteps, s=self.s, device=self.device)
        else:
            raise ValueError(f"Unknown schedule: {schedule}. Choose 'linear' or 'cosine'.")

        # clip betas into (0,1)
        betas = torch.clamp(betas, min=1e-8, max=0.999)

        self.betas = betas
        self.alphas = 1.0 - self.betas
        self.alpha_cum_prod = torch.cumprod(self.alphas, dim=0)  # \bar\alpha_t
        self.sqrt_alpha_cum_prod = torch.sqrt(self.alpha_cum_prod)
        self.sqrt_one_minus_alpha_cum_prod = torch.sqrt(1.0 - self.alpha_cum_prod)

    def _cosine_betas(self, timesteps, s=0.008, device='cpu'):
        """
        Cosine schedule (implementation from Nichol & Dhariwal / common libs).
        Returns a torch.tensor of shape [timesteps] on specified device.
        """
        steps = timesteps + 1
        # Use numpy for stable float ops, then convert to torch
        t = np.linspace(0, timesteps, steps, dtype=np.float64)
        f = np.cos(((t / timesteps) + s) / (1 + s) * (np.pi / 2.0)) ** 2
        alpha_cumprod = f / f[0]
        # betas are 1 - alpha_cumprod[t] / alpha_cumprod[t-1]
        betas = 1.0 - (alpha_cumprod[1:] / alpha_cumprod[:-1])
        betas = np.clip(betas, a_min=0.0, a_max=0.999)
        return torch.tensor(betas, dtype=torch.float32, device=device)

    # FORWARD DIFFUSION PROCESS
    def add_noise(self, original, noise, t):
        t = t.to(self.device)
        original_shape = original.shape
        batch_size = original_shape[0]

        sqrt_alpha_cum_prod = self.sqrt_alpha_cum_prod[t].reshape(batch_size)
        sqrt_one_minus_alpha_cum_prod = self.sqrt_one_minus_alpha_cum_prod[t].reshape(batch_size)

        for _ in range(len(original_shape) - 1):
            sqrt_alpha_cum_prod = sqrt_alpha_cum_prod.unsqueeze(-1)
            sqrt_one_minus_alpha_cum_prod = sqrt_one_minus_alpha_cum_prod.unsqueeze(-1)

        xt = sqrt_alpha_cum_prod * original + sqrt_one_minus_alpha_cum_prod * noise
        return xt

    # REVERSE DIFFUSION PROCESS
    def sample_prev_timestep(self, xt, noise_pred, t):
        t = torch.tensor(t, device=self.device) if not torch.is_tensor(t) else t.to(self.device)
        # predict x0
        x0_pred = (xt - (self.sqrt_one_minus_alpha_cum_prod[t] * noise_pred)) / (self.sqrt_alpha_cum_prod[t] + 1e-12)
        # compute mean
        mean = xt - ((self.betas[t] * noise_pred) / (self.sqrt_one_minus_alpha_cum_prod[t] + 1e-12))
        mean = mean / torch.sqrt(self.alphas[t] + 1e-12)

        if int(t.item()) == 0:
            return mean, x0_pred
        else:
            # variance according to DDPM paper
            variance = (1.0 - self.alpha_cum_prod[t - 1]) / (1.0 - self.alpha_cum_prod[t])
            variance = variance * self.betas[t]
            sigma = torch.sqrt(variance + 1e-12)
            z = torch.randn(xt.shape, device=xt.device)
            sample = mean + sigma * z
            return sample, x0_pred




#### UNET ####
# shape = [batch, channels=6, seq_length=64]




class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, t_emb_dim, down_sample, num_heads):
        super().__init__()

        self.down_sample = down_sample
        self.resnet_conv_first = nn.Sequential(
            nn.GroupNorm(8, in_channels),
            nn.SiLU(),
            nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        )
        self.t_emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(t_emb_dim, out_channels)
        )
        self.resnet_conv_second = nn.Sequential(
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        )
        self.attention_norm = nn.GroupNorm(8, out_channels)
        self.attention = nn.MultiheadAttention(out_channels, num_heads, batch_first=True)

        self.residual_input_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)               # 1x1 conv1d layer for makiing sure residual connections work well !

        self.down_sample_conv = nn.Conv1d(out_channels, out_channels, kernel_size=4,
                                          stride=2, padding=1) if self.down_sample else nn.Identity()
        


    def forward(self, x, t_emb):
        out = x
        #Resnet-Block
        resnet_input = out
        out = self.resnet_conv_first(out)
        # out = out + self.t_emb_layers(t_emb)[:, :, None, None]
        out = out + self.t_emb_layers(t_emb).unsqueeze(-1)       # ← correct shape!                    ##adding t_emb across channels of the output from 1st resnet block
        out = self.resnet_conv_second(out)
        out = out + self.residual_input_conv(resnet_input)                       #residual-1

        #Attention Block
        #batch_size, channels, h, w = out.shape
        in_attn = self.attention_norm(out)
        in_attn = in_attn.permute(0,2,1)
        #in_attn = in_attn.transpose(1,2)                               #transpose to make sure channels are the feature dimension for attention
        out_attn, _ = self.attention(in_attn, in_attn, in_attn)        #Q=K=V :: self-attention
        out_attn = out_attn.permute(0,2,1)
        #out_attn = out_attn.transpose(1,2).reshape(batch_size, channels, h,w)           # back to orignal size by transpose-->reshape

        out = out+out_attn

        #we can loop over this entire combination of 2-blocks if multiple layers needed
        #but here we will go forward with just one layer

        out = self.down_sample_conv(out)

        return out
        


class MidBlock(nn.Module):
    def __init__(self, in_channels, out_channels, t_emb_dim, num_heads):
        super().__init__()
        self.resnet_conv_first = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(8, in_channels),
                nn.SiLU(),
                nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            ),
            nn.Sequential(
                nn.GroupNorm(8, out_channels),
                nn.SiLU(),
                nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
            )
        ])

        self.t_emb_layers = nn.ModuleList([
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(t_emb_dim, out_channels)
            ),
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(t_emb_dim, out_channels)
            )
        ])

        self.resnet_conv_second = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(8, out_channels),
                nn.SiLU(),
                nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
            ),
            nn.Sequential(
                nn.GroupNorm(8,out_channels),
                nn.SiLU(),
                nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
            )
        ])

        self.attention_norm = nn.GroupNorm(8, out_channels)
        self.attention = nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
        self.residual_input_conv = nn.ModuleList([
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
            nn.Conv1d(out_channels, out_channels, kernel_size=1)
        ])

    def forward(self, x, t_emb):
        out = x

        #first resnet-block
        resnet_input = out
        out = self.resnet_conv_first[0](out)
        # out = out + self.t_emb_layers(t_emb)[:, :, None, None]
        out = out + self.t_emb_layers[0](t_emb).unsqueeze(-1)       # ← correct shape!
        out = self.resnet_conv_second[0](out)
        out = out + self.residual_input_conv[0](resnet_input)


        #Attention Block
        #batch_size, channels, h, w = out.shape
        in_attn = self.attention_norm(out)
        in_attn = in_attn.permute(0,2,1)
        #in_attn = in_attn.transpose(1,2)                               #transpose to make sure channels are the feature dimension for attention
        out_attn, _ = self.attention(in_attn, in_attn, in_attn)        #Q=K=V :: self-attention
        out_attn = out_attn.permute(0,2,1)
        #out_attn = out_attn.transpose(1,2).reshape(batch_size, channels, h,w)           # back to orignal size by transpose-->reshape

        out = out+out_attn


        #second resnet block
        resnet_input = out
        out = self.resnet_conv_first[1](out)
        # out = out + self.t_emb_layers(t_emb)[:, :, None, None]
        out = out + self.t_emb_layers[1](t_emb).unsqueeze(-1)       # ← correct shape!
        out = self.resnet_conv_second[1](out)
        out = out + self.residual_input_conv[1](resnet_input)


        #if multiple layers then self-attention + second resnet block will have a loop

        return out

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, t_emb_dim, up_sample, num_heads):
        super().__init__()
        self.up_sample = up_sample
        self.resnet_conv_first = nn.Sequential(
            nn.GroupNorm(8, in_channels),
            nn.SiLU(),
            nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        )
        self.t_emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(t_emb_dim, out_channels)
        )
        self.resnet_conv_second = nn.Sequential(
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        )

        self.attention_norm = nn.GroupNorm(8, out_channels)
        self.attention = nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
        self.residual_input_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.up_sample_conv = nn.ConvTranspose1d(in_channels//2, in_channels //2, kernel_size=4,
                                                 stride=2, padding=1) if self.up_sample else nn.Identity()

    def forward(self, x, out_down,t_emb):

        #same as downblock but with some key changes

        #upsample the input and concat with corresponding downblock output
        x = self.up_sample_conv(x)
        x = torch.cat(tensors=[x, out_down], dim=1)
        #other approach : concat-->resnet--->self-attn--->upsample
        

        #resnet block
        out = x
        resnet_input = out
        out = self.resnet_conv_first(out)
        # out = out + self.t_emb_layers(t_emb)[:, :, None, None]
        out = out + self.t_emb_layers(t_emb).unsqueeze(-1)       # ← correct shape!
        out = self.resnet_conv_second(out)
        out = out + self.residual_input_conv(resnet_input)

        #Attention Block
        #batch_size, channels, h, w = out.shape
        in_attn = self.attention_norm(out)
        in_attn = in_attn.permute(0,2,1)
        
        #in_attn = in_attn.transpose(1,2)                               #transpose to make sure channels are the feature dimension for attention
        out_attn, _ = self.attention(in_attn, in_attn, in_attn)        #Q=K=V :: self-attention
        out_attn = out_attn.permute(0,2,1)
        #out_attn = out_attn.transpose(1,2).reshape(batch_size, channels, h,w)           # back to orignal size by transpose-->reshape

        out = out+out_attn

        return out


#final UNet Model Class

class UNet(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.im_channels = model_config['im_channels']
        self.down_channels = model_config['down_channels']                     #converts i channels to i+1 channels (total 3 downsample blocks)
        self.mid_channels = model_config['mid_channels']
        self.t_emb_dim = model_config['time_emb_dim']
        self.num_heads = model_config['num_heads']
        self.down_sample = [True, True, False]

        #this timestep projection module will only be called once to get the initial timestep representations
        self.t_proj = nn.Sequential(
            nn.Linear(self.t_emb_dim, self.t_emb_dim),
            nn.SiLU(),
            nn.Linear(self.t_emb_dim, self.t_emb_dim)
        )


        #input to have same number of channels
        self.up_sample = list(reversed(self.down_sample))
        self.conv_in = nn.Conv1d(self.im_channels, self.down_channels[0], kernel_size=3, padding=1)


        self.downs = nn.ModuleList([])
        for i in range(len(self.down_channels)-1):
            self.downs.append(DownBlock(self.down_channels[i], self.down_channels[i+1], self.t_emb_dim,
                             down_sample=self.down_sample[i], num_heads=self.num_heads))

        
        self.mids = nn.ModuleList([])
        for i in range(len(self.mid_channels)-1):
            self.mids.append(MidBlock(self.mid_channels[i], self.mid_channels[i+1], self.t_emb_dim, num_heads=self.num_heads))
            
        self.ups = nn.ModuleList([])
        for i in reversed(range(len(self.down_channels)-1)):
            self.ups.append(UpBlock(self.down_channels[i]*2, self.down_channels[i-1] if i != 0 else 16,
                                    self.t_emb_dim, up_sample=self.down_sample[i], num_heads=self.num_heads))

        self.norm_out = nn.GroupNorm(8,16)
        self.conv_out = nn.Conv1d(16, self.im_channels, kernel_size=3, padding=1)


    def forward(self, x, t):
        out = self.conv_in(x)
        t_emb = get_time_embedding(t, self.t_emb_dim)
        t_emb = self.t_proj(t_emb)

        down_outs = []

        for down in self.downs:
            #print(out.shape)
            down_outs.append(out)
            out = down(out, t_emb)

        for mid in self.mids:
            #print(out.shape)
            out = mid(out, t_emb)


        for up in self.ups:
            down_out = down_outs.pop()
            #print(out.shape, down_out.shape)
            out = up(out, down_out, t_emb)


        out = self.norm_out(out)
        out = nn.SiLU()(out)
        out = self.conv_out(out)
        return out



def wrap_angle(x):
    pi = torch.tensor(np.pi, dtype=x.dtype, device=x.device)
    return ((x+pi) % (2*pi)) - pi

# def wrapped_smooth_l1_loss(pred, target, mask, beta=0.1):
#     """
#     pred, target: [batch, 6, 64] – should be angle residuals
#     beta: smooth L1 transition point (standard beta=1.0)
#     """
#     diff = wrap_angle(pred - target)
#     abs_diff = torch.abs(diff)
#     loss = torch.where(
#         abs_diff < beta,
#         0.5 * (diff ** 2) / beta,
#         abs_diff - 0.5 * beta
#     )
#     masked_loss = loss * mask  # mask is 1.0 where valid, 0.0 where invalid
#     return masked_loss.sum() / mask.sum()  # Average over valid positions only


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

    print('='*20)
    print('DDPM TRAINING PIPELINE')
    print('='*20)
    # with open('protein_dataset_file_list.pkl', 'rb') as f:
    #     npy_files = pickle.load(f)  

    ## Specify the folder path with training data here ::
    train_data_folder = sys.argv[1]
    npy_files = glob.glob(os.path.join(train_data_folder,'*.npy'))
    
    dataset = ProteinAngleDataset(npy_files,num_workers=8)
    print('Created Dataset')
    print('Continuing to create loader object....')
    protein_loader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True)


    config = {
        "dataset_params": {
            "im_path": "test128/"
        },
        "diffusion_params": {
            "num_timesteps": 1000,
            "beta_start": 0.0001,
            "beta_end": 0.02
        },
        "model_params": {
            "im_channels": 6,
            "im_size": 128,
            "down_channels": [32, 64, 128, 256],
            "mid_channels": [256, 256, 128],
            "down_sample": [True, True, False],
            "time_emb_dim": 128,
            # "num_down_layers": 2,
            # "num_mid_layers": 2,
            # "num_up_layers": 2,
            "num_heads": 4
        },
        "train_params": {
            "task_name": "default_fd128",
            "batch_size": 32,
            "num_epochs": 400,
            "num_samples": 10,
            "num_grid_rows": 10,
            "lr": 1e-5,
            "ckpt_name": "ddpm_foldingdiff1.pth"
        }
    }




    diffusion_config = config['diffusion_params']
    dataset_config = config['dataset_params']
    model_config = config['model_params']
    train_config = config['train_params']

    scheduler = LinearNoiseScheduler(num_timesteps=diffusion_config['num_timesteps'],
                                     schedule='linear',
                                    beta_start=diffusion_config['beta_start'],
                                    beta_end=diffusion_config['beta_end'],
                                    device='cuda')

    # scheduler = LinearNoiseScheduler(num_timesteps=diffusion_config['num_timesteps'],
    #                                 schedule='cosine',
    #                                 s=0.008,
    #                                 device='cuda')


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


            


