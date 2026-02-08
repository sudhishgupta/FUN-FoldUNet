''' IMPORTANT PARTS '''

import torch
import torch.nn as nn
import numpy as np
import math

def wrap_angle(x):
    pi = torch.tensor(np.pi, dtype=x.dtype, device=x.device)
    return ((x+pi) % (2*pi)) - pi

def wrapped_smooth_l1_loss(pred, target, mask, beta=0.1):
    """
    pred, target: [batch, 6, 64] – should be angle residuals
    beta: smooth L1 transition point (standard beta=1.0)
    """
    diff = wrap_angle(pred - target)
    abs_diff = torch.abs(diff)
    loss = torch.where(
        abs_diff < beta,
        0.5 * (diff ** 2) / beta,
        abs_diff - 0.5 * beta
    )
    masked_loss = loss * mask  # mask is 1.0 where valid, 0.0 where invalid
    return masked_loss.sum() / mask.sum()  # Average over valid positions only


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
#         return wrap_angle(xt)                    #final equation giving wrapped(xt)
        

#     # REVERSE DIFFUSION PROCESS

#     def sample_prev_timestep(self, xt, noise_pred, t):

#         t = torch.tensor(t, device=self.device) if not torch.is_tensor(t) else t.to(self.device)

#         #original image saving for visualization
#         x0 = (xt - (self.sqrt_one_minus_alpha_cum_prod[t]*noise_pred)) / self.sqrt_alpha_cum_prod[t]
#         #x0 = torch.clamp(x0, -1., 1.)

#         #computing mean at timestep t for sampling

#         mean = xt - ((self.betas[t] * noise_pred) / (self.sqrt_one_minus_alpha_cum_prod[t]))
#         mean = mean / torch.sqrt(self.alphas[t])


#         # at timestep 0 noise is not added
#         if t == 0:
#             mean_wrapped = wrap_angle(mean)
#             x0_wrapped = wrap_angle(x0)
#             return mean_wrapped,x0_wrapped
#         else:
#             variance = (1. - self.alpha_cum_prod[t-1]) / (1. - self.alpha_cum_prod[t])
#             variance = variance * self.betas[t]
#             sigma = variance ** 0.5
#             z = torch.randn(xt.shape).to(xt.device)

#             sample = mean + sigma * z
            
#             sample_wrapped = wrap_angle(sample)
#             x0_wrapped = wrap_angle(x0)

#             return sample_wrapped, x0_wrapped



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

        self.residual_input_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)               # 1x1 conv2d layer for makiing sure residual connections work well !

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
                             down_sample=self.down_sample[i], num_heads=4))

        
        self.mids = nn.ModuleList([])
        for i in range(len(self.mid_channels)-1):
            self.mids.append(MidBlock(self.mid_channels[i], self.mid_channels[i+1], self.t_emb_dim, num_heads=4))
            
        self.ups = nn.ModuleList([])
        for i in reversed(range(len(self.down_channels)-1)):
            self.ups.append(UpBlock(self.down_channels[i]*2, self.down_channels[i-1] if i != 0 else 16,
                                    self.t_emb_dim, up_sample=self.down_sample[i], num_heads=4))

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



            


        