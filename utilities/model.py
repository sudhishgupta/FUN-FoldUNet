#### UNET ####
# shape = [batch, channels=6, seq_length=64]

import torch
import torch.nn as nn

def get_time_embedding(time_steps, t_emb_dim):
    factor = 10000 ** ((torch.arange(
        start = 0, end = t_emb_dim // 2, device = time_steps.device) /  (t_emb_dim // 2)
                      ))
    t_emb = time_steps[:, None].repeat(1, t_emb_dim // 2) / factor
    t_emb = torch.cat([torch.sin(t_emb), torch.cos(t_emb)], dim=-1)

    return t_emb


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
