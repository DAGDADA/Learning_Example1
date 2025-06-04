import torch
from torch import nn
from torch.nn import functional as F, Upsample
from attention import SelfAttention, CrossAttention

class Diffusion(nn.Module):
    def __init__(self):
        self.time_embedding = TimeEmbedding(320) # 时间嵌入的大小
        self.unet = UNET()
        self.final = UNET_OutputLayer(320,4)

    def forward(self,latent: torch.Tensor,context: torch.Tensor,time:torch.Tensor)->torch.Tensor:
        # latent: (Batch_size,4,height/8,width/8)
        # context: (Batch_size,Seq_len,Dim)
        # time: (1,320)

        # (1,320) -> (1,1280)
        time = self.time_embedding(time)  # 类似于Transformor的位置编码

        # (Batch_size,4,height/8,width/8) -> (Batch_size,320,height/8,width/8)
        output  = self.unet(latent,context,time)

        # (Batch_size,320,height/8,width/8) -> (Batch_size,4,height/8,width/8)
        output  = self.final(output)
        return output


class UNET_OutputLayer(nn.Module):
    def __init__(self,in_channels:int,out_channels:int):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32,in_channels)
        self.conv = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=3,padding=1)

    def forward(self,x):
        #  x:(Batch_size,320,height/8,width/8)
        x = self.groupnorm(x)

        x = F.silu(x)
        #  x:(Batch_size,320,height/8,width/8) -> x:(Batch_size,4,height/8,width/8)
        x = self.conv(x)
        return x

class UNET_AttentionBlock(nn.Module):
    def __init__(self,n_head:int,n_emd:int,d_context=768):
        super().__init__()
        channels = n_head * n_emd
        self.groupnorm = nn.GroupNorm(32,channels,eps=1e-6)
        self.conv_input = nn.Conv2d(in_channels=channels,out_channels=channels,kernel_size=1,padding=0)

        self.layernorm_1 = nn.LayerNorm(channels)
        self.attention_1 = SelfAttention(channels)
        self.layernorm_2 = nn.LayerNorm(channels)
        self.attention_2 = SelfAttention(channels)
        self.layernorm_3 = nn.LayerNorm(channels)
        self.linear_geglu_1 = nn.Linear(channels,4*channels*2)
        self.linear_geglu_2 = nn.Linear(4*channels,channels)

        self.conv_outpue = nn.Conv2d(in_channels=channels,out_channels=channels,kernel_size=1,padding=0)

    def forward(self,x,context):
        # x:(batch_size,feature,height,width)
        # context:(Batch_size,Seq_len,Dim)

        residual_long = x
        x = self.groupnorm(x)
        x = self.conv_input(x)

        n,c,h,w = x.size()
        # x: (batch_size, feature, height, width) ->x:(batch_size,feature,height*width)
        x = x.view(n,c,h*w)
        # x:(batch_size,feature,height*width) -> x:(batch_size,height*width,feature)
        x = x.tranpose(-1,-2)

        # 归一化+自注意力
        residual_short = x
        x = self.layernorm_1(x)
        self.attention_1(x)
        x += residual_short

        # 归一化+交叉注意力
        residual_short = x
        x = self.layernorm_2(x)
        self.attention_2(x,context)
        x += residual_short

        residual_short = x
        x = self.layernorm_3(x)
        x,gate = self.linear_geglu_1(x).chunk(2,-1)
        x = x * F.gelu(gate)

        x = self.linear_geglu_2(x)
        x += residual_short
        # x:(batch_size,height*width,feature) -> x:(batch_size,feature,height*width)
        x = x.transpose(-1,-2)
        x = x.view(n,c,h,w)

        return self.conv_outpue(x) + residual_long




class UNET_ResidualBlock(nn.Model):
    def __init__(self,in_channels:int,out_channels:int,n_time:int):  # 增加了时间步的嵌入
        super().__init__()
        self.groupnorm_feature = nn.GroupNorm(32,in_channels)
        self.conv_feature = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=3,padding=1)
        self.linear_time = nn.Linear(n_time,out_channels)

        self.groupnorm_merged = nn.GroupNorm(32, out_channels)
        self.conv_merged = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=3,padding=1)

        if(in_channels == out_channels):
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=1,padding=0)

    def forward(self,feature,time):  # 将潜在特征与时间嵌入关联起来
        # feature:(batch_size,in_channels,height,width)
        # time :(1,1280)
        residual = feature
        feature = self.groupnorm_feature(feature)
        feature = F.silu(feature)
        feature = self.conv_feature(feature)

        time = F.silu(time)
        time = self.linear_time(time)
        merge = feature + time.unsqueeze(-1).unsqueeze(-1) # time 没有batch_size in_channels的维度
        merge = self.groupnorm_merge(merge)
        merge = F.silu(merge)
        merge = self.conv_merged(merge)
        return merge + self.residual_layer(residual)



class UNET(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Module([
            # (Batch_size,4,height/8,width/8) -> (Batch_size,320,height/8,width/8)
            SwitchSequential(nn.Conv2d(4, 320, kernel_size=3, padding=1)),
            SwitchSequential(UNET_ResidualBlock(320,320),UNET_AttentionBlock(8,40)),
            SwitchSequential(UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40)),

            # (Batch_size,320,height/8,width/8) -> (Batch_size,320,height/16,width/16)
            SwitchSequential(nn.Conv2d(320,320,kernel_size=3, stride=2, padding=1)),
            SwitchSequential(UNET_ResidualBlock(320, 640), UNET_AttentionBlock(8, 80)),  # (Batch_size,320,height/16,width/16) -> (Batch_size,640,height/16,width/16)

            # (Batch_size,640,height/16,width/16) -> (Batch_size,640,height/32,width/32)
            SwitchSequential(nn.Conv2d(640, 640, kernel_size=3, stride=2, padding=1)),
            SwitchSequential(UNET_ResidualBlock(640, 1280), UNET_AttentionBlock(8, 160)),
            SwitchSequential(UNET_ResidualBlock(1280, 1280), UNET_AttentionBlock(8, 160)),  # (Batch_size,640,height/16,width/16) -> (Batch_size,1280,height/32,width/32)

            # (Batch_size,1280,height/32,width/32) -> (Batch_size,1280,height/64,width/64)
            SwitchSequential(nn.Conv2d(1280, 1280, kernel_size=3, stride=2, padding=1)),

            SwitchSequential(UNET_ResidualBlock(1280, 1280)),
            SwitchSequential(UNET_ResidualBlock(1280, 1280)),
            SwitchSequential(UNET_ResidualBlock(1280, 1280)),  # (Batch_size,1280,height/64,width/64)
        ])

        self.bottleneck = SwitchSequential(
            UNET_ResidualBlock(1280, 1280),
            UNET_AttentionBlock(8, 160),
            UNET_ResidualBlock(1280, 1280)
        )

        self.decoder = nn.Module([
            # (Batch_size,2560,height/64,width/64) -> (Batch_size,1280,height/64,width/64)
            SwitchSequential(UNET_ResidualBlock(2560,1280)),
            SwitchSequential(UNET_ResidualBlock(2560,1280)),
            SwitchSequential(UNET_ResidualBlock(2560, 1280),UpSample(1280)),
            SwitchSequential(UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8,160)),
            SwitchSequential(UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)),
            SwitchSequential(UNET_ResidualBlock(1920, 1280), UNET_AttentionBlock(8, 160),Upsample(1280)),
            SwitchSequential(UNET_ResidualBlock(1920, 640), UNET_AttentionBlock(8, 80)),
            SwitchSequential(UNET_ResidualBlock(1280, 640), UNET_AttentionBlock(8, 80)),
            SwitchSequential(UNET_ResidualBlock(960, 640), UNET_AttentionBlock(8, 80),Upsample(640)),
            SwitchSequential(UNET_ResidualBlock(960, 320), UNET_AttentionBlock(8, 40)),
            SwitchSequential(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 40)),
            SwitchSequential(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 40)),
        ])

class UpSample(nn.Module):
    def __init__(self, channels:torch.Tensor):
        super().__init__()
        self.conv = nn.Conv2d(channels,channels,kernel_size=3, padding=1)

    def forward(self,x:torch.Tensor) ->torch.Tensor:
        # (Batch_size,feature,height,width) -> (Batch_size, feature, height*2,width*2)
        x = F.interpolate(x,scale_factor=2,mode='nearest')
        return self.conv(x)

class SwitchSequential(nn.Sequential):
    def forward(self, x:torch.Tensor, context:torch.Tensor, time:torch.Tensor) -> torch.Tensor:
        for layer in self:
            if isinstance(layer, UNET_AttentionBlock):
                x = layer(x, context)  # 计算潜在向量与提示的交叉注意力
            elif isinstance(layer, UNET_ResidualBlock):
                x = layer(x, time)  # 计算潜在向量与时间步匹配
            else: x = layer(x)
        return x

class TimeEmbedding(nn.Module):
    def __init__(self, n_embd:int):
        super().__init__()
        self.linear_1 = nn.Linear(n_embd,4 * n_embd)
        self.linear_2 = nn.Linear(4 * n_embd,4 * n_embd)

    def forward(self,x:torch.Tensor)->torch.Tensor:
        # x: (1,320) -> (1,1280)
        x = self.linear_1(x)

        x = F.silu(x)

        # x: (1,1280) -> (1,1280)
        x = self.linear_2(x)
        return x