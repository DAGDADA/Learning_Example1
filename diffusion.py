import torch
from torch import nn
from torch.nn import functional as F
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