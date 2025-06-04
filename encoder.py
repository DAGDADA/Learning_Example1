import torch
from torch import nn
from torch.nn import functional as F
from decoder import VAE_AttentionBlock, VAE_ResidualBlock

class VAE_Encoder(nn.Module):
    # 将图像转换为更小的东西，可以视为压缩图像
    # 但变分自编码器实际上不是在做压缩，而是在学习一个潜空间，是在学习一个分布，其输出实际上是均值与对数方差
    # 图像大小不断减小，但channel不断增大
    def __init__(self):
        super().__init__(
            # (Batch_size,Channel,Height,Width) -> (Batch_size,128,Height, Width)
            nn.Conv2d(3, 128, kernel_size=3, padding=1),

            # (Batch_size,128,Height,Width) -> (Batch_size,128,Height, Width)
            VAE_ResidualBlock(128, 128),
            VAE_ResidualBlock(128, 128),

            # (Batch_size,128,Height,Width) -> (Batch_size,128,Height/2, Width/2)
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),

            # (Batch_size,128,Height/2, Width/2) -> (Batch_size,256,Height/2, Width/2)
            VAE_ResidualBlock(128, 256),

            # (Batch_size,256,Height/2,Width/2) -> (Batch_size,128,Height/2, Width/2)
            VAE_ResidualBlock(256, 256),

            # (Batch_size,256,Height/2,Width/2) -> (Batch_size,256,Height/4,Width/4)
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),

            # (Batch_size,256,Height/4, Width/4) -> (Batch_size,512,Height/4, Width/4)
            VAE_ResidualBlock(256, 512),

            # (Batch_size,512,Height/4, Width/4) -> (Batch_size,512,Height/4, Width/4)
            VAE_ResidualBlock(512, 512),

            # (Batch_size,512,Height/4, Width/4) -> (Batch_size,512,Height/8,Width/8)
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),

            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),

            # (Batch_size,512,Height/8,Width/8) -> (Batch_size,512,Height/8,Width/8)
            VAE_AttentionBlock(512),

            # (Batch_size,512,Height/8,Width/8) -> (Batch_size,512,Height/8,Width/8)
            VAE_ResidualBlock(512, 512),

            # (Batch_size,512,Height/8,Width/8) -> (Batch_size,512,Height/8,Width/8)
            nn.GroupNorm(32, 512),

            # (Batch_size,512,Height/8,Width/8) -> (Batch_size,512,Height/8,Width/8)
            nn.SiLU(),

            # (Batch_size,512,Height/8,Width/8) -> (Batch_size,8,Height/8,Width/8)
            nn.Conv2d(512,8, kernel_size=3,padding=1),

            # (Batch_size,8,Height/8,Width/8) -> (Batch_size,8,Height/8,Width/8)
            nn.Conv2d(8,8,kernel_size=1,padding=0)
        )

    def forward(self, x:torch.Tensor,noise:torch.Tensor) -> torch.Tensor:
        # x:(Batch_size,channel,Height,Width)
        # noise:(Batch_size,8 Height/8,Width/8)
        for module in self: # 向前传播过程经过每一个module
            if getattr(module, 'stride', None)==(2,2):
                x = F.pad(0,1,0,1)
            x = module(x)

        # （Batch_size,8, Height/8,Width/8) -> 2个 (Batch_size,4,Height/8,Width/8)
        mean, log_variance = torch.chunk(x, chunks=2, dim=1) # 沿着维度分成两个张量

        # (Batch_size,4,Height/8,Width/8)  -限制一下log_variance的范围
        log_variance = torch.clamp(log_variance, min=-30, max=20)

        # (Batch_size,4,Height/8,Width/8)  获得方差
        variance = log_variance.exp()

        # (Batch_size,4,Height/8,Width/8)  获得标准差
        stdev = variance.exp().sqrt()

        # if want to z = N(0,1) -> N(mean,stdev)=x
        # x = mean + stdev * z
        x = mean + stdev * noise # noise 利用噪声生成器使用特定的种子生成

        # 数据的缩放常数
        x *= x * 0.18215

        return x