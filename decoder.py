import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention

class VAE_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        # 由归一化和卷积组成
        # 归一化思想，不希望结果波动太大，影响损失函数的波动，继而影响训练速度
        self.groupnorm_1  = nn.GroupNorm(32, in_channels)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.groupnorm_2 = nn.GroupNorm(32, out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1,padding=0)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        # x:(Batch_size,in_channels,height,width)
        # 跳跃残差连接
        residue = x

        x = self.groupnorm_1(x)
        x = F.silu(x)
        x = self.conv_1(x)

        x = self.groupnorm_2(x)
        x = F.silu(x)
        x = self.conv_2(x)

        return x + self.residual_layer(residue)


class VAE_AttentionBlock(nn.Module):
    def __init__(self,channels:int):
        super().__init__()
        # 组归一化 以及 自注意力
        # 特征来源于卷积，组归一化和层归一化相近，但不同的是，组归一化但不是对所有的特征进行归一化
        # 而是基于彼此接近的特征将具有相近的分布思想，将其分组进行归一化
        self.groupnorm = nn.GroupNorm(32,channels)
        self.attention = SelfAttention(1,channels)

    def forward(self,x:torch.Tensor) -> torch.Tensor:
        # x:(batch_size,feature,height,width)
        # 注意力块会被多处使用，不在此定义具体的大小
        residue = x

        n,c,h,w = x.shape

        # 对当前图像的所有像素进行自注意力计算
        # x:(batch_size,feature,height,width)  -> x:(batch_size,feature,height * width)
        x = x.view(n,c,h * w)

        # x:(batch_size,feature,height * width) -> x:(batch_size,height * width,feature)
        # 可以将之视为像素序列，每一个像素有自己的嵌入即像素特征，然后像素相互关联
        x = x.transpose(-1,-2)

        # x:(batch_size,height * width,feature) -> x:(batch_size,height * width,feature)
        x = self.attention(x)  # 自注意力

        #  x:(batch_size,height * width,feature) -> x:(batch_size,feature,height * width)
        x = x.transpose(-1,-2)

        # x:(batch_size,feature,height * width) -> x:(batch_size,feature,height,width)
        x = x.view((n,c,h,w))

        x += residue

        return x


class VAE_Decoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            # (Batch_size,4,Height/8,Width/8)
            nn.Conv2d(4,4,kernel_size=1,padding=0),
            nn.Conv2d(4,512,kernel_size=3,padding=1),  # (Batch_size,512,Height/8,Width/8)

            VAE_ResidualBlock(512,512),
            VAE_AttentionBlock(512),

            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),   # (Batch_size,512,Height/8,Width/8) -> (Batch_size,512,Height/8,Width/8)

            # (Batch_size,512,Height/8,Width/8) -> (Batch_size,512,Height/4,Width/4)
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512,512,kernel_size=3,padding=1),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),

            # (Batch_size,512,Height/4,Width/4) -> (Batch_size,512,Height/2,Width/2)
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512,512,kernel_size=3,padding=1),
            VAE_ResidualBlock(512, 256),
            VAE_ResidualBlock(256, 256),
            VAE_ResidualBlock(256, 256),  # (Batch_size,512,Height/2,Width/2) - > (Batch_size,256,Height/2,Width/2)

            # (Batch_size,256,Height/2,Width/2) -> (Batch_size,256,Height,Width)
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256,256,kernel_size=3,padding=1),
            VAE_ResidualBlock(256, 128),
            VAE_ResidualBlock(128, 128),
            VAE_ResidualBlock(128, 128),  # (Batch_size,256,Height,Width) -> (Batch_size,128,Height,Width)

            nn.GroupNorm(32,128),

            nn.SiLU(),

            # (Batch_size,128,Height,Width) -> (Batch_size,3,Height,Width)
            nn.Conv2d(128,3,kernel_size=3,padding=1)
        )

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        # x:(Batch_size,4,Height/8,Width/8)
        x /= 0.18215

        for module in self:
            x = module(x)

        # (Batch_size,3,Height,Width)
        return x
