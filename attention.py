# stable diffusion中包含两种注意力机制，自注意力与交叉注意力

import torch
from torch import nn
from torch.nn import functional as F
import math


# 自注意力的q、w、k都来自一处
# 交叉注意力中q来自一处，w、k来自另一处

class CrossAttention(nn.Module):
    def __init__(self,n_head:int,d_emd:int,d_cross:int,in_proj_bias=True,out_proj_bias=True):
        super().__init__()
        self.q_proj = nn.Linear(d_emd,d_emd,bias=in_proj_bias)
        self.k_proj = nn.Linear(d_cross,d_emd,bias=in_proj_bias)
        self.v_proj = nn.Linear(d_cross,d_emd,bias=in_proj_bias)
        self.out_proj = nn.Linear(d_emd,d_emd,bias=out_proj_bias)
        self.n_head = n_head
        self.d_head = d_emd // n_head

    def forward(self,x,y):
        # x : latent : batch_size, seq_len_q, dim_q
        # y : context : batch_size, seq_len_kv, dim_kv
        input_shape = x.shape
        batch_size,seq_len,d_embd = input_shape
        interm_shape = (batch_size,-1,self.n_head,self.d_head)

        q = self.q_proj(x)
        k = self.k_proj(y)
        v = self.v_proj(y)

        q  = q.view(interm_shape).transpose(1,2)
        k = k.view(interm_shape).transpose(1,2)
        v = v.view(interm_shape).transpose(1,2)

        weight = q @ k.transpose(-1,-2)
        weight /= math.sqrt(self.d_head)
        weight = F.softmax(weight,dim=-1)

        output = weight @ v
        output = output.transpose(1,2).contiguous()
        output = self.out_proj(output)

        return output

class SelfAttention(nn.Module):
    def __init__(self, n_head:int, d_embed:int,in_proj_bias=True,out_proj_bias=True):
        super().__init__()

        # WQ,WK,WV 用一个巨大的线性层来表示，而非矩阵
        self.in_proj = nn.Linear(d_embed,3 * d_embed,bias=in_proj_bias) # 应用注意力之前对输入进行投影
        self.out_proj = nn.Linear(d_embed,d_embed,bias=out_proj_bias)
        self.n_head = n_head
        self.d_head = d_embed//n_head

    def forward(self, x:torch.tensor, causal_mask=False):
        # 掩码是一种将一个特定的token与它之后的token关联，而只与其之前的token关联的方法
        # x:(batch_size,seq_len,dim)
        input_shape = x.shape
        batch_size, sequence_length, d_embed = input_shape

        intermim_shape = (batch_size, sequence_length, self.n_head, self.d_head)

        # x:(batch_size,seq_len,dim) -> (batch_size,seq_len,dim*3) -> 3个 (batch_size,seq_len,dim)
        q,k,v = self.in_proj(x).chunk(3, dim=-1)

        # (batch_size,seq_len,dim) -> ((batch_size,seq_len,h,dim/h) -> (batch_size,h,seq_len,dim/h)
        q = q.view(intermim_shape).transpose(1,2)
        k = k.view(intermim_shape).transpose(1,2)
        v = v.view(intermim_shape).transpose(1,2)

        # (batch_size,h,seq_len,seq_len)
        weight = q @ k.transpose(-1,-2)

        if causal_mask:
            # 创建一个掩码，其中上三角由1构成
            mask = torch.ones_like(weight,dtype=torch.bool).triu(diagonal=1)
            # 用负无穷填充掩码
            weight.masked_fill_(mask,-torch.inf)

        weight /= math.sqrt(self.d_head)
        weight = F.softmax(weight,dim=-1)

        # (batch_size,h,seq_len,dim/h)
        output = weight @ v

        # (batch_size,seq_len,h,dim/h)
        output = output.transpose(1,2)

        # (batch_size, seq_len, dim)
        output = output.reshape(input_shape)

        # 与WO矩阵相乘 (batch_size, seq_len, dim) -> (batch_size, seq_len, dim)
        output = self.out_proj(output)

        return output

