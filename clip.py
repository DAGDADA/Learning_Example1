import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention

class CLIP(nn.Module):
    def __init__(self):
        # 49408表示词汇表的大小，768是clip固定的嵌入大小。77表示序列长度
        self.embedding = CLIPEmbedingding(49408,768,77)

        self.layers = nn.Module([
            # 12 表示多头注意力数量
            CLIPLayer(12,768) for i in range(12) # 类似于Transformor的编码器
        ])

        self.Layernorm = nn.LayerNorm(768)

    # 输入的id通常表示词汇表中每个词位置的数字
    def forward(self,tokens:torch.LongTensor) -> torch.FloatTensor:
        tokens = tokens.type(torch.long)

        # (Batch_size, Seq_len) -> (Batch_size, Seq_len,Dim)
        state = self.embedding(tokens)

        for layer in self.layers:
            state = layer(state)

        # (Batch_size,Seq_len,Dim)
        output = self.Layernorm(state)

        return output

class CLIPLayer(nn.Module):
    def __init__(self,n_head:int,n_embd:int):
        super().__init__()
        self.layernorm_1 = nn.LayerNorm(n_embd)
        self.attention = SelfAttention(n_head,n_embd)
        self.layernorm_2 = nn.LayerNorm(n_embd)
        self.linear_1 = nn.Linear(n_embd,4 * n_embd)
        self.linear_2 = nn.Linear(4 * n_embd,n_embd)

    def forward(self,x:torch.Tensor) -> torch.Tensor:
        # (Batch_size,Seq_len,Dim)
        residue = x
        ## 自注意力
        x = self.layernorm_1(x)
        x = self.attention(x,causal_maks=True)
        x += residue

        ## 前馈
        residue = x
        x = self.layernorm_2(x)
        x = self.linear_1(x)
        x = x * torch.sigmoid(1.702 * x) # QuickGELU函数，没有原理实践中好用
        x = self.linear_2(x)
        x += residue

        return x


class CLIPEmbedingding(nn.Module):
    # 词汇表大小，嵌入大小，词的数量
    def __init__(self,n_vocab:int,n_embd:int,n_token:int):
        super().__init__()

        self.token_embedding = nn.Embedding(n_vocab,n_embd) # 嵌入的数量即词汇表，每一个嵌入词向量的维度
        self.position_embedding = nn.Parameter(torch.zeros(n_token,n_embd)) # 不是由正弦函数获得的，而是一种学习到的参数（从预训练模型中load）

    def forward(self,tokens:torch.LongTensor) -> torch.FloatTensor:
        # (Batch_size, Seq_len) -> (Batch_size, Seq_len,Dim)
        x = self.token_embedding(tokens)
        x += self.position_embedding
        return x