import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

## MTAN attention ##
class MultiTimeAttention(nn.Module):
    
    def __init__(self, input_dim, nhidden=16, embed_time=16, n_heads=1, pdrop=0.5):
        super(MultiTimeAttention, self).__init__()
        assert embed_time % n_heads == 0
        self.embed_time = embed_time
        self.embed_time_k = embed_time // n_heads
        self.h = n_heads
        self.dim = input_dim
        self.nhidden = nhidden
        self.linears = nn.ModuleList([nn.Linear(embed_time, embed_time), 
                                      nn.Linear(embed_time, embed_time),
                                      nn.Linear(input_dim*n_heads, nhidden)])
        self.pdrop = pdrop
        self.dropout = nn.Dropout(pdrop)
        
    def attention(self, query, key, value, mask=None):
        "Compute 'Scaled Dot Product Attention'"
        dim = value.size(-1)
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(d_k)
        scores = scores.unsqueeze(-1).repeat_interleave(dim, dim=-1)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(-3) == 0, -1e9)
        p_attn = F.softmax(scores, dim = -2)
        p_attn = self.dropout(p_attn) if self.pdrop > 0 else p_attn
        return torch.sum(p_attn*value.unsqueeze(-3), -2), p_attn    
    
    def forward(self, query, key, value, mask=None):
        "Compute 'Scaled Dot Product Attention'"
        batch, seq_len, dim = value.size()
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        value = value.unsqueeze(1)
        query, key = [l(x).view(x.size(0), -1, self.h, self.embed_time_k).transpose(1, 2) for l, x in zip(self.linears, (query, key))]
        x, _ = self.attention(query, key, value, mask)
        x = x.transpose(1, 2).contiguous().view(batch, -1, self.h * dim)
        return self.linears[-1](x)