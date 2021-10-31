import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

__doc__ = """
Attention mechanisms computation

Classes
-------
MultiTimeAttention : Attention mechanism of the mTan encoder
"""

## MTAN attention ##
class MultiTimeAttention(nn.Module):
    """
    Class for the attention mechanism of the mTan encoder. Inherited from torch.nn.Module

    Attributes:
    -----------
    embed_time : int
        dimension of time embeddings
    embed_time_k : int
        dimension of time embedding per attention head
    h : int
        number of attention heads
    dim : int
        number of input data time series * 2
    nhidden : int
        dimension returned by the `forward` method
    linears: torch.nn.ModuleList
        list of linear layers used in the attention mecanism
    pdrop : float
        drop probability used in dropout
    dropout : torch.nn.Dropout
        dropout function

    Methods
    -------
    attention(query, key, value, mask=None)
        Computes scaled dot product attention
    forward(query, key, value, mask=None)
        Computes attention mechanism
    """

    def __init__(self, input_dim, nhidden, embed_time, n_heads, pdrop):
        """
        Parameters
        ----------
        input_dim : int
            2 * number of time series
        n_hidden : int
            dimension to output
        embed_time : int
            dimension used to embed time
        n_heads : int
            number of attention heads, such as embed_time / n_heads is an integer
        pdrop : float
            probability of dropout
        """
        
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
        """
        Computes 'Scaled Dot Product Attention'
        
        Parameters
        ----------
        query: torch.tensor
            query values
        key : torch.tensor
            key values
        value : torch.tensor
            original values
        mask : torch.tensor or NoneType, optional
            defines which values have to be masked (default is None)

        Returns
        -------
        A torch.tensor resulting from the SDPA computation. Dropout is applied if pdrop > 0 in __init__()
        """
        dim = value.size(-1)
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(d_k)
        scores = scores.unsqueeze(-1).repeat_interleave(dim, dim=-1)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(-3) == 0, -1e9)
        p_attn = F.softmax(scores, dim = -2)
        p_attn = self.dropout(p_attn) if self.pdrop > 0 else p_attn
        return torch.sum(p_attn*value.unsqueeze(-3), -2)
    
    def forward(self, query, key, value, mask=None):
        """
        Computes attention mechanism
        
        Parameters
        ----------
        query: torch.tensor
            query values
        key : torch.tensor
            key values
        value : torch.tensor
            original values
        mask : torch.tensor or NoneType, optional
            defines which values have to be masked (default is None)

        Returns
        -------
        A torch.tensor resulting from the attention computation
        """

        batch, seq_len, dim = value.size()
        if mask is not None:
            mask = mask.unsqueeze(1)
        value = value.unsqueeze(1)
        query, key = [l(x).view(x.size(0), -1, self.h, self.embed_time_k).transpose(1, 2) for l, x in zip(self.linears, (query, key))]
        x = self.attention(query, key, value, mask)
        x = x.transpose(1, 2).contiguous().view(batch, -1, self.h * dim)
        return self.linears[-1](x)