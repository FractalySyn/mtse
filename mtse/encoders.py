import torch
import torch.nn as nn
import numpy as np
from .attention import MultiTimeAttention

def default_regressor(nhidden):
    return nn.Sequential(
        nn.Linear(nhidden, 150),
        nn.ReLU(),
        nn.Linear(150, 150),
        nn.ReLU(),
        nn.Linear(150, 1))

def default_classifier(n_out, nhidden):
    return nn.Sequential(
        nn.Linear(nhidden, 150),
        nn.ReLU(),
        nn.Linear(150, 150),
        nn.ReLU(),
        nn.Linear(150, n_out))

## MTAN encoder
class enc_mtan_reg(nn.Module):
 
    def __init__(self, n_ts, model_type='regression', regressor=None, classifier=None, classif_out=2, seq_encoder=None, nhidden=16, embed_time=16, 
                 n_heads=1, learn_emb=True, freq=10., device='cuda'):
        super(enc_mtan_reg, self).__init__()
        assert embed_time % n_heads == 0
        self.freq = freq
        self.embed_time = embed_time
        self.learn_emb = learn_emb
        self.dim = n_ts
        self.device = device
        self.nhidden = nhidden
        self.query = torch.linspace(0, 1., 64)
        self.model_type = model_type
        self.att = MultiTimeAttention(2*n_ts, nhidden, embed_time, n_heads)
        if model_type == 'regression' and regressor is not None:
            self.model = regressor
        elif model_type == 'classification' and classifier is not None:
            self.model = classifier
        elif model_type == 'regression':
            self.model = default_regressor(nhidden)
        elif model_type == 'classification':
            self.model = default_classifier(classif_out, nhidden)
        self.enc = nn.GRU(nhidden, nhidden, bidirectional=False) if seq_encoder is None else seq_encoder
        if learn_emb:
            self.periodic = nn.Linear(1, embed_time-1)
            self.linear = nn.Linear(1, 1)     
    
    def learn_time_embedding(self, tt):
        tt = tt.to(self.device)
        tt = tt.unsqueeze(-1)
        out2 = torch.sin(self.periodic(tt))
        out1 = self.linear(tt)
        return torch.cat([out1, out2], -1)       
        
    def time_embedding(self, pos, d_model):
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model)
        position = 48.*pos.unsqueeze(2)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(np.log(self.freq) / d_model))
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe   
       
    def forward(self, x, time_steps, encode_ts=False):
        time_steps = time_steps.cpu()
        mask = x[:, :, self.dim:]
        mask = torch.cat((mask, mask), 2)
        if self.learn_emb:
            key = self.learn_time_embedding(time_steps).to(self.device)
            query = self.learn_time_embedding(self.query.unsqueeze(0)).to(self.device)
        else:
            key = self.time_embedding(time_steps, self.embed_time).to(self.device)
            query = self.time_embedding(self.query.unsqueeze(0), self.embed_time).to(self.device)            
        out = self.att(query, key, x, mask)
        out = out.permute(1, 0, 2)
        _, out = self.enc(out)
        return self.model(out.squeeze(0)).squeeze() if not encode_ts else out