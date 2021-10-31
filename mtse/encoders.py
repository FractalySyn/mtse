import torch
import torch.nn as nn
import numpy as np
from .attention import MultiTimeAttention

__doc__ = """
Time series encoders

Classes
-------
mtan_enc : MTAN encoder

Functions
---------
default_regressor : default top regressor when not provided in `mtan_enc`
default_classifier : default top classifier when not provided in `mtan_enc`
"""

def default_regressor(nhidden):
    """default top regressor when not provided in `mtan_enc`"""
    return nn.Sequential(
        nn.Linear(nhidden, 150),
        nn.ReLU(),
        nn.Linear(150, 150),
        nn.ReLU(),
        nn.Linear(150, 1))

def default_classifier(n_out, nhidden):
    """default top classifier when not provided in `mtan_enc`"""
    return nn.Sequential(
        nn.Linear(nhidden, 150),
        nn.ReLU(),
        nn.Linear(150, 150),
        nn.ReLU(),
        nn.Linear(150, n_out))

## MTAN encoder
class mtan_enc(nn.Module):
    """
    MTAN encoder, inherited from torch.nn.Module

    Attributes
    ----------
    embed_time : int
        dimension of time embeddings
    learn_emb : bool
        if True, time embedding is learnt by the model
    dim : int
        number of time series
    device : str
        device, 'cuda' or 'cpu'
    nhidden : int
        dimension of hidden layer, passed to the classifier / regressor
    query : torch.tensor
        query values, dimension equals to specified value or embed_time
    freq : float
        parameter of the time embedding when static (learn_emb set to False)
    model_type: str
        top model type
    att : MultiTimeAttention
        attention mechanism
    model : torch.nn model
        top model
    enc : torch.nn model
        RNN-type model
    periodic : torch.nn.Linear
        for time embedding when dynamic
    linear : torch.nn.Linear
        for time embedding when dynamic

    Methods
    -------
    learn_time_embedding(tt)
        Time embedding architecture when dynamic
    time_embedding(pos, d_model)
        Time embedding architecture when static
    forward(self, x, time_steps, encode_ts=False)
        PyTorch forward method, feed forward the model when instance is called
    """
 
    def __init__(self, n_ts, model_type, regressor, classifier, seq_encoder, nhidden, embed_time, 
                 n_heads, learn_emb, pdrop, device='cuda', query=None, freq=1., classif_out=2):
        """
        Parameters
        ----------
        n_ts : int
            number of time series
        model_type : str
            'regression' or 'classification'
        regressor : torch model or NoneType
            used to specify a custom top regressor
        classifier : torch model or NoneType
            used to specify a custom top classifier
        seq_encoder : torch model or NoneType
            used to specify a custom RNN 
        nhidden : int
            dimension of the first hidden layer
        embed_time : int
            dimension used to embed time
        n_heads : int
            number of attention heads, such as embed_time / n_heads is an integer
        learn_emb : bool
            if True, time embedding is learnt by the model
        pdrop : float
            probability of dropout
        device : str, optional
            device, 'cuda' or 'cpu' (default is 'cuda')
        query : torch.tensor or NoneType, optional
            query values dimension; if None, it is set to `embed_time` (default is None')
        freq : float, optional
            parameter of the time embedding when static, i.e. `learn_emb` set to False (default is 1.)
        classif_out : int, optional
            number of classes in case of classification, used if `classifier` is None (default is 2)
        """
        super(mtan_enc, self).__init__()
        assert embed_time % n_heads == 0
        self.embed_time = embed_time
        self.learn_emb = learn_emb
        self.dim = n_ts
        self.device = device
        self.nhidden = nhidden
        self.query = torch.linspace(0, 1., embed_time) if query is None else torch.linspace(0, 1., query)
        self.freq = freq
        self.model_type = model_type
        self.att = MultiTimeAttention(2*n_ts, nhidden, embed_time, n_heads, pdrop)
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
        """
        Parameters
        ----------
        x : torch.tensor
            sequences
        time_steps : torch.tensor
        encode_ts : bool
            if True, returns the embeddings of time series instead of predictions

        Returns
        -------
        predictions or embeddings as a torch.tensor
        """
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