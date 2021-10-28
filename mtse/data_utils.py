from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np

def transform_seq(seq, dim, order_data, norm):
    assert type(dim) == int
    seq = np.concatenate(seq[:dim*2+1], 1) 
    if norm is not None:
        for i, values in zip(range(dim), norm):
            seq[:, i] = (seq[:, i] - values[0]) / values[1]
    if order_data:
        top, bottom = [], []
        for i in range(len(seq)):
            if seq[i][dim:dim*2].sum() != 0:
                top.append(i)
            else:
                seq[i][dim*2] = 0
                bottom.append(i)
        return np.concatenate([seq[top], seq[bottom]])
    else:
        return seq

class mtan_Dataset(Dataset):    
    def __init__(self, data, order_data=True, reduce_data=True, test_data=False, norm=None):
        order_data = reduce_data if reduce_data else order_data 
        self.test_data = test_data
        self.dim = int((len(data[0])-2) / 2) if not test_data else int((len(data[0])-1) / 2)
        self.sequences = torch.tensor([transform_seq(seq, self.dim, order_data, norm) for seq in data]).float()
        if reduce_data:
            max_tps = [seq[:, self.dim:-1].sum(1).sum() for seq in self.sequences]
            self.sequences = self.sequences[:, :int(np.max(max_tps)), :]
        if not test_data:
            self.target = torch.tensor([seq[(self.dim*2+1)] for seq in data]).float()
    def __len__(self):
        return len(self.sequences)    
    def __getitem__(self, idx):
        return (self.sequences[idx], self.target[idx]) if not self.test_data else self.sequences[idx]



# assert 1==3, 'ass1'