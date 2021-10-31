from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np

__doc__ = """
Dataset classes inherited from the torch.utils.data.Dataset class

Classes
-------
mtan_Dataset : MTAN dataset class

Functions
---------
transform_seq : Helper for the mtan_Dataset class
"""

def transform_seq(seq, dim, order_data, norm=None):
    """
    Helper for the mtan_Dataset class

    Parameters
    ----------
    seq : ndarray
        sequence of a single observation, contains time series data, masks, time points and optionally the target / label value
    dim : int
        number of time series
    order_data : bool
        if True, reorder data such that unobserved time points are put at the bottom of the output
    norm : list-like or NoneType
        if not None, used to standardize time series data; of shape (dim, 2), each element provides the mean and the standard deviation of a time series

    Returns
    -------
    transformed ndarray
    """

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
    """
    MTAN dataset class inherited from the torch.utils.data.Dataset class
    
    Attributes
    ----------
    test_data : bool
        True if the dataset provided is a test set i.e. it does not contain targets / labels
    dim : int
        number of time series
    sequences : torch.tensor
        returned sequences
    target : torch.tensor
        returned targets / labels

    Methods
    -------
    __len__() 
        Returns the number of observations

    __getitem__(idx)
        Returns the sequence (and target / label if it exists) at index idx
    """

    def __init__(self, data, order_data, reduce_data, norm, test_data=False):
        """
        Parameters
        ----------
        data : list of list or similar
            data to process
        order_data : bool
            if True, reorder data such that unobserved time points are put at the bottom of the output
        reduce_data : bool
            if True, reduces the sequences to the max number of observed time points among the whole dataset; if True, order_data is set to True
        norm : list-like or NoneType
            if not None, used to standardize time series data; of shape (dim, 2), each element provides the mean and the standard deviation of a time series
        test_data : bool, optional
            True if the dataset provided is a test set i.e. it does not contain targets / labels
            (default is False)
        """

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
        """Returns the number of observations"""
        return len(self.sequences)    

    def __getitem__(self, idx):
        """Returns the sequence (and target / label if it exists) at index idx"""
        return (self.sequences[idx], self.target[idx]) if not self.test_data else self.sequences[idx]

