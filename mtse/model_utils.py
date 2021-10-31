import torch
import torch.nn as nn
import torch.nn.functional as F

def rmspe(pred, true):
    """Computes RMSPE"""
    return torch.mean(((true - pred) / true)**2)**0.5

def rmse(pred, true):
    """Computes RMSE"""
    return torch.mean((true - pred)**2)**0.5

def mape(pred, true):
    """Computes MAPE"""
    return torch.mean(torch.abs((true - pred) / true))

def acc(pred, true):
    """Computes Accuracy"""
    return torch.mean(pred == true)

def evaluate_model(model, data_loader, dim, loss, argmax=False, device='cuda'):
    """Predicts and computes the loss over the provided data
    
    Parameters
    ----------
    model : torch.nn.Module
        model containing the weights to compute the predictions
    data_loader : torch.utils.data.DataLoader
        data to evaluate
    dim : int
        number of time series
    loss : torch loss
        loss function
    argmax : bool, optional
        if True, an arg max is applied to predicted values (default is False)
    device : str, optional
        'cuda' or 'cpu', (default is 'cuda')

    Returns
    -------
    float
    """

    pred = []; true = []
    for data_batch, label in data_loader:
        data_batch, label = data_batch.to(device), label.float().to(device)
        with torch.no_grad():
            out = model(torch.cat((data_batch[:, :, :dim], data_batch[:, :, dim:2*dim]), 2), data_batch[:, :, -1])
        if argmax:
            pred = pred.argmax(1)
        pred.append(out); true.append(label)
    pred = torch.cat(pred, 0).squeeze()
    true = torch.cat(true, 0).squeeze()
    return loss(pred, true)

def predict(test_loader, device, model, model_type, dim):
    """
    Parameters
    ----------
    test_loader : torch.utils.data.DataLoader
        data to predict on
    device : str
        'cuda' or 'cpu'
    model : torch.nn.Module
        model containing the weights to compute the predictions
    model_type : str
        'regression' or 'classification', for the latter an arg max is applied
    dim : int
        number of time series

    Returns
    -------
    torch.tensor
    """
    pred = []
    for test_batch in test_loader:
        test_batch = test_batch.to(device)
        with torch.no_grad():
            out = model(test_batch[:, :, :dim*2], test_batch[:, :, -1])
            if model_type=='classification':
                out = out.argmax(1)
        pred.append(out)
    return torch.cat(pred, 0).squeeze()

def count_parameters(model):
    """Returns the number of weights that can be trained in the provided model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.

    Attributes
    ----------
    patience : int
        how many epochs to wait before stopping when loss is not improving
    min_delta : float
        minimum difference between new loss and old loss for new loss to be considered as an improvement
    counter : int
        number of epochs without improvement
    best_loss : float or NoneType
        validation loss of the last epoch for which the counter was equal to 0
    early_stop : bool
        if True, the training will break
    """
    def __init__(self, patience=5, min_delta=0.):
        """
        Parameters
        ---------
        patience : int, optional
            how many epochs to wait before stopping when loss is not improving (default is 5)
        min_delta : float, optional
            minimum difference between new loss and old loss for new loss to be considered as an improvement
            (default is 0.)
        """

        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        """
        Parameters
        ----------
        val_loss : torch loss or float
        """
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                print('Early stopping')
                self.early_stop = True

