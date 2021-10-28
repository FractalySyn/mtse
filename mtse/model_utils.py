import torch
import torch.nn as nn
import torch.nn.functional as F

def rmspe(pred, true):
    return torch.mean(((true - pred) / true)**2)**0.5

def rmse(pred, true):
    return torch.mean((true - pred)**2)**0.5

def mape(pred, true):
    return torch.mean(torch.abs((true - pred) / true))

def acc(pred, true):
    return torch.mean(pred == true)

def evaluate_model(model, test_loader, dim, loss, argmax=False, device='cuda'):
    pred = []; true = []
    for test_batch, label in test_loader:
        test_batch, label = test_batch.to(device), label.float().to(device)
        with torch.no_grad():
            out = model(torch.cat((test_batch[:, :, :dim], test_batch[:, :, dim:2*dim]), 2), test_batch[:, :, -1])
        if argmax:
            pred = pred.argmax(1)
        pred.append(out); true.append(label)
    pred = torch.cat(pred, 0).squeeze()
    true = torch.cat(true, 0).squeeze()
    return loss(pred, true)

def predict(test_loader, device, model, model_type, dim):
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
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, patience=5, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    def __call__(self, val_loss):
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

