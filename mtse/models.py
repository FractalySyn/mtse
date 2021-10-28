import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from random import SystemRandom
import time
from torch.utils.data import DataLoader, Dataset
import pickle

from .data_utils import transform_seq, mtan_Dataset
from .encoders import default_regressor, default_classifier, enc_mtan_reg
from .model_utils import *


class mtse:

    def __init__(self, device='auto', seed=None, experiment_id=None):
        
        self.order_data=True
        self.reduce_data=True
        self.batch_size=64
        self.norm=False

        self.experiment_id = int(SystemRandom().random()*100000) if experiment_id is None else experiment_id
        print('Experiment id:', experiment_id)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device!='cpu' else torch.device(device)
        print('Device:', device)
        
        if seed is not None:
            print(f'Setting seed to {seed}')
            torch.manual_seed(seed)
            np.random.seed(seed)
            torch.cuda.manual_seed(seed)  
    
    
    def load_data(self, train_data=None, val_data=None, test_data=None, order_data=True, reduce_data=True, norm=None, batch_size=64, shuffle=False):
        
        self.order_data=order_data
        self.reduce_data=reduce_data
        self.batch_size=batch_size

        print("\n### Loading data ###")
        if (train_data is not None and val_data is None) or (val_data is not None and train_data is None):
            raise ValueError('You must provide both train and validation set.')
        if train_data is None and test_data is None and val_data is None:
            raise ValueError('You must provide both train and validation set OR only test set OR all the three datasets.')
        
        self.train_loader = DataLoader(mtan_Dataset(train_data, order_data=order_data, reduce_data=reduce_data, norm=norm), batch_size=batch_size, shuffle=shuffle) if train_data is not None else None
        self.val_loader = DataLoader(mtan_Dataset(val_data, order_data=order_data, reduce_data=reduce_data, norm=norm), batch_size=batch_size, shuffle=shuffle) if val_data is not None else None
        self.test_loader = DataLoader(mtan_Dataset(test_data, test_data=True, order_data=order_data, reduce_data=reduce_data, norm=norm), batch_size=batch_size, shuffle=shuffle) if test_data is not None else None
        print('...done...')

        try:
            self.n_ts = self.train_loader.dataset.dim
        except:
            self.n_ts = self.test_loader.dataset.dim


    def build_model(self, encoder='mtan', model_type='regression', regressor=None, classifier=None, classif_out=2, seq_encoder=None, nhidden=64, embed_time=64, n_heads=1, learn_emb=True, optim='default', sched='default', lr=0.01, checkpoint_path=None, early_stop=0, min_delta=0):
        
        print('\n### Building model ###')
        self.model_type = model_type
        if encoder == 'mtan':
            self.ts_encoder = enc_mtan_reg(n_ts=self.n_ts, model_type=model_type, regressor=regressor, classifier=classifier, classif_out=classif_out, 
                                           seq_encoder=seq_encoder, nhidden=nhidden, embed_time=embed_time, n_heads=n_heads,
                                           learn_emb=learn_emb, device=self.device).to(self.device)
        else:
            raise ValueError(f"The {encoder} encoder is not supported.")
        
        self.params = list(self.ts_encoder.parameters())
        self.optimizer = torch.optim.Adam(self.params, lr=lr) if optim == 'default' else eval(optim)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=12, factor=0.8, verbose=True) if sched=='default' else eval(sched)
        
        if checkpoint_path is not None:
            print('\t\tloading pre-trained model')
            checkpoint = torch.load(checkpoint_path)
            try:
                self.ts_encoder.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            except:
                raise ValueError('Make sure that the parameters you provided in arguments match those of the pre-trained model')
        print('The model has', count_parameters(self.ts_encoder)/1e6, 'million parameters\n', 'Scheduler:', self.scheduler, '\nOptimizer:', self.optimizer)
        
        self.early_stop = early_stop
        if self.early_stop > 0:
            self.Early_stop = EarlyStopping(early_stop, min_delta=min_delta)

    
    def train(self, cuda_empty_cache=True, lossf='rmse', plot_results=True, save_plot=True, x_start=10, argmax=False, 
                          predict_strategy=None, save_strategy='best', val_loss_threshold=100., n_iters=100):
        
        print('\n### Training ###')
        if cuda_empty_cache and self.device.type == 'cuda':
            torch.cuda.empty_cache()
            print('GPU cache emptied')
        torch.backends.cudnn.enabled=False
        best_val_loss = float('inf'); total_time = 0.; train_losses, val_losses = [], []    
        lf = {'mae': nn.L1Loss(), 'mape': mape, 'rmse': rmse, 'rmspe': rmspe, 'mse': nn.MSELoss(), 'accuracy': acc, 'cross_entropy': nn.CrossEntropyLoss()}
        save_strategy = predict_strategy if predict_strategy == 'best' else save_strategy
        argmax = True if lossf == 'cross_entropy' else argmax
        try:
            criterion = lf[lossf]
        except:
            criterion = lossf
        for itr in range(1, n_iters + 1):        
            train_loss = 0; train_n = 0
            start_time = time.time()
            for train_batch, label in self.train_loader:
                train_batch, label = train_batch.to(self.device), label.to(self.device) 
                batch_len  = train_batch.shape[0]
                out = self.ts_encoder(train_batch[:, :, :self.n_ts*2], train_batch[:, :, -1])
                loss = criterion(out, label)
                train_loss += loss.item() * batch_len
                train_n += batch_len
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            val_loss = evaluate_model(self.ts_encoder, self.val_loader, dim=self.n_ts, loss=criterion, argmax=argmax, device=self.device)
            train_losses.append(train_loss/train_n); val_losses.append(val_loss)
            self.scheduler.step(val_loss)
            total_time += time.time() - start_time
            best_val_loss = min(best_val_loss, val_loss)
            print('Iter: {}, train_loss: {:.6f}, val_loss: {:.6f}'.format(itr, train_loss/train_n, val_loss))
            
            if save_strategy == 'best' and val_loss == best_val_loss and val_loss <= val_loss_threshold:
                print('\t\t saving best model')
                fname = str(self.experiment_id) + '_best.h5'
                torch.save({'epoch': itr, 'model_state_dict': self.ts_encoder.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict(), 'loss': -loss}, fname)
                self.best_model = self.ts_encoder

            if self.early_stop > 0:
                self.Early_stop(val_loss)
                if self.Early_stop.early_stop:
                    break
        
        print('Model trained. Best validation loss: {}, total time: {}, average time per iteration: {}'.format(best_val_loss, total_time, total_time/n_iters))
        
        if save_strategy == 'last' and val_loss <= val_loss_threshold:
            print('\t\t saving last model')
            torch.save({'epoch': itr, 'model_state_dict': self.ts_encoder.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict(), 'loss': -loss}, str(self.experiment_id) + '_last.h5')

        if plot_results:
            print(f'\n### Plotting results ###')        
            fig = plt.figure(figsize=(10, 6), dpi=100)
            train_losses = train_losses[x_start-1:]
            val_losses = val_losses[x_start-1:]
            plt.plot([x for x in range(len(train_losses))], [l for l in train_losses], label='training loss')
            plt.plot([x for x in range(len(val_losses))], [l for l in val_losses], label='validation loss')
            plt.ylabel(lossf); plt.xlabel("Iteration")
            plt.xticks(ticks=[x for x in range(len(train_losses))], labels=[x for x in range(x_start, len(train_losses)+x_start)])
            plt.legend(); plt.show()
            if save_plot:
                fig.savefig(f'results_{self.experiment_id}.png', format='png')
                with open(f'results_{self.experiment_id}.pickle', 'wb') as f:
                    pickle.dump(fig, f)
                print('figure saved as `results_{self.experiment_id}.png` and as `results_{self.experiment_id}.pickle` \n')
            self.results = fig        

        if predict_strategy is not None:
            print('\n### Predicting the target on the test set ###')
            if self.test_loader is None:
                raise ValueError('Make sure you have provided the test dataset')
            if predict_strategy == 'last':
                return predict(self.test_loader, self.device, self.ts_encoder, model_type=self.model_type, dim=self.n_ts)
            elif predict_strategy == 'best':
                try:
                    pt_encoder = self.best_model                    
                except:
                    raise ValueError('No best model found. Note that if `val_loss_threshold` is set to a very small value, there may not be a saved best model even though training has been performed.')      
                self.preds = predict(self.test_loader, self.device, pt_encoder, model_type=self.model_type, dim=self.n_ts)
            else:
                raise ValueError('`predict_strategy` must be equal to `best` or `last`.')
            return self.preds

    def predict(self, checkpoint='best'):
        print('\n### Predicting the target on the test set ###')
        if checkpoint == 'best':
            try:
                self.preds = predict(self.test_loader, self.device, self.best_model, model_type=self.model_type, dim=self.n_ts)
            except:
                raise ValueError('To predict from the best model, you need to train the model on this instance first. Note that if `val_loss_threshold` is set to a very small value, there may not be a saved best model even though training has been performed.')
        elif checkpoint == 'last':
            print('Caution: if no training has been performed, the prediction will be random and irrelevant.')
            try:
                self.preds = predict(self.test_loader, self.device, self.ts_encoder, model_type=self.model_type, dim=self.n_ts)
            except:
                raise ValueError('Make sure you have built the model first.')
        else:
            try:
                print('\t\t loading saved weights of the pretrained model.')
                pt_model = torch.load(checkpoint)
            except:
                raise ValueError('The checkpoint has not been found.')
            pt_encoder = self.ts_encoder
            pt_encoder.load_state_dict(pt_model['model_state_dict'])
            self.preds = predict(self.test_loader, self.device, pt_encoder, model_type=self.model_type, dim=self.n_ts)
        return self.preds

    
    def encode_ts(self, embed_pandas=True, data_to_embed='test', has_label=False, checkpoint='best'):
        
        print('\n### Embedding data ###')
        if type(data_to_embed) != str:
            data_loader = DataLoader(mtan_Dataset(data_to_embed, test_data=(not has_label), order_data=self.order_data, reduce_data=self.reduce_data, norm=self.norm), batch_size=self.batch_size, shuffle=False)
        elif data_to_embed == 'train' and self.train_loader is not None:
            data_loader = self.train_loader
            has_label=True
        elif data_to_embed == 'val' and self.val_loader is not None:
            data_loader = self.val_loader
            has_label=True
        elif data_to_embed == 'test' and self.test_loader is not None:
            data_loader = self.test_loader
        else:
            raise ValueError('Choose a dataset you have loaded with the `load_data()` method or input a new one.')
        n_ts = data_loader.dataset.dim

        if checkpoint == 'best':
            try:
                pt_encoder = self.best_model
            except:
                raise ValueError('To encode from the best model, you need to train the model on this instance first. Note that if `val_loss_threshold` is set to a very small value, there may not be a saved best model even though training has been performed.')
        elif checkpoint == 'last':
            print('Caution: if no training has been performed, the prediction will be random and irrelevant.')
            try:
                pt_encoder = self.ts_encoder
            except:
                raise ValueError('Make sure you have built the model first.')
        else:
            try:
                print('\t\t loading saved weights of the best model at epoch', checkpoint['epoch']) 
                pt_model = torch.load(checkpoint)
            except:
                raise ValueError('The checkpoint has not been found.')
            pt_encoder = self.ts_encoder
            pt_encoder.load_state_dict(pt_model['model_state_dict'])

        embeds, labels = torch.tensor([]), torch.tensor([])
        if has_label:
            for data_batch, label in data_loader:
                data_batch = data_batch.to(self.device)
                with torch.no_grad():
                    out = pt_encoder(data_batch[:, :, :2*n_ts], data_batch[:, :, -1], encode_ts=True).squeeze()                
                try:
                    embeds = torch.cat([embeds, out.cpu().detach()])
                except:
                    embeds = torch.cat([embeds, out])
                labels = torch.cat([labels, label])
            if embed_pandas:
                embeds = pd.DataFrame(embeds.numpy())
                embeds['target'] = labels
                return embeds
            else:
                return embeds.numpy(), np.array(labels)
        else:
            for data_batch in data_loader:
                data_batch = data_batch.to(self.device)
                with torch.no_grad():
                    out = pt_encoder(data_batch[:, :, :2*n_ts], data_batch[:, :, -1], encode_ts=True).squeeze()                
                try:
                    embeds = torch.cat([embeds, out.cpu().detach()])
                except:
                    embeds = torch.cat([embeds, out])
            if embed_pandas:
                return pd.DataFrame(embeds.numpy())
            else:
                return embeds.numpy()




def run_model(train_data=None, val_data=None, test_data=None, encoder='mtan', batch_size=64, shuffle=True, device='auto', 
              cuda_empty_cache=True, data_to_embed=None, has_label=True, embed_pandas=True, order_data=True, reduce_data=True,
              seed=None, model_type='regression', regressor=None, classifier=None, classif_out=2, seq_encoder=None,
              nhidden=64, embed_time=64, n_heads=1, learn_emb=True, optim='default', sched='default', early_stop=0, min_delta=0,
              n_iters=100, lr=0.01, lossf='rmse', plot_results=True, x_start=10, norm=None, argmax=False,
              predict_strategy=None, save_strategy='best', val_loss_threshold=100., encode_ts=False, checkpoint_path=None):  

    experiment_id = int(SystemRandom().random()*100000)
    print('Experiment id:', experiment_id)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device!='cpu' else torch.device(device)
    print('Device:', device)

    if seed is not None:
        print(f'Setting seed to {seed}')
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed(seed)  

    print("\n### Loading data ###")
    if cuda_empty_cache and device.type == 'cuda':
        torch.cuda.empty_cache()
    if predict_strategy != 'checkpoint':
        train_loader = DataLoader(mtan_Dataset(train_data, order_data=order_data, reduce_data=reduce_data, norm=norm), batch_size=batch_size, shuffle=shuffle) if train_data is not None else None
        val_loader = DataLoader(mtan_Dataset(val_data, order_data=order_data, reduce_data=reduce_data, norm=norm), batch_size=batch_size, shuffle=shuffle) if val_data is not None else None
    if predict_strategy is not None: 
        test_loader = DataLoader(mtan_Dataset(test_data, test_data=True, order_data=order_data, reduce_data=reduce_data, norm=norm), batch_size=batch_size, shuffle=shuffle) if test_data is not None else None

    print('\n### Building model ###')
    n_ts = train_loader.dataset.dim
    if encoder == 'mtan':
        ts_encoder = enc_mtan_reg(n_ts=n_ts, model_type=model_type, regressor=regressor, classifier=classifier, classif_out=classif_out, 
                                  seq_encoder=seq_encoder, nhidden=nhidden, embed_time=embed_time, n_heads=n_heads,
                                  learn_emb=learn_emb, device=device).to(device)
    else:
        raise ValueError(f"The encoder {encoder} is not supported.")

    params = list(ts_encoder.parameters())
    optimizer = torch.optim.Adam(params, lr=lr) if optim == 'default' else optim
    if optim != 'default':
        optimizer.add_param_group({'params': params})
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=12, factor=0.8, verbose=True) if sched=='default' else eval(sched)

    if checkpoint_path is not None:
        print('\t\tloading pre-trained model')
        checkpoint = torch.load(checkpoint_path)
        try:
            ts_encoder.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except:
            raise ValueError('Make sure that the parameters you provided in arguments match those of the pre-trained model')
    print('The model has', count_parameters(ts_encoder)/1e6, 'million parameters\n', 'Scheduler:', scheduler, '\nOptimizer:', optimizer)

    if not encode_ts and predict_strategy!='checkpoint':

        print('\n### Training ###')
        torch.backends.cudnn.enabled=False
        best_val_loss = float('inf'); total_time = 0.; train_losses, val_losses = [], []    
        lf = {'mae': nn.L1Loss(), 'mape': mape, 'rmse': rmse, 'rmspe': rmspe, 'mse': nn.MSELoss(), 'accuracy': acc, 'cross_entropy': nn.CrossEntropyLoss()}
        save_strategy = predict_strategy if predict_strategy == 'best' else save_strategy
        argmax = True if lossf == 'cross_entropy' else argmax
        try:
            criterion = lf[lossf]
        except:
            criterion = lossf
        if early_stop > 0:
            Early_stop = EarlyStopping(early_stop, min_delta=min_delta)

        for itr in range(1, n_iters + 1):        
            train_loss = 0; train_n = 0
            start_time = time.time()
            for train_batch, label in train_loader:
                train_batch, label = train_batch.to(device), label.to(device) 
                batch_len  = train_batch.shape[0]
                out = ts_encoder(train_batch[:, :, :n_ts*2], train_batch[:, :, -1])
                loss = criterion(out, label)
                train_loss += loss.item() * batch_len
                train_n += batch_len
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            val_loss = evaluate_model(ts_encoder, val_loader, dim=n_ts, loss=criterion, argmax=argmax, device=device)
            train_losses.append(train_loss/train_n); val_losses.append(val_loss)
            scheduler.step(val_loss)
            total_time += time.time() - start_time
            best_val_loss = min(best_val_loss, val_loss)
            print('Iter: {}, train_loss: {:.6f}, val_loss: {:.6f}'.format(itr, train_loss/train_n, val_loss))
            if save_strategy == 'best' and val_loss == best_val_loss and val_loss <= val_loss_threshold:
                print('\t\t saving model')
                fname = str(experiment_id) + '_best.h5'
                torch.save({'epoch': itr, 'model_state_dict': ts_encoder.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'loss': -loss}, fname)
            if early_stop > 0:
                Early_stop(val_loss)
                if Early_stop.early_stop:
                    break

        print('Model trained. Best validation loss: {}, total time: {}, average time per iteration: {}'.format(best_val_loss, total_time, total_time/n_iters))

        if save_strategy == 'last' and val_loss <= val_loss_threshold:
            print('\t\t saving model')
            torch.save({'epoch': itr, 'model_state_dict': ts_encoder.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'loss': -loss}, str(experiment_id) + '_last.h5')

        if plot_results:
            print(f'\n### Plotting results ###\nfigure saved as `results_{experiment_id}.png` and as `results_{experiment_id}.pickle` \n')        
            fig = plt.figure(figsize=(10, 6), dpi=100)
            train_losses = train_losses[x_start-1:]
            val_losses = val_losses[x_start-1:]
            plt.plot([x for x in range(len(train_losses))], [l for l in train_losses], label='training loss')
            plt.plot([x for x in range(len(val_losses))], [l for l in val_losses], label='validation loss')
            plt.ylabel(lossf); plt.xlabel("Iteration")
            plt.xticks(ticks=[x for x in range(len(train_losses))], labels=[x for x in range(x_start, len(train_losses)+x_start)])
            plt.legend(); plt.show()
            fig.savefig(f'results_{experiment_id}.png', format='png')
            with open(f'results_{experiment_id}.pickle', 'wb') as f:
                pickle.dump(fig, f)
    
    if not encode_ts and predict_strategy is not None:
        print('\n### Predicting the target on the test set ###')
        if test_data is None:
            raise ValueError('Make sure you have provided the test data you want to predict')
        if predict_strategy == 'last' or predict_strategy == 'checkpoint':
            return predict(test_loader, device, ts_encoder, model_type=model_type, dim=n_ts)
        elif predict_strategy == 'best':
            checkpoint = torch.load(fname)
            ts_encoder.load_state_dict(checkpoint['model_state_dict'])
            print('\t\t loading saved weights of the best model at epoch', checkpoint['epoch'])            
            return predict(test_loader, device, ts_encoder, model_type=model_type, dim=n_ts)

    
    elif encode_ts:
        print('\n### Embedding data ###')
        data_loader = DataLoader(mtan_Dataset(data_to_embed, test_data=(not has_label), order_data=order_data, reduce_data=reduce_data, norm=norm), batch_size=batch_size, shuffle=False)
        embeds, labels = torch.tensor([]), torch.tensor([])
        if has_label:
            for data_batch, label in data_loader:
                data_batch = data_batch.to(device)
                with torch.no_grad():
                    out = ts_encoder(data_batch[:, :, :2*n_ts], data_batch[:, :, -1], encode_ts=True).squeeze()                
                try:
                    embeds = torch.cat([embeds, out.cpu().detach()])
                except:
                    embeds = torch.cat([embeds, out])
                labels = torch.cat([labels, label])
            if embed_pandas:
                embeds = pd.DataFrame(embeds.numpy())
                embeds['target'] = labels
                return embeds
            else:
                return embeds.numpy(), np.array(labels)
        else:
            for data_batch in data_loader:
                data_batch = data_batch.to(device)
                with torch.no_grad():
                    out = ts_encoder(data_batch[:, :, :2*n_ts], data_batch[:, :, -1], encode_ts=True).squeeze()                
                try:
                    embeds = torch.cat([embeds, out.cpu().detach()])
                except:
                    embeds = torch.cat([embeds, out])
            if embed_pandas:
                return pd.DataFrame(embeds.numpy())
            else:
                return embeds.numpy()