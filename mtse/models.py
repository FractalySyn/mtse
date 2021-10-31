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
from .encoders import default_regressor, default_classifier, mtan_enc
from .model_utils import *


class mtse:
    """
    Main class for training, encoding, predicting
    
    Attributes
    ----------
    experiment_id : str or numeric type
        ID of the experiment, used for saving model related objects
    device : torch.device
    order_data : bool
        if True, reorder data when data is loaded
    reduce_data : bool
        if True, reduces the sequences size when data is loaded
    norm : list-like or NoneType
        if not None, used to standardize time series data
    batch_size : int
    train_loader : torch.utils.data.DataLoader
    test_loader : torch.utils.data.DataLoader
    val_loader : torch.utils.data.DataLoader
    n_ts : int
        number of time series
    model_type : str
        'regression' or 'classification'
    ts_encoder : torch.nn.Module
    params : list
        model parameters
    optimizer : torch.optim.Optimizer
    scheduler : torch.optim.lr_scheduler.LRScheduler
    early_stop : int
        early stopping threshold
    Early_stop : EarlyStopping
        early stopping scheduler
    best_model : torch.nn.Module
    results : pyplot figure
    preds : torch.tensor

    Methods
    -------
    load_data
        Loads data
    build_model
        Builds the encoder architecture
    train
        Training method
    predict
        Prediction method
    encode_ts
        Multivariate time series embedding method
    """

    def __init__(self, device='cuda', seed=None, experiment_id=None):
        """
        Parameters
        ----------
        device : str, optional
            'cpu' or 'cuda'; if set to 'cuda', tests GPU availability first (default is 'cuda')
        seed : int or NoneType, optional
            if not None, set the seed everywhere to the specified integer, allowing for reproducibility (default is None)
        experiment_id : numeric type or string or NoneType, optional
            ID of the experiment, used for saving model related objects
        """

        self.order_data=True
        self.reduce_data=True
        self.batch_size=64
        self.norm=None

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
        """
        Loads data

        Parameters
        ----------
        train_data : list of list or similar or NoneType, optional
            shape: list-like of n observations, each observation is a tuple or a list of size N = n_ts*2+2, the n-th element is a numeric type indicating 
                   the target / label value, the first N-1 elements are ndarrays containing the n_ts time series, their n_ts mask and 
                   the time points (default is None)
        val_data : list of list or similar or NoneType, optional
            shape: same as train_data (default is None)
        test_data : list of list or similar or NoneType, optional
            shape: same as train_data, except that each element is of size N = n_ts*2+1 because target / label values should not be provided (default is None)
        order_data : bool, optional
            if True, reorder data such that unobserved time points are put at the bottom of the output (default is True)
        reduce_data : bool, optional
            if True, reduces the sequences to the max number of observed time points among the whole dataset; if True, order_data is set to True (default is True)
        norm : list-like or NoneType, optional
            if not None, used to standardize time series data; of shape (dim, 2), each element provides the mean and the standard deviation of a time series (default is None)
        batch_size : int, optional
            (default is 64)
        shuffle : bool, optional
            if set to True, data are shuffled according to the seed value of the instance (default is False)

        Raises
        ------
        ValueError
        """
        
        self.order_data=order_data
        self.reduce_data=reduce_data
        self.batch_size=batch_size
        self.norm=norm

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


    def build_model(self, encoder='mtan', model_type='regression', regressor=None, classifier=None, classif_out=2, seq_encoder=None, nhidden=64, 
                    embed_time=64, n_heads=1, learn_emb=False, optim='default', sched='default', lr=0.01, early_stop=0, min_delta=0., 
                    checkpoint_path=None, pdrop=0.5, query=None, freq=1., cuda_empty_cache=True):
        """
        Builds the encoder architecture

        Parameters
        ----------
        encoder : str, optional
            encoder architecture, currently supports 'mtan' (deafult is mtan)
        model_type : str, optional
            'regression' or 'classification' (default is regression)
        regressor : torch model or NoneType, optional
            used to specify a custom top regressor (default is None)
        classifier : torch model or NoneType, optional
            used to specify a custom top classifier (default is None)
        classif_out : int, optional
            number of classes in case of classification (default is 2)
        seq_encoder : torch model or NoneType, optional
            used to specify a custom RNN (default is None)
        nhidden : int, optional
            dimension of the first hidden layer (default is 64)
        embed_time : int, optional
            dimension used to embed time (default is 64)
        n_heads : int, optional
            number of attention heads, such as embed_time / n_heads is an integer (default is 1)
        learn_emb : bool, optional
            if True, time embedding is learnt by the model (default is False)
        optim : str, optional
            if set to 'default', an Adam optimizer is used; to specify another torch.optim Optimizer class, 
            provide it in a string as follows 'torch.optim.OPTIMIZER(params=self.params, **kwargs)'. Note
            that `params=self.params` is strictly required. 
            For example: `mymodel.build_model(..., optim='torch.optim.SGD(params=self.params, lr=1e-4)', ...)`
            (default is 'default')
        sched : str, optional
            if set to 'default', a ReduceLROnPlateau with patience 12 and factor 0.8 is used; to specify another LR 
            scheduler, provide it in a string as follows 'torch.optim.lr_scheduler.LRSCHEDULER(self.optimizer, **kwargs)'.
            Note that the first argument `self.optimizer` is strictly required. 
            For example: `mymodel.build_model(..., sched='torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=5)', ...)`
            (default is 'default')
        lr : float, optional
            default learning rate used in the Adam optimizer when `optim` is set to 'default' (default is 0.01)
        early_stop : int, optional
            early stopping threshold, initialized only if set to a value strictly positive (default is 0)
        min_delta : float, optional
            minimum difference of the loss to be considered as an improving step in the Early Stopping process;
            when such an improvement occurs, the Early Stopping process is re-initialized
            (default is 0.)
        checkpoint_path : str or NoneType, optional
            if a checkpoint path is specified, it will be loaded; allows training, prediction or encoding to resume
            from a pre-trained model
            Caution: model architecture arguments must be the same as those of the pre-trained model
            (default is None)        
        pdrop : float, optional
            probability of dropout (default is 0.5)
        query : torch.tensor or NoneType, optional
            query values dimension; if None, it is set to `embed_time` (default is None)
        freq : float, optional
            parameter of the time embedding when static, i.e. `learn_emb` is set to False (default is 1.)  
        cuda_empty_cache : bool, optional
            if True and `device` has been set to 'cuda' in this instance, empties cuda cache before sending the 
            model to the GPU (default is True)      

        Raises
        ------
        ValueError
        """

        print('\n### Building model ###')

        if cuda_empty_cache and self.device.type == 'cuda':
            torch.cuda.empty_cache()
            print('GPU cache emptied')

        self.model_type = model_type
        if encoder == 'mtan':
            self.ts_encoder = mtan_enc(n_ts=self.n_ts, model_type=model_type, regressor=regressor, classifier=classifier, classif_out=classif_out, 
                                           seq_encoder=seq_encoder, nhidden=nhidden, embed_time=embed_time, n_heads=n_heads,
                                           learn_emb=learn_emb, device=self.device, pdrop=pdrop, query=query, freq=freq).to(self.device)
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
        
        self.early_stop = early_stop
        if self.early_stop > 0:
            self.Early_stop = EarlyStopping(early_stop, min_delta=min_delta)

        print('The model has', count_parameters(self.ts_encoder)/1e6, 'million parameters\n', 'Scheduler:', self.scheduler, '\nOptimizer:', self.optimizer,
                '\nEarly Stopping:', self.early_stop, 'steps')

    
    def train(self, lossf='rmse', plot_results=True, save_plot=True, x_start=10, argmax=False, 
              predict_strategy=None, save_strategy='best', val_loss_threshold=100., n_iters=100):
        """
        Parameters
        ----------
        lossf : str or Callable, optional
            loss function; can be 'mae', 'mape', 'mse', 'rmse', 'rmspe', 'accuracy', 'cross_entropy' or 
            a function taking as input two tensors, predicted values and true values, and returning a
            loss computation that allows for backpropagation. 
            See PyTorch documentation for more details of how to create a custom loss function.
            (default is 'rmse')
        plot_results : bool, optional
            if True, a simple plot is showed at the end of training, displaying the evolution of loss on
            the training and the validation sets. 
            The figure can then be accessed via the attribute `results`.
            (default is True)
        save_plot : bool, optional
            if True, the results plot is saved as a png and as a pickle file (default is True)
        x_start : int, optional
            the left limit of the plot x-axis, useful when the first iterations exhibit high losses
            (default is 10)
        argmax : bool, optional
            if True, the predictions are passed to an arg max function during model evaluation; use this
            only when the classifier loss function is computed over a unique prediction per item like
            accuracy. Not suited for Cross Entropy.
            (default is False)
        predict_strategy : str or NoneType, optional
            if set to None, no prediction is performed after training; if set to 'best', the test data target
            is predicted using the best model with respect to the validation loss; if set to 'last', the 
            last iteration model is used to predict over the test set
            (default is None)
        save_strategy : str or NoneType, optional
            if set to None, no model is saved; if set to 'best', the best model is saved; if set to 'last',
            the last iteration model is saved.
            Caution: if `predict_strategy` is set to 'best', `save_strategy` becomes 'best' as well.
            The best model becomes accessible via the attribute `best_model`.
            (default is 'best')
        val_loss_threshold : float, optional
            the minimum loss value to reach before saving the best model (default is 100.)
        n_iters : int, optional
            number of epochs (default is 100)

        Returns
        -------
        if `predict_strategy` is not None, the method returns the predictions as a torch.tensor

        Raises
        ------
        NotImplementedError, ValueError
        """

        print('\n### Training ###')
        
        torch.backends.cudnn.enabled=False
        best_val_loss = float('inf'); total_time = 0.; train_losses, val_losses = [], []    
        save_strategy = predict_strategy if predict_strategy == 'best' else save_strategy
        argmax = True if lossf == 'cross_entropy' else argmax
        
        lf = {'mae': nn.L1Loss(), 'mape': mape, 'rmse': rmse, 'rmspe': rmspe, 'mse': nn.MSELoss(), 'accuracy': acc, 'cross_entropy': nn.CrossEntropyLoss()}
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
            if x_start <= 0:
                raise ValueError('x_start should be greater than 0')
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
                    raise NotImplementedError('No best model found. Note that if `val_loss_threshold` is set to a very small value, there may not be a saved best model even though training has been performed.')      
                self.preds = predict(self.test_loader, self.device, pt_encoder, model_type=self.model_type, dim=self.n_ts)
            else:
                raise ValueError('`predict_strategy` must be equal to `best` or `last`.')
            return self.preds

    def predict(self, checkpoint='best'):
        """
        Parameters
        ----------
        checkpoint : str, optional
            either 'best' to predict using the best model, or 'last' to use the last model, or a specific path to a pre-trained model
            (default is 'best')

        Returns
        -------
        predictions performed on the test data as a torch.tensor

        Raises
        ------
        NotImplementedError, ValueError
        """
        
        print('\n### Predicting the target on the test set ###')

        if checkpoint == 'best':
            try:
                self.preds = predict(self.test_loader, self.device, self.best_model, model_type=self.model_type, dim=self.n_ts)
            except:
                raise NotImplementedError('To predict from the best model, you need to train the model on this instance first. Note that if `val_loss_threshold` is set to a very small value, there may not be a saved best model even though training has been performed.')
        elif checkpoint == 'last':
            print('Caution: if no training has been performed, the prediction will be random and irrelevant.')
            try:
                self.preds = predict(self.test_loader, self.device, self.ts_encoder, model_type=self.model_type, dim=self.n_ts)
            except:
                raise NotImplementedError('Make sure you have built the model first.')
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

    
    def encode_ts(self, checkpoint='best', data_to_embed='test', embed_pandas=True, has_label=False):
        """
        Embeds multivariate time series

        Parameters
        ----------
        checkpoint : str, optional
            either 'best' to encode using the best model, or 'last' to use the last model, or a specific path to a pre-trained model
            (default is 'best')
        data_to_embed : str or a list of lists or similar, optional
            can be 'train', 'test' or 'val', in which case the data used is one of those loaded with the `load_data` method;
            can be a dataset in the same format as the one required in the `load_data` method, refer to this method documentation
            (default is 'test')
        embed_pandas : bool, optional
            if True, returns a pandas DataFrame of shape (N, embed_time)
            if False, returns a numpy array
            (default is True)
        has_label : bool, optional
            used to specify if the data provided (if not a string) contains target / label values
            (default is False)

        Returns
        -------
        if `embed_pandas` is set to True, returns a pandas DataFrame of shape (N, embed_time) containing the embeddings for each observation
        if `embed_pandas` is set to False, returns a numpy array containing the embeddings for each observation

        Raises
        ------
        NotImplementedError, ValueError
        """
        
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
            raise NotImplementedError('Choose a dataset you have loaded with the `load_data()` method or input a new one.')
        n_ts = data_loader.dataset.dim

        if checkpoint == 'best':
            try:
                pt_encoder = self.best_model
            except:
                raise NotImplementedError('To encode from the best model, you need to train the model on this instance first. Note that if `val_loss_threshold` is set to a very small value, there may not be a saved best model even though training has been performed.')
        elif checkpoint == 'last':
            print('Caution: if no training has been performed, the prediction will be random and irrelevant.')
            try:
                pt_encoder = self.ts_encoder
            except:
                raise NotImplementedError('Make sure you have built the model first.')
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
