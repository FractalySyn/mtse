# Multi Time Series Encoders

The objective of this python package is to make easy the encoding and the classification/regression of multivariate time series (**mts**) data even when these are asynchronous. We say that data are of type **mts** when each observation is associated with multiple time series (e.g. the vital signs of a patient at a specific period).

## Installation

The current version has been developed in Python 3.7. It also works in Python 3.8. If you encounter an issue, please try to run it again in a virtual machine containing Python 3.7 or 3.8.

```bash
pip install mtse
```

## Sample code

```python
import mtse

### Load sample data ###
train, val, test, norm = mtse.get_sample(return_norm=True)

### Using the class `mtse` ###
mtan = mtse.mtse(device='cuda', seed=1, experiment_id='mtan')
mtan.load_data(train, val, test, norm=norm)
mtan.build_model('mtan', 'regression', learn_emb=True, early_stop=10)
mtan.train(cuda_empty_cache=True, lossf='mape', n_iters=200, save_startegy='best')
mtan.predict(checkpoint='best')
```

**More details and examples in the documentation**

## What can be implemented / improved

#### Encoders
  - [x] mTAN - Multi Time Attention Network - encoder
  - [ ] mTAN - Multi Time Attention Network - encoder-decoder
  - [ ] SeFT - Set Function for Time series
  - [ ] STraTS - Self-supervised Transformer for Time-Series
  - [ ] ODE-based encoders

Note that we only implemented the mTAN encoder as a baseline for now. At this stage, this model works only for supervised learning, meaning that it uses the target variable to compute the loss and update the encoder weights. Thus, the priority would be to implement an unsupervised encoder next (encoder-decoder models or self-supervised encoders).

#### Other features
  - Cross-validation evaluation, prediction and encoding
  - Support for other data inputs in the dataset classes (currently the `mtan_Dataset` class)
  - Support for time-series forecasting and inference tasks

## References

Satya Narayan Shukla and Benjamin Marlin, ["Multi-Time Attention Networks for Irregularly Sampled Time Series"](https://openreview.net/forum?id=4c0J6lwQ4_), *International Conference on Learning Representations*, 2021.
