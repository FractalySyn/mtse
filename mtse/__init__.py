from mtse.models import mtse
from mtse.load_data import get_sample

__version__ = '0.1.1'

__doc__ = """
mste - Multi Time Series Encoders
#################################

Modules
-------
attention.py : Attention mechanism computation
data_utils.py : Dataset classes inherited from the torch.utils.data.Dataset class
encoders.py : Time series encoders
load_data.py : Contains a function for sample data loading. Imported to the package root
model_utils.py: Helper functions for training, evaluating and predicting
models.py: mtse class for handy use of the models of this package. Imported to the package root
"""

def __dir__():
    return list(globals().keys())