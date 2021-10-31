import pickle
import pkg_resources

def get_sample(return_norm=True):
    """
    Parameters
    ----------
    return_norm : bool, optional
        if True, load the normalization values (mean, std) 
        (default is True)
    
    Returns
    -------
    sample train, val and test data, and normalization values if requested
    """
    #stream = pkg_resources.resource_stream(__name__, 'data/train.pickle')
    train = pickle.load(pkg_resources.resource_stream('mtse', "data/train.pkl"))
    test = pickle.load(pkg_resources.resource_stream('mtse', "data/test.pkl"))
    val = pickle.load(pkg_resources.resource_stream('mtse', "data/val.pkl"))
    if return_norm:
        return train, val, test, pickle.load(pkg_resources.resource_stream('mtse', "data/norm.pkl"))
    else:
        return train, val, test


        