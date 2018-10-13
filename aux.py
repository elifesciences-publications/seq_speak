import numpy as np
import os


class Generic(object):
    """Class for generic object."""
    
    def __init__(self, **kwargs):
        
        for k, v in kwargs.items():
            self.__dict__[k] = v
            

class TempContainer(object):
    """
    Class to contain temporary variables.
    
    Usage:
    >>> TMP = TempContainer()
    >>> TMP.num = (3 + 1 + 5 + 6 + 3)
    >>> TMP.denom = (2 + 4 + 3 + 2)
    >>> x = TMP.num / TMP.denom
    >>> TMP.reset()
    """
    
    def reset(self):
        self.__dict__ = {}

        
# MATH

def sgmd(x):
    """Sigmoid (logistic) function."""
    return 1 / (1 + np.exp(-x))


def lognormal_mu_sig(mean, std):
    """Get log-normal params from mean and std."""
    if mean <= 0:
        raise ValueError('Mean must be > 0 for log-normal distribution')
    
    b = 1 + (std**2)/(mean**2)
    
    mu = np.log(mean/np.sqrt(b))
    sig = np.sqrt(np.log(b))
    
    return mu, sig
