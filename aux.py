import numpy as np
import os

cc = np.concatenate


class Generic(object):
    """Class for generic object."""
    
    def __init__(self, **kwargs):
        
        for k, v in kwargs.items():
            self.__dict__[k] = v
            

class Ephemeral(object):
    """
    Class to contain temporary variables.
    
    Usage:
    >>> E = E()
    >>> E.num = (3 + 1 + 5 + 6 + 3)
    >>> E.denom = (2 + 4 + 3 + 2)
    >>> x = E.num / E.denom
    >>> E.clear()
    """
    
    def clear(self):
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


# DATA PROCESSING
def get_segments(x, t=None):
    """
    Return the numerical indices indicating the segments of non-False x-values.
    :param x: boolean time-series
    :param t: vector containing indices to use if not 0 to len(x)
    :return: starts, ends, which are numpy arrays containing the start and end idxs of
        segments of consecutive Trues (end idxs are according to Python convention, e.g.,
        np.array([False, False, False, True, True, False]) yields (array([3]), array([5]))
    """

    if t is None:
        t = np.arange(len(x), dtype=int)

    starts = t[(np.diff(cc([[False], x]).astype(int)) == 1).nonzero()[0]]
    ends = t[(np.diff(cc([x, [False]]).astype(int)) == -1).nonzero()[0]] + 1

    return starts, ends
