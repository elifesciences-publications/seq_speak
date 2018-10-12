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
