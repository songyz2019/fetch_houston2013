from .core.fetch_houston2013 import fetch_houston2013
from .core.fetch_muufl import fetch_muufl
from .core.fetch_trento import fetch_trento
from .core.common import DataMetaInfo

from .util.split_spmatrix import split_spmatrix
from .util.fileio import read_roi


__all__ = ['fetch_houston2013', 'fetch_muufl', 'fetch_trento', 'split_spmatrix', 'read_roi', 'DataMetaInfo']


# If torch is imported, add these API
try:
    import torch
    from .torch.datasets import Houston2013, Muufl, Trento
    from .util.lbl2rgb import lbl2rgb

    __all__ += ['Houston2013', 'Muufl', 'Trento', 'lbl2rgb']

except ImportError:
    pass



