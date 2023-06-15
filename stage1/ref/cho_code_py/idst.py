import numpy as np
from scipy.fftpack import dst

def idst(a, n=None):
    """
    Inverse discrete sine transform (Used in Poisson reconstruction)

    Parameters:
    a : 1D or 2D numpy array
        If a is a 1D array, it is the vector to be transformed.
        If a is a 2D array, the IDST operation is applied to each column.
    n : int, optional
        Length of the transformed output.

    Returns:
    b : 1D or 2D numpy array
        The transformed vector or matrix.
    """
    if n is None:
        if min(a.shape) == 1:
            n = len(a)
        else:
            n = a.shape[0]
    n2 = n + 1
    b = 2/n2 * dst(a, n=n, type=1)  # DST type 1 is equivalent to MATLAB's default DST
    return b
