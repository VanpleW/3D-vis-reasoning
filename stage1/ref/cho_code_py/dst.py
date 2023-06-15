import numpy as np

def dst(a, n=None):
    """
    DST   Discrete sine tranform         (Used in Poisson reconstruction)
    Y = DST(X) returns the discrete sine transform of X.
    The vector Y is the same size as X and contains the
    discrete sine transform coefficients.
    Y = DST(X,N) pads or truncates the vector X to length N
    before transforming.
    If X is a matrix, the DST operation is applied to each
    column. This transform can be inverted using IDST.
    """

    if np.min(a.shape) == 1:
        if a.shape[1] > 1:
            do_trans = 1
        else:
            do_trans = 0
        a = a.flatten()
    else:
        do_trans = 0

    if n is None:
        n = a.shape[0]

    m = a.shape[1]

    # Pad or truncate a if necessary
    if a.shape[0] < n:
        aa = np.zeros((n,m))
        aa[0:a.shape[0], :] = a
    else:
        aa = a[0:n, :]

    y = np.zeros((2*(n+1),m))
    y[1:n+1, :] = aa
    y[n+2:2*(n+1), :] = -np.flipud(aa)
    yy = np.fft.fft(y)
    b = yy[1:n+1, :]/(-2*1j)

    if np.isrealobj(a):
        b = np.real(b)

    if do_trans:
        b = np.transpose(b)

    return b
