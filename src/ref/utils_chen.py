import numpy as np
from scipy.signal import convolve2d
from scipy.misc import imresize
from scipy.interpolate import interp2d


def init_kernel(minsize):
    k = np.zeros((minsize, minsize))
    k[(minsize - 1)//2, (minsize - 1)//2:(minsize - 1)//2+1] = 0.5
    return k

def downSmpImC(I, ret):
    if ret == 1:
        return I

    sig = 1/np.pi*ret
    g0 = np.arange(-50, 51)*2*np.pi
    sf = np.exp(-0.5*g0**2*sig**2)
    sf = sf / np.sum(sf)
    csf = np.cumsum(sf)
    csf = np.minimum(csf, csf[::-1])
    ii = np.where(csf > 0.05)

    sf = sf[ii]
    I = convolve2d(sf, sf.transpose(), I, 'valid')

    gx, gy = np.meshgrid(np.arange(1, I.shape[1]+1, 1/ret), np.arange(1, I.shape[0]+1, 1/ret))
    sI = interp2d(I, gx, gy, 'bilinear')
    return sI

def resizeKer(k, ret, k1, k2):
    k = imresize(k, ret)
    k = np.maximum(k, 0)
    k = fixsize(k, k1, k2)
    if np.max(k) > 0:
        k = k / np.sum(k)
    return k

def fixsize(f, nk1, nk2):
    k1, k2 = f.shape

    while k1 != nk1 or k2 != nk2:
        if k1 > nk1:
            s = np.sum(f, axis=1)
            if s[0] < s[-1]:
                f = f[1:, :]
            else:
                f = f[:-1, :]

        if k1 < nk1:
            s = np.sum(f, axis=1)
            if s[0] < s[-1]:
                tf = np.zeros((k1+1, f.shape[1]))
                tf[:k1, :] = f
                f = tf
            else:
                tf = np.zeros((k1+1, f.shape[1]))
                tf[1:k1+1, :] = f
                f = tf

        if k2 > nk2:
            s = np.sum(f, axis=0)
            if s[0] < s[-1]:
                f = f[:, 1:]
            else:
                f = f[:, :-1]

        if k2 < nk2:
            s = np.sum(f, axis=0)
            if s[0] < s[-1]:
                tf = np.zeros((f.shape[0], k2+1))
                tf[:, :k2] = f
                f = tf
            else:
                tf = np.zeros((f.shape[0], k2+1))
                tf[:, 1:k2+1] = f
                f = tf

        k1, k2 = f.shape

    return f

