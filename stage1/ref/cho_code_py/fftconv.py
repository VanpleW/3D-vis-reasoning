from scipy.signal import fftconvolve
import numpy as np
from pypher import psf2otf

def fftconv(I, filt, b_otf=False):
    if I.shape[2] == 3:
        H, W, ch = I.shape
        otf = psf2otf(filt, [H, W])
        cI = np.zeros_like(I)
        cI[:,:,0] = fftconv(I[:,:,0], otf, True)
        cI[:,:,1] = fftconv(I[:,:,1], otf, True)
        cI[:,:,2] = fftconv(I[:,:,2], otf, True)
        return cI
    else:
        if b_otf:
            cI = np.real(np.fft.ifft2(np.fft.fft2(I) * filt))
        else:
            cI = np.real(np.fft.ifft2(np.fft.fft2(I) * psf2otf(filt, I.shape)))
        return cI
