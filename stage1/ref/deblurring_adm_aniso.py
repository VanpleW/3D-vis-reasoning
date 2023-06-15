import numpy as np
from scipy.fftpack import fft2, ifft2
from scipy.signal import convolve2d
from pypher import psf2otf 

def computeDenominator(y, k):
    sizey = y.shape
    otfk  = psf2otf(k, sizey) 
    Nomin1 = np.conj(otfk) * fft2(y)
    Denom1 = abs(otfk) ** 2
    Denom2 = abs(psf2otf([1,-1],sizey)) ** 2 + abs(psf2otf([1,-1],sizey).T) ** 2
    return Nomin1, Denom1, Denom2


def deblurring_adm_aniso(B, k, lambda_, alpha):
    beta = 1/lambda_
    beta_rate = 2*np.sqrt(2)
    beta_min = 0.001

    m, n = B.shape
    I = B

    if ((k.shape[0] % 2 != 1) | (k.shape[1] % 2 != 1)):
        print('Error - blur kernel k must be odd-sized.')
        return

    Nomin1, Denom1, Denom2 = computeDenominator(B, k)
    Ix = np.concatenate((np.diff(I, 1, 1), (I[:,0] - I[:,n-1])[:, None]), axis=1)
    Iy = np.concatenate((np.diff(I, 1, 0), (I[0,:] - I[m-1,:])[None, :]), axis=0)

    while beta > beta_min:
        gamma = 1/(2*beta)
        Denom = Denom1 + gamma*Denom2

        if alpha==1:
            Wx = np.maximum(np.abs(Ix) - beta*lambda_, 0)*np.sign(Ix)
            Wy = np.maximum(np.abs(Iy) - beta*lambda_, 0)*np.sign(Iy)
        else:
            Wx = solve_image(Ix, 1/(beta*lambda_), alpha)
            Wy = solve_image(Iy, 1/(beta*lambda_), alpha)

        Wxx = np.concatenate((Wx[:,n-1] - Wx[:, 0], -np.diff(Wx, 1, 1)), axis=1)
        Wxx = Wxx + np.concatenate((Wy[m-1,:] - Wy[0, :], -np.diff(Wy, 1, 0)), axis=0)

        Fyout = (Nomin1 + gamma*fft2(Wxx))/Denom
        I = np.real(ifft2(Fyout))

        Ix = np.concatenate((np.diff(I, 1, 1), (I[:,0] - I[:,n-1])[:, None]), axis=1)
        Iy = np.concatenate((np.diff(I, 1, 0), (I[0,:] - I[m-1,:])[None, :]), axis=0)

        beta = beta/2
    return I
