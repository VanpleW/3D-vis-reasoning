from scipy.signal import convolve2d
from scipy.signal import fftconvolve
from scipy.fftpack import fft2, ifft2
from scipy.optimize import fmin_cg
import numpy as np
from pypher import psf2otf, otf2psf

def compute_Ax(x, p):
    x_f = psf2otf(x, p['img_size'])
    y = otf2psf(p['m'] * x_f, p['psf_size'])
    y = y + p['lambda'] * x
    return y

def estimate_psf(blurred_x, blurred_y, latent_x, latent_y, weight, psf_size):
    latent_xf = fft2(latent_x)
    latent_yf = fft2(latent_y)
    blurred_xf = fft2(blurred_x)
    blurred_yf = fft2(blurred_y)

    b_f = np.conj(latent_xf) * blurred_xf + np.conj(latent_yf) * blurred_yf
    b = np.real(ifft2(b_f, psf_size))

    p = {
        'm': np.conj(latent_xf) * latent_xf + np.conj(latent_yf) * latent_yf,
        'img_size': blurred_xf.shape,
        'psf_size': psf_size,
        'lambda': weight,
    }

    psf = np.ones(psf_size) / np.prod(psf_size)
    psf = fmin_cg(psf, b, args=(compute_Ax, p), maxiter=20, gtol=1e-5)

    psf[psf < np.max(psf) * 0.05] = 0
    psf = psf / np.sum(psf)

    return psf
