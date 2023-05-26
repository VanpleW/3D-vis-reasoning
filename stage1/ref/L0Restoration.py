from scipy.signal import convolve2d
from scipy.fftpack import fft2, ifft2
import numpy as np


def L0Restoration(Im, kernel, lambda_, kappa=2.0):
    H, W = Im.shape
    Im = wrap_boundary_liu(Im, opt_fft_size([H, W] + kernel.shape - 1))

    S = Im
    betamax = 1e5
    fx = np.array([1, -1])
    fy = np.array([[1], [-1]])

    N, M, D = Im.shape
    sizeI2D = [N, M]
    otfFx = psf2otf(fx, sizeI2D)
    otfFy = psf2otf(fy, sizeI2D)

    KER = psf2otf(kernel, sizeI2D)
    Den_KER = np.abs(KER)**2

    Denormin2 = np.abs(otfFx)**2 + np.abs(otfFy)**2
    if D > 1:
        Denormin2 = np.repeat(Denormin2, D, axis=2)
        KER = np.repeat(KER, D, axis=2)
        Den_KER = np.repeat(Den_KER, D, axis=2)

    Normin1 = np.conj(KER) * fft2(S)

    beta = 2 * lambda_
    while beta < betamax:
        Denormin = Den_KER + beta * Denormin2
        h = np.concatenate((np.diff(S, 1, 1), S[:, :1, :] - S[:, -1:, :]), axis=1)
        v = np.concatenate((np.diff(S, 1, 0), S[:1, :, :] - S[-1:, :, :]), axis=0)

        if D == 1:
            t = (h**2 + v**2) < lambda_ / beta
        else:
            t = np.sum((h**2 + v**2), axis=2) < lambda_ / beta
            t = np.repeat(t[:, :, np.newaxis], D, axis=2)

        h[t] = 0
        v[t] = 0

        Normin2 = np.concatenate((h[:, -1:, :] - h[:, :1, :], -np.diff(h, 1, 1)), axis=1)
        Normin2 = Normin2 + np.concatenate((v[-1:, :, :] - v[:1, :, :], -np.diff(v, 1, 0)), axis=0)

        FS = (Normin1 + beta * fft2(Normin2)) / Denormin
        S = np.real(ifft2(FS))

        beta = beta * kappa

    S = S[:H, :W, :]

    return S
