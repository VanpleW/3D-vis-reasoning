import numpy as np
from scipy import fftpack
from skimage.filters import threshold_otsu
from pypher import psf2otf
from .chen_code_py.LMG import LMG

def psf2otf(psf, shape):
    psf = np.pad(psf, ((0, shape[0] - psf.shape[0]), (0, shape[1] - psf.shape[1])), 'constant')
    for i in range(len(psf.shape)):
        psf = np.roll(psf, -int(psf.shape[i] / 2), i)
    return fftpack.fftn(psf, shape)

def L0_LMG_deblur(Im, kernel, lambda_, wei_grad, kappa=2.0):
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
        Denormin2 = np.tile(Denormin2, [1, 1, D])
        KER = np.tile(KER, [1, 1, D])
        Den_KER = np.tile(Den_KER, [1, 1, D])
    Normin1 = np.conj(KER) * fftpack.fftn(S)
    patch_size = 35
    mybeta_pixel = lambda_ / threshold_otsu((S**2).flatten())

    for iter in range(4):
        J, A = LMG(S, patch_size)
        t = 2 - J
        t2 = lambda_ / (2 * mybeta_pixel)
        t3 = np.abs(t) - t2
        t3[t3 < 0] = 0
        u = np.sign(t) * t3
        alpha3 = mybeta_pixel * 2
        for i in range(4):
            M, N = A.shape
            sparse_eye = np.eye(M, N)
            subsitute_I = np.linalg.solve((mybeta_pixel * (A.T @ A) + alpha3 * sparse_eye), (mybeta_pixel * A.T @ (2 - u.flatten()) + alpha3 * S.flatten()))
            subsitute_I = subsitute_I.reshape(u.shape)
            beta = 2 * wei_grad
            while beta < betamax:
                h = np.c_[np.diff(S, axis=1), S[:, 0, :] - S[:, -1, :]]
                v = np.r_[np.diff(S, axis=0), S[0, :, :] - S[-1, :, :]]
                th = h**2 < wei_grad / beta
                tv = v**2 < wei_grad / beta
                h[th] = 0
                v[tv] = 0
                Normin2 = np.c_[h[:, -1, :] - h[:, 0, :], -np.diff(h, axis=1)]
                Normin2 = Normin2 + np.r_[v[-1, :, :] - v[0, :, :], -np.diff(v, axis=0)]
                FS = (Normin1 + beta * fftpack.fftn(Normin2) + alpha3 * fftpack.fftn(subsitute_I)) / (Den_KER + beta * Denormin2 + alpha3)
                S = np.real(fftpack.ifftn(FS))
                beta = beta * kappa
            alpha3 = alpha3 * 4
        mybeta_pixel = mybeta_pixel * 4

    return S
