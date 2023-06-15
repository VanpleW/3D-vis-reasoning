import numpy as np
from skimage.restoration import denoise_bilateral
from .cho_code_py.wrap_boundary_liu import wrap_boundary_liu
from .deblurring_adm_aniso import deblurring_adm_aniso
from .L0Restoration import L0Restoration


def bilateral_filter(diff, sigma_color, sigma_space):
    return denoise_bilateral(diff, sigma_color=sigma_color, sigma_space=sigma_space, multichannel=True)

def ringing_artifacts_removal(y, kernel, lambda_tv, lambda_l0, weight_ring):
    H, W, _ = y.shape
    y_pad = wrap_boundary_liu(y, [H, W] + np.array(kernel.shape) - 1)
    Latent_tv = []
    for c in range(y.shape[2]):
        Latent_tv.append(deblurring_adm_aniso(y_pad[:,:,c], kernel, lambda_tv, 1))
    Latent_tv = np.stack(Latent_tv, axis=2)[:H, :W, :]
    
    if weight_ring == 0:
        result = Latent_tv
        return result

    Latent_l0 = L0Restoration(y_pad, kernel, lambda_l0, 2)
    Latent_l0 = Latent_l0[:H, :W, :]

    diff = Latent_tv - Latent_l0
    bf_diff = bilateral_filter(diff, 3, 0.1)
    result = Latent_tv - weight_ring * bf_diff

    return result
