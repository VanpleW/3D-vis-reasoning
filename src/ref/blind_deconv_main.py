import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt

def blind_deconv_main(blur_B, k, lambda_lmg, lambda_grad, threshold, opts):
    # derivative filters
    dx = np.array([[-1, 1], [0, 0]])
    dy = np.array([[-1, 0], [1, 0]])

    H, W = blur_B.shape
    blur_B_w = wrap_boundary_liu(blur_B, opt_fft_size([H, W] + np.array(k.shape) - 1))
    blur_B_tmp = blur_B_w[:H, :W]
    Bx = convolve2d(blur_B_tmp, dx, 'valid')
    By = convolve2d(blur_B_tmp, dy, 'valid')

    for iter in range(opts['xk_iter']):
        if lambda_lmg == 0:
            # L0 deblurring
            S = L0Restoration(blur_B, k, lambda_grad, 2.0)
        else:
            S = L0_LMG_deblur(blur_B_w, k, lambda_lmg, lambda_grad, 2.0)
            S = S[:H, :W]

        # Necessary for refining gradient
        latent_x, latent_y, threshold = threshold_pxpy_v1(S, max(k.shape), threshold)

        k = estimate_psf(Bx, By, latent_x, latent_y, 2, k.shape)

        print('pruning isolated noise in kernel...')
        CC = bwconncomp(k, 8)
        for ii in range(CC['NumObjects']):
            currsum = np.sum(k[CC['PixelIdxList'][ii]])
            if currsum < .1:
                k[CC['PixelIdxList'][ii]] = 0

        k[k < 0] = 0
        k = k / np.sum(k)

        # Parameter updating
        if lambda_lmg != 0:
            lambda_lmg = max(lambda_lmg / 1.1, 1e-4)
        else:
            lambda_lmg = 0

        if lambda_grad != 0:
            lambda_grad = max(lambda_grad / 1.1, 1e-4)
        else:
            lambda_grad = 0

        plt.figure(1)
        S[S < 0] = 0
        S[S > 1] = 1
        plt.subplot(1, 3, 1)
        plt.imshow(blur_B, cmap='gray')
        plt.title('Blurred image')
        plt.subplot(1, 3, 2)
        plt.imshow(S, cmap='gray')
        plt.title('Interim latent image')
        plt.subplot(1, 3, 3)
        plt.imshow(k, cmap='gray')
        plt.title('Estimated kernel')
        plt.show()

    k[k < 0] = 0
    k = k / np.sum(k)

    return k, lambda_lmg, lambda_grad, S
