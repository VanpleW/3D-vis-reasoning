import numpy as np
from scipy.signal import convolve2d
from scipy import stats

def threshold_pxpy_v1(latent, psf_size, threshold=None):

    # mask indicates region with gradients. outside of mask shud be smooth...

    if threshold is None:
        threshold = 0
        b_estimate_threshold = True
    else:
        b_estimate_threshold = False

    denoised = latent

    # derivative filters
    dx = np.array([[-1, 1], [0, 0]])
    dy = np.array([[-1, 0], [1, 0]])

    px = convolve2d(denoised, dx, mode='valid')
    py = convolve2d(denoised, dy, mode='valid')
    pm = px**2 + py**2

    # if this is the first prediction, then we need to find an appropriate
    # threshold value by building a histogram of gradient magnitudes
    if b_estimate_threshold:
        pd = np.arctan(py/px)
        pm_steps = np.arange(0, 2, 0.00006)
        H1 = np.flip(np.cumsum(np.histogram(pm[np.logical_and(pd >= 0, pd < np.pi/4)], bins=pm_steps)[0]))
        H2 = np.flip(np.cumsum(np.histogram(pm[np.logical_and(pd >= np.pi/4, pd < np.pi/2)], bins=pm_steps)[0]))
        H3 = np.flip(np.cumsum(np.histogram(pm[np.logical_and(pd >= -np.pi/4, pd < 0)], bins=pm_steps)[0]))
        H4 = np.flip(np.cumsum(np.histogram(pm[np.logical_and(pd >= -np.pi/2, pd < -np.pi/4)], bins=pm_steps)[0]))

        th = max([max(psf_size)*20, 10])

        for t in range(len(pm_steps)):
            min_h = min([H1[t], H2[t], H3[t], H4[t]])
            if min_h >= th:
                threshold = pm_steps[-t-1]
                break

    # thresholding
    m = pm < threshold
    while np.all(m):
        threshold = threshold * 0.81
        m = pm < threshold

    px[m] = 0
    py[m] = 0

    # update prediction parameters
    if b_estimate_threshold:
        threshold = threshold
    else:
        threshold = threshold / 1.1

    return px, py, threshold
