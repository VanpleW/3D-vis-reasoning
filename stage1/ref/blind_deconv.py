import numpy as np
from . import utils_chen as utils, blind_deconv_main as bdm
from .cho_code_py.threshold_pxpy_v1 import threshold_pxpy_v1
from .cho_code_py.adjust_psf_center import adjust_psf_center

def blind_deconv(y, lambda_dark, lambda_grad, opts):
    # Do multi-scale blind deconvolution

    # gamma correct
    if opts['gamma_correct'] != 1:
        y = y**opts['gamma_correct']

    b = np.zeros(opts['kernel_size'])

    ret = np.sqrt(0.5)
    maxitr = max(np.floor(np.log(5/min(opts['kernel_size']))/np.log(ret)), 0)
    num_scales = maxitr + 1
    print('Maximum iteration level is %d' % num_scales)

    retv = ret**np.arange(0, maxitr+1)
    k1list = np.ceil(opts['kernel_size']*retv)
    k1list = k1list + (k1list % 2 == 0)
    k2list = np.ceil(opts['kernel_size']*retv)
    k2list = k2list + (k2list % 2 == 0)

    # derivative filters
    dx = np.array([[-1, 1], [0, 0]])
    dy = np.array([[-1, 0], [1, 0]])

    # blind deconvolution - multiscale processing
    for s in range(num_scales, 0, -1):
        if s == num_scales:
            # at coarsest level, initialize kernel
            ks = utils.init_kernel(k1list[s-1])
            k1 = k1list[s-1]
            k2 = k1  # always square kernel assumed
        else:
            # upsample kernel from previous level to next finer level
            k1 = k1list[s-1]
            k2 = k1  # always square kernel assumed

            # resize kernel from previous level
            ks = utils.resizeKer(ks, 1/ret, k1list[s-1], k2list[s-1])

        cret = retv[s-1]
        ys = utils.downSmpImC(y, cret)

        print('Processing scale %d/%d; kernel size %dx%d; image size %dx%d' % 
              (s, num_scales, k1, k2, ys.shape[0], ys.shape[1]))

        # Useless operation
        if s == num_scales:
             _, _, threshold = threshold_pxpy_v1(ys, max(ks.shape))

            # Initialize the parameter: ???
            # if threshold < lambda_grad/10 and threshold != 0:
            #     lambda_grad = threshold
            #     lambda_dark = threshold_image_v1(ys)
            #     lambda_dark = lambda_grad

        ks, lambda_dark, lambda_grad, interim_latent = bdm.blind_deconv_main(ys, ks, lambda_dark, lambda_grad, threshold, opts)

        # center the kernel
        ks = adjust_psf_center(ks)
        ks[ks < 0] = 0
        sumk = np.sum(ks)
        ks = ks / sumk

        # set elements below threshold to 0
        if s == 1:
            kernel = ks
            if opts['k_thresh'] > 0:
                kernel[kernel < np.max(kernel)/opts['k_thresh']] = 0
            else:
                kernel[kernel < 0] = 0
            kernel = kernel / np.sum(kernel)

    return kernel, interim_latent