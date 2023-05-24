'''
    test different stages of the project
'''
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from PIL import Image
from tqdm import tqdm

from . import utils


def test_dummy_stage1(noise_type: str, method:str) -> list:
    """ 
        dummy test for stage 1 noise images with data provided by the course

        input:
        - noise_type:   str, 
                        'gaussian_blur',    
                        'sp_noise', 
                        'gaussian_blur', 
                        'speckle_noise',
                        'motion_blur'

        - method:       str, 
                        'median', 
                        'bilateral', 
                        'wiener', 
                        'wavelet'
        
        return: a list of psnr and ssim values
    """
    psnr_val = 0
    ssim_val = 0
    # choose denoise method
    sol = utils.solver(method)

    length, files = utils.load_dummy(noise_type)
    print('length of files: ', length)
    for noise_img, img in tqdm(files):
        img = np.array(Image.open(img))/255.0
        noise_img = np.array(Image.open(noise_img))/255.0
        # denoise with the solver
        dn_img = sol(noise_img)
        # calculate psnr and ssim
        psnr_val += psnr(img, dn_img)
        ssim_val += ssim(img, dn_img, multichannel=True, channel_axis=2, data_range=dn_img.max() - dn_img.min())

    # load PSNR and SSIM into a list according to the noise type and method
    return [noise_type, method, psnr_val/length, ssim_val/length]
