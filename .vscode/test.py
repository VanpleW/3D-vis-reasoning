'''
    test different stages of the project
'''
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import pandas as pd

from utils import load_dummy, load_images, solver, psd, psd_noise


def test_dummy_stage1(noise_type: str, method:str) -> list:
    """ 
        dummy test for stage 1 noise images with data provided by the course,
        input the noise type and return a list of psnr and ssim
    """
    stats = pd.DataFrame(columns=['noise_type', 'method', 'psnr', 'ssim'])
    # choose denoise method
    sol = solver(method)

    for img, noise_img in load_dummy(noise_type):
        img = np.array(Image.open(img))
        noise_img = np.array(Image.open(noise_img))
        # denoise with the solver
        dn_img = sol(noise_img, noise_type)
        psnr_val = psnr(img, noise_img)
        ssim_val = ssim(img, noise_img, multichannel=True)
        # load PSNR and SSIM into a list according to the noise type and method
        stats.loc[len(stats)] = [noise_type, method, psnr_val, ssim_val]
