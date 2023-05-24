import numpy as np
import os
import cv2
from natsort import natsorted
from glob import glob

from . import denoise

# load dummy images from a folder
def load_dummy(noise_type: str):
    dummy_path = os.getcwd() + '/stage1_data/' + noise_type
    print(dummy_path)
    if os.path.exists(dummy_path):
        noise_files = natsorted(glob(dummy_path + '/*.png'))
        clean_files = natsorted(glob(os.getcwd() + '/stage1_data/input_imgs/*'))
        return len(noise_files), zip(noise_files, clean_files)
    else: 
        print('Error input noise type!')
        return None, None
    
# load images from uploaded test_folder
def load_images(path: str) -> list:
    return None

# select denoise method
def solver(method: str):
    solverDict = {
        'median': denoise.DenoiseSolver.median_filter,
        'bilateral': denoise.DenoiseSolver.bilateral_filter,
        'wiener': denoise.DenoiseSolver.wiener_filter,
        'wavelet': denoise.DenoiseSolver.wavelet_transform
        }
    return solverDict[method]

# grid search for best parameters
def grid_search(method: str):
    return None