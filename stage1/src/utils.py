import numpy as np
import os
import cv2
from natsort import natsorted
from glob import glob
from src.denoise import DenoiseSolver

# load dummy images from a folder
def load_dummy(noise_type: str):
    dummy_path = os.getcwd() + '/stage1_data/' + noise_type
    #print(dummy_path)
    if os.path.exists(dummy_path):
        noise_files = natsorted(glob(dummy_path + '/*.png'))
        clean_files = natsorted(glob(os.getcwd() + '/stage1_data/input_imgs/*'))
        return len(noise_files), zip(noise_files, clean_files)
    else: 
        print('Error input noise type!')
        return None, None
    
# load images from uploaded test_folder
def load_images(path: str):
    if os.path.exists(path):
        files = natsorted(glob(path + '/*.png'))
        return zip(files)
    else:
        print('Error input path!')
        return None

# select denoise method
def solver(method: str):
    solverDict = {
        'median': DenoiseSolver.median_filter,
        'bilateral': DenoiseSolver.bilateral_filter,
        'wiener': DenoiseSolver.wiener_filter,
        'wavelet': DenoiseSolver.wavelet_transform,
        'LMG': DenoiseSolver.lmg_transforms
        }
    return solverDict[method]

# grid search for best parameters
def grid_search(method: str):
    return None