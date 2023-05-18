import numpy as np
import os
import cv2
from natsort import natsorted
from glob import glob

from stage1.denoise import SpatialFilter, FrequencyFilter

# load dummy images from a folder
def load_dummy(noise_type: str) -> list:
    path = 'stage1/' + noise_type + '/*'
    noise_files = natsorted(glob(path))
    clean_files = natsorted(glob('stage1/input_imgs/*'))
    return zip(noise_files, clean_files)

# load images from uploaded test_folder
def load_images(path: str) -> list:
    return None

# select denoise method
def solver(method: str):
    solverDict = {
        'median': SpatialFilter.median_filter,
        'bilateral': SpatialFilter.bilateral_filter,
        'wiener': SpatialFilter.wiener_filter,
        'wavelet': FrequencyFilter.wavelet_filter
        }
    return solverDict[method]



# calculate the power spectrum density of an image
def psd(img: np.array) -> np.array:

    # calculate the 2D FFT of the input image
    img_fft = np.fft.fft2(img)
    # shift the zero-frequency component to the center of the spectrum
    img_fft_shift = np.fft.fftshift(img_fft)
    # calculate the power spectrum density
    img_psd = np.abs(img_fft_shift) ** 2
    return img_psd


# calculate the power spectrum density of noise inside a image
def psd_noise(img: np.array, noise_type: str) -> np.array:

    # calculate the 2D FFT of the input image
    img_fft = np.fft.fft2(img)
    # shift the zero-frequency component to the center of the spectrum
    img_fft_shift = np.fft.fftshift(img_fft)
    # calculate the power spectrum density
    img_psd = np.abs(img_fft_shift) ** 2
    return img_psd

