import numpy as np
import cv2
from skimage import filters
from skimage.morphology import disk
import pywt


class DenoiseSolver():
    """
    Denoise with 2D Spatial or Frequential Filters:

    - Median filter
    - Bilateral filter
    - wiener filter
    - wavelet transform

    input: img, kernal_size, K, sigma
    output: img_denoised
    """
    def median_filter(img, kernal_size: int = 3) -> np.array:
        img_denoised = np.zeros(img.shape)
        for i in range(3):
            dummy = np.copy(img[:,:,i])
            img_denoised[:,:,i] = filters.median(dummy, disk(kernal_size))
        return img_denoised
    
    
    def bilateral_filter(img, kernal_size: int = 3) -> np.array:
        if img.dtype != np.float32:
            img = img.astype(np.float32)  # Convert to floating point and scale to [0, 1]
        img_denoised = cv2.bilateralFilter(img, kernal_size, 75, 75)
        return img_denoised
    
    
    def wiener_filter(img, kernal_size: int = 3, K: int = None) -> np.array:
        """Apply a Wiener filter to a 2D image."""
        img_denoised = np.zeros(img.shape)
        for i in range(3): # loop over each channel
            dummy = np.copy(img[:,:,i])
            dummy = np.fft.fft2(dummy)
            kernel = np.fft.fft2(kernel, s = dummy.shape)
            kernel = np.conj(kernal_size) / (np.abs(kernel) ** 2 + K)
            dummy = dummy * kernel
            dummy = np.abs(np.fft.ifft2(dummy))
            img_denoised[:,:,i] = dummy
        return img_denoised
    

    def wavelet_transform(img, sigma: int = 0.1 ) -> np.array:
        """Apply a wavelet transform to a 2D image."""
        coeffs = pywt.wavedec2(img, 'db8', level=4)
        # Threshold the wavelet coefficients
        threshold = sigma * np.sqrt(2 * np.log(img.size))
        for i in range(1, len(coeffs)):
            coeffs[i] = pywt.threshold(coeffs[i], threshold)
        # Perform an inverse wavelet transform
        img_denoised = pywt.waverec2(coeffs, 'db8')
        return img_denoised