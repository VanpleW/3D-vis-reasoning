import numpy as np
import cv2
from skimage import filters
from skimage.morphology import disk
import pywt
from ..ref.blind_deconv import blind_deconv

class DenoiseSolver():
    """
    Denoise with 2D Spatial or Frequential Filters:

    - Median filter
    - Bilateral filter
    - wiener filter
    - wavelet transform
    - lmg transform

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
    
    
    def wiener_filter(img, kernal_size: int = 3, K: float = 0.01) -> np.array:
        """Apply a Wiener filter to a 2D image."""
        img_denoised = np.zeros(img.shape)
        kernel = np.ones((kernal_size, kernal_size)) / (kernal_size ** 2)
        for i in range(3): # loop over each channel
            dummy = np.copy(img[:,:,i])
            dummy = np.fft.fft2(dummy)
            kernel_f = np.fft.fft2(kernel, s = dummy.shape)
            kernel_f = np.conj(kernel_f) / (np.abs(kernel_f) ** 2 + K)
            dummy = dummy * kernel_f
            dummy = np.abs(np.fft.ifft2(dummy))
            img_denoised[:,:,i] = dummy
        return img_denoised
    

    def wavelet_transform(img, sigma: float = 0.1 ) -> np.array:
        """Apply a wavelet transform to a 2D image."""
        coeffs = pywt.wavedec2(img, 'db8', level=4)
        # Threshold the wavelet coefficients
        threshold = sigma * np.sqrt(2 * np.log(img.size))
        for i in range(1, len(coeffs)):
            coeffs[i] = pywt.threshold(coeffs[i], threshold)
        # Perform an inverse wavelet transform
        img_denoised = pywt.waverec2(coeffs, 'db8')
        return img_denoised
    

    def lmg_transform(img, patch_size: int = 16) -> np.array:
        """Apply a LMG transform to a 2D image."""
        opts = {}
        opts['prescale'] = 1 #downsampling
        opts.xk_iter = 5  #the iterations
        opts.gamma_correct = 1.0
        opts.k_thresh = 20
        lambda_lmg =4e-3
        lambda_grad =4e-3
        opts.gamma_correct = 1
        lambda_tv = 0.001
        lambda_l0 = 5e-4
        weight_ring = 1
        _, img_denoised = blind_deconv(img, lambda_lmg, lambda_grad, opts)
        return img_denoised