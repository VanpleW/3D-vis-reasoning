import numpy as np

class SpatialFilter:
    """
    Denoise with 2D Spatial Filters:

    - Median filter
    -  filter
    - Bilateral filter

    """
    def __init__(self, img: np.array, noise_type: str):
        self.img = img
        self.noise_type = noise_type
        self.img_denoised = None


    def median_filter(self, kernal_size: int = None) -> np.array:

        # zero padding the input image
        padded_img = np.pad(self.img, (kernal_size//2, kernal_size//2), 'constant', constant_values=(0, 0))
        # create an empty array to store the filtered image
        filtered_img = np.zeros_like(self.img)

        # iterate over the row
        for i in range(self.img.shape[0]):
            row = padded_img[i:i+kernal_size, :]
            filtered_img[i, :] = np.apply_along_axis(lambda x: np.median(x), axis=1, row)
        # iterate over the column
        for j in range(self.img.shape[1]):
            col = filtered_img[:, j:j+kernal_size]
            filtered_img[:, j] = np.apply_along_axis(lambda x: np.median(x), axis=0, col)

        self.img_denoised = filtered_img
        return self.img_denoised
    

    def gaussian_filter(self, kernal_size):

        self.img_denoised = None
        return self.img_denoised
    
    def bilateral_filter(self, kernal_size):
        self.img_denoised = bilateralFilter(self.img, kernel_size, 75, 75)
        return self.img_denoised
    