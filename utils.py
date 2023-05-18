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

