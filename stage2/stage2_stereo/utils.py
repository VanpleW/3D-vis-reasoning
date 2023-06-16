from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import cv2
import numpy as np
import largestinteriorrectangle as lir


def search_params(left_img, right_img, dis_gt_img, bs, md):
    stereo = cv2.StereoBM_create(numDisparities=md, blockSize=bs)
    disparity = stereo.compute(left_img, right_img)
    disparity = cv2.normalize(disparity, None, alpha=0, beta=md, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    #mask = np.logical_and(disparity = 0, dis_gt_img = 0)
    mask = np.logical_or(disparity == 0, dis_gt_img == 0)
    masked_sq_err = np.ma.array(np.square(disparity - dis_gt_img), mask=mask)
    rmse = np.sqrt(masked_sq_err.mean())
    return rmse

def semiglob_search_params(left_img, right_img, dis_gt_img, bs, md):
    stereo = cv2.StereoSGBM_create(numDisparities=md, blockSize=bs)
    disparity = stereo.compute(left_img, right_img)
    disparity = cv2.normalize(disparity, None, alpha=0, beta=md, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    mask = np.logical_or(disparity == 0, dis_gt_img == 0)
    masked_sq_err = np.ma.array(np.square(disparity - dis_gt_img), mask=mask)
    rmse = np.sqrt(masked_sq_err.mean())
    return rmse


def psnr_ssim_rect(orig_img, rectified_img):
    # Convert the image to grayscale
    gray = cv2.cvtColor(rectified_img, cv2.COLOR_BGR2GRAY)
    # Threshold the image to create a binary image (black and white)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    grid = thresh.astype(np.bool)

    rectangle = lir.lir(grid)
    
    (x_min, y_min) = lir.pt1(rectangle)
    (x_max, y_max) = lir.pt2(rectangle)

    # Crop the image using the inscribed rectangle's coordinates
    cropped_img = orig_img[y_min:y_max, x_min:x_max]
    cropped_img_rect = rectified_img[y_min:y_max, x_min:x_max]
    
    psnr_out = psnr(cropped_img, cropped_img_rect)
    ssim_out = ssim(cropped_img, cropped_img_rect, multichannel=True)

    return (psnr_out, ssim_out)