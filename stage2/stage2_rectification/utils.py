import largestinteriorrectangle as lir
import cv2
import numpy as np
import os
from natsort import natsorted
from glob import glob
from PIL import Image
from tqdm import tqdm

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

"""
    function psnr_ssim_rect:

    - this function takes in the original image and the rectified image and returns the PSNR and SSIM values
"""
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
    print(cropped_img.shape, cropped_img_rect.shape)
    
    psnr_out = psnr(cropped_img, cropped_img_rect)
    ssim_out = ssim(cropped_img, cropped_img_rect, multichannel=True, data_range=cropped_img_rect.max() - cropped_img_rect.min(), channel_axis=2)

    return (psnr_out, ssim_out)


"""
    function unwrap:

    - this function take the image and the rectified image and unwraps the rectified image
    
"""
def unwrap(orig_img: np.ndarray, wraped_img: np.ndarray, ft_detector, ft_matcher, ft_ratio: np.float_, MIN_MATCH_COUNT=10) -> np.ndarray:

    # Convert the image to grayscale
    orig_img_gray = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
    wraped_img_gray = cv2.cvtColor(wraped_img, cv2.COLOR_BGR2GRAY)

    # get the keypoints and corresponding descriptors
    kp1, des1 = ft_detector.detectAndCompute(orig_img_gray, None)
    kp2, des2 = ft_detector.detectAndCompute(wraped_img_gray, None)
    
    # match.distance is a float between {0:100} - lower means more similar
    matches = ft_matcher.knnMatch(des1, des2, k=4)

    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < ft_ratio * n.distance:
            good.append(m)

    # compute homography with RANSAC
    if len(good) > MIN_MATCH_COUNT:
        # get the good key points positions
        source_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        destination_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

        # compute homography matrix and get inliers mask
        H, mask = cv2.findHomography(source_pts, destination_pts, cv2.RANSAC, 5.0)
        # get inliers keypoints
        pts1 = source_pts[mask.ravel()==1]
        pts2 = destination_pts[mask.ravel()==1]

        h, w = orig_img_gray.shape
        # get the corners from the image_1 ( the object to be "detected" )
        corners = np.float32([ [0,0], [0,h-1], [w-1,h-1], [w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(corners, H)
        # draw the detected rectangle
        wraped = cv2.polylines(wraped_img, [np.int32(dst)], True, (255, 0, 0), 3, cv2.LINE_AA)

    else:
        print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
        return None

    # compute the unwarping
    unwrapped_img = cv2.warpPerspective(orig_img, H, (wraped.shape[1], wraped.shape[0]))

    return unwrapped_img



""" 
    func load_images:
    - load original and wraped images from a folder 
"""
def load_images(path):
    orig_path = path + '/input_imgs'
    wraped_path = path + '/warped_imgs'
    if os.path.exists(orig_path) and os.path.exists(wraped_path):
        original = natsorted(glob(orig_path + '/*.png'))
        wraped = natsorted(glob(wraped_path + '/*.png'))
        return len(original), zip(original, wraped)
    else: 
        print('The images are gone! Check your folder!')
        return None, None
    


"""
    func cal_psnr_ssim:
    - calculate the psnr and ssim for all the images in the folder
"""
def cal_psnr_ssim(files, f_detector, f_matcher, f_ratio, num):
    
    psnr_sum = 0
    ssim_sum = 0

    for orig_img, wrap_img in tqdm(files):
        orig_img = (np.array(Image.open(orig_img)) * 255).astype(np.uint8)
        wrap_img = (np.array(Image.open(wrap_img)) * 255).astype(np.uint8)
        # unwrap the image
        unwrapped_img = unwrap(orig_img, wrap_img, f_detector, f_matcher, f_ratio)
        # calculate psnr and ssim
        psnr_val, ssim_val = psnr_ssim_rect(orig_img, unwrapped_img)
        psnr_sum += psnr_val
        ssim_sum += ssim_val

    #print('PSNR: ', psnr_sum/num, 'and SSIM: ', ssim_sum/num)
    return (psnr_sum/num, ssim_sum/num)