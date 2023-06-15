import numpy as np
from scipy import ndimage

def adjust_psf_center(psf):
    Y, X = np.meshgrid(range(1, psf.shape[1]+1), range(1, psf.shape[0]+1))
    xc1 = sum2(psf * X)
    yc1 = sum2(psf * Y)
    xc2 = (psf.shape[1]+1) / 2
    yc2 = (psf.shape[0]+1) / 2
    xshift = round(xc2 - xc1)
    yshift = round(yc2 - yc1)
    psf = warpimage(psf, np.array([[1, 0, -xshift], [0, 1, -yshift]]))
    return psf

def sum2(arr):
    return np.sum(arr)

def warpimage(img, M):
    if img.shape[2] == 3:
        warped = np.zeros_like(img)
        for i in range(3):
            warped[:,:,i] = warpProjective2(img[:,:,i], M)
        warped[np.isnan(warped)] = 0
    else:
        warped = warpProjective2(img, M)
        warped[np.isnan(warped)] = 0
    return warped

def warpProjective2(im, A):
    if A.shape[0] > 2:
        A = A[:2, :]
    
    x, y = np.meshgrid(range(1, im.shape[1]+1), range(1, im.shape[0]+1))
    coords = np.array([x.flatten(), y.flatten(), np.ones(np.prod(im.shape))])
    warpedCoords = np.dot(A, coords)
    xprime = warpedCoords[0, :] 
    yprime = warpedCoords[1, :] 
    
    result = ndimage.map_coordinates(im, [yprime, xprime], order=1, mode='nearest').reshape(im.shape)
    return result
