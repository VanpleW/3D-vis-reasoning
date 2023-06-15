import numpy as np
from scipy.fft import dst, idst

def solve_min_laplacian(boundary_image):
    H, W = boundary_image.shape

    # Laplacian
    f = np.zeros((H,W))

    # boundary image contains image intensities at boundaries
    boundary_image[1:-1, 1:-1] = 0
    j = np.arange(1, H-1)
    k = np.arange(1, W-1)
    f_bp = np.zeros((H,W))
    f_bp[np.ix_(j,k)] = -4*boundary_image[np.ix_(j,k)] + boundary_image[np.ix_(j,k+1)] + \
        boundary_image[np.ix_(j,k-1)] + boundary_image[np.ix_(j-1,k)] + boundary_image[np.ix_(j+1,k)]

    f1 = f - f_bp

    # DST Sine Transform algo starts here
    f2 = f1[1:-1,1:-1]

    # compute sine transform
    tt = dst(f2, norm='ortho')
    f2sin = dst(tt.T, norm='ortho').T

    # compute Eigen Values
    x, y = np.meshgrid(np.arange(1, W-1), np.arange(1, H-1))
    denom = (2*np.cos(np.pi*x/(W-2))-2) + (2*np.cos(np.pi*y/(H-2)) - 2)

    # divide
    f3 = f2sin/denom

    # compute Inverse Sine Transform
    tt = idst(f3, norm='ortho')
    img_tt = idst(tt.T, norm='ortho').T

    # put solution in inner points; outer points obtained from boundary image
    img_direct = boundary_image
    img_direct[1:-1,1:-1] = 0
    img_direct[1:-1,1:-1] = img_tt

    return img_direct

def wrap_boundary_liu(img, img_size):
    H, W, Ch = img.shape
    H_w = img_size[0] - H
    W_w = img_size[1] - W

    ret = np.zeros((img_size[0], img_size[1], Ch))
    for ch in range(Ch):
        alpha = 1
        HG = img[:,:,ch]

        r_A = np.zeros((alpha*2+H_w, W))
        r_A[:alpha, :] = HG[-alpha:, :]
        r_A[-alpha:, :] = HG[:alpha, :]
        a = (np.arange(H_w))/(H_w-1)
        r_A[alpha:-alpha, 0] = (1-a)*r_A[alpha-1,0] + a*r_A[-alpha,0]
        r_A[alpha:-alpha, -1] = (1-a)*r_A[alpha-1,-1] + a*r_A[-alpha,-1]

        A2 = solve_min_laplacian(r_A[alpha:-alpha+1,:])
        r_A[alpha:-alpha+1,:] = A2
        A = r_A

        r_B = np.zeros((H, alpha*2+W_w))
        r_B[:, :alpha] = HG[:, -alpha:]
        r_B[:, -alpha:] = HG[:, :alpha]
        a = (np.arange(W_w))/(W_w-1)
        r_B[0, alpha:-alpha] = (1-a)*r_B[0,alpha-1] + a*r_B[0,-alpha]
        r_B[-1, alpha:-alpha] = (1-a)*r_B[-1,alpha-1] + a*r_B[-1,-alpha]

        B2 = solve_min_laplacian(r_B[:, alpha:-alpha+1])
        r_B[:,alpha:-alpha+1] = B2
        B = r_B

        r_C = np.zeros((alpha*2+H_w, alpha*2+W_w))
        r_C[:alpha, :] = B[-alpha:, :]
        r_C[-alpha:, :] = B[:alpha, :]
        r_C[:, :alpha] = A[:, -alpha:]
        r_C[:, -alpha:] = A[:, :alpha]

        C2 = solve_min_laplacian(r_C[alpha:-alpha+1, alpha:-alpha+1])
        r_C[alpha:-alpha+1, alpha:-alpha+1] = C2
        C = r_C

        A = A[alpha:-alpha-1, :]
        B = B[:, alpha:-1-alpha]
        C = C[alpha+1:-alpha, alpha+1:-alpha]

        ret[:,:,ch] = np.block([[img[:,:,ch], B], [A, C]])

    return ret
