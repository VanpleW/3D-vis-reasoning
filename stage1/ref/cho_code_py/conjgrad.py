import numpy as np

def conjgrad(x, b, maxIt, tol, Ax_func, func_param, visfunc=None):
    """
    Conjugate gradient optimization

    Written by Sunghyun Cho (sodomau@postech.ac.kr)
    """
    
    r = b - Ax_func(x, func_param)
    p = r
    rsold = np.sum(r * r)

    for iter in range(maxIt):
        Ap = Ax_func(p, func_param)
        alpha = rsold / np.sum(p * Ap)
        x = x + alpha * p
        if visfunc is not None:
            visfunc(x, iter, func_param)
        r = r - alpha * Ap
        rsnew = np.sum(r * r)
        if np.sqrt(rsnew) < tol:
            break
        p = r + (rsnew / rsold) * p
        rsold = rsnew
    
    return x
