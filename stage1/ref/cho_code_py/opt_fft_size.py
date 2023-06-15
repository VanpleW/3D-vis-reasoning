import numpy as np

# Persistent variable
opt_fft_size_LUT = None
LUT_size = 4096

def opt_fft_size(n):
    """
    Compute an optimal data length for Fourier transforms.
    Written by Sunghyun Cho (sodomau@postech.ac.kr)
    """
    global opt_fft_size_LUT

    if opt_fft_size_LUT is None:
        print('generate opt_fft_size_LUT')
        opt_fft_size_LUT = np.zeros(LUT_size)

        e2 = 1
        while e2 <= LUT_size:
            e3 = e2
            while e3 <= LUT_size:
                e5 = e3
                while e5 <= LUT_size:
                    e7 = e5
                    while e7 <= LUT_size:
                        if e7 <= LUT_size:
                            opt_fft_size_LUT[e7-1] = e7
                        if e7*11 <= LUT_size:
                            opt_fft_size_LUT[e7*11-1] = e7*11
                        if e7*13 <= LUT_size:
                            opt_fft_size_LUT[e7*13-1] = e7*13
                        e7 *= 7
                    e5 *= 5
                e3 *= 3
            e2 *= 2

        nn = 0
        for i in range(LUT_size, 0, -1):
            if opt_fft_size_LUT[i-1] != 0:
                nn = i
            else:
                opt_fft_size_LUT[i-1] = nn

    m = np.zeros(n.shape)
    for c in range(np.prod(n.shape)):
        nn = n.item(c)
        if nn <= LUT_size:
            m.itemset(c, opt_fft_size_LUT[nn-1])
        else:
            m.itemset(c, -1)
    return m
