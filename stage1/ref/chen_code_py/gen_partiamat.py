import numpy as np
from scipy.sparse import coo_matrix

def gen_partialmat(im_row, im_col):
    rowpx_seq = []
    colpx_seq = []
    valuepx_seq = []
    rowpy_seq = []
    colpy_seq = []
    valuepy_seq = []

    for j in range(im_col):
        for i in range(im_row):
            ind = i + j * im_row

            if i == 0:
                rowpy_seq.extend([ind, ind])
                colpy_seq.extend([ind, ind + 1])
                valuepy_seq.extend([-1, 1])
            else:
                rowpy_seq.extend([ind, ind])
                colpy_seq.extend([ind - 1, ind])
                valuepy_seq.extend([-1, 1])

            if j == 0:
                rowpx_seq.extend([ind, ind])
                colpx_seq.extend([ind, ind + im_row])
                valuepx_seq.extend([-1, 1])
            else:
                rowpx_seq.extend([ind, ind])
                colpx_seq.extend([ind, ind - im_row])
                valuepx_seq.extend([1, -1])

    px_mat = coo_matrix((valuepx_seq, (rowpx_seq, colpx_seq)))
    py_mat = coo_matrix((valuepy_seq, (rowpy_seq, colpy_seq)))

    return px_mat, py_mat

