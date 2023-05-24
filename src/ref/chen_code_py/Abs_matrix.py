import numpy as np
from scipy.sparse import csr_matrix

def abs_matrix(I):
    M, N, C = I.shape
    abs_I = np.zeros_like(I)
    if C == 1:
        a = np.abs(I)
        abs_I = np.divide(a, I, out=np.ones_like(a), where=I!=0)
    else:
        for cc in range(C):
            a = np.abs(I[:,:,cc])
            abs_I[:,:,cc] = np.divide(a, I[:,:,cc], out=np.ones_like(a), where=I[:,:,cc]!=0)
    abs_I[np.isnan(abs_I)] = 1

    sparse_index_row = np.arange(M*N)
    sparse_index_col = np.arange(M*N)

    Abs_mat = csr_matrix((abs_I.flatten(), (sparse_index_row, sparse_index_col)), shape=(M*N, M*N))

    return Abs_mat
