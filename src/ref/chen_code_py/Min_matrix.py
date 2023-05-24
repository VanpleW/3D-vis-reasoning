import numpy as np
from scipy.sparse import coo_matrix

def min_matrix(I, patch_size):
    M, N, C = I.shape
    J_index = np.zeros((M, N))

    if patch_size % 2 == 0:  # if even number
        raise ValueError('Invalid Patch Size: Only odd number sized patch supported.')

    padsize = patch_size // 2
    h = np.ceil(patch_size / 2).astype(int)

    for m in range(M):
        for n in range(N):
            patch = I[max(0, m-padsize):min(M, m+padsize+1), max(0, n-padsize):min(N, n+padsize+1)]
            h1, h2, _ = patch.shape
            tmp = np.min(patch, axis=2)
            tmp_idx = np.argmin(tmp)

            ori_i = h - (patch_size - h1)
            ori_j = h - (patch_size - h2)

            if ori_i != h and m > h:
                ori_i = h1 + 1 - ori_i
            if ori_j != h and n > h:
                ori_j = h2 + 1 - ori_j

            J_need = np.ceil(tmp_idx / h1).astype(int)
            I_need = tmp_idx - (J_need - 1) * h1

            i_quote = m + I_need - ori_i
            j_quote = n + J_need - ori_j

            J_index[m, n] = (j_quote - 1) * M + i_quote

    sparse_index_row = np.arange(M * N)
    ss = np.ones((M, N)).flatten()
    sparse_index_col = J_index.flatten()

    min_mat = coo_matrix((ss, (sparse_index_row, sparse_index_col)), shape=(M*N, M*N))

    return min_mat
