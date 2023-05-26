from . import gen_partiamat, Max_matrix, Min_matrix, Abs_matrix
import numpy as np

def LMG(img, patch_size):
    M, N = img.shape
    px_mat, py_mat = gen_partiamat.gen_partialmat(M, N)
    px = np.reshape(px_mat.dot(img.flatten()), (M, N))
    py = np.reshape(py_mat.dot(img.flatten()), (M, N))
    abs_x_mat = Abs_matrix.abs_matrix(px)
    abs_y_mat = Abs_matrix.abs_matrix(py)

    max_tv_mat = Max_matrix.max_matrix(np.abs(px) + np.abs(py), patch_size)
    output_img = max_tv_mat.dot((abs_x_mat.dot(px_mat) + abs_y_mat.dot(py_mat)).dot(img.flatten()))
    output_img = np.reshape(output_img, (M, N))
    A = max_tv_mat.dot(abs_x_mat.dot(px_mat) + abs_y_mat.dot(py_mat))

    return output_img, A
