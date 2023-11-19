import numpy as np
from numba import njit


@njit(cache=True)
def to_integral_numba(img: np.ndarray) -> np.ndarray:
    int_im = np.zeros((img.shape[0] + 1, img.shape[1] + 1))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            int_im[i + 1, j + 1] = int_im[i, j + 1] + \
                int_im[i + 1, j] - int_im[i, j] + img[i, j]

    return int_im


@njit(cache=True)
def to_integral_numba_uint32(img: np.ndarray) -> np.ndarray:
    int_im = np.zeros((img.shape[0] + 1, img.shape[1] + 1), dtype=np.uint32)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            int_im[i + 1, j + 1] = int_im[i, j + 1] + \
                int_im[i + 1, j] - int_im[i, j] + img[i, j]

    return int_im


@njit(cache=True)
def square(img):
    img_sq = np.zeros_like(img, dtype=np.uint16)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img_sq[i, j] = img[i, j] * img[i, j]

    return img_sq


@njit(cache=True)
def calc_norm(img_int: np.ndarray, img_sq_int: np.ndarray) -> np.ndarray:
    """
    calculate the normalization factor from the integral image of the squared image
    """

    n_rows, n_cols = img_int.shape
    val_sum = img_int[n_rows - 2, n_cols -
                      2] - img_int[n_rows - 2,
                                   1] - img_int[1, n_cols - 2] + img_int[1, 1]
    val_sq_sum = img_sq_int[n_rows - 2, n_cols - 2] - img_sq_int[
        n_rows - 2, 1] - img_sq_int[1, n_cols - 2] + img_sq_int[1, 1]
    area = (n_rows - 3) * (n_cols - 3)

    return np.sqrt(area * val_sq_sum - val_sum * val_sum)