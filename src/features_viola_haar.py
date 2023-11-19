import numpy as np
from numba import njit, prange


@njit(cache=True)
def generate_haar_features(grid_size):
    # generate all possible haar features
    TOTAL_FTS = 162336
    FT_DIM = 5

    output = np.zeros((TOTAL_FTS, FT_DIM), dtype=np.int32)

    count = 0
    # generate feature type 2h

    for w in range(2, grid_size + 1, 2):
        for h in range(1, grid_size + 1):
            for x in range(0, grid_size - w + 1):
                for y in range(0, grid_size - h + 1):
                    output[count, :] = [1, x, y, w, h]
                    count += 1

    # generate feature type 2v
    for w in range(1, grid_size + 1):
        for h in range(2, grid_size + 1, 2):
            for x in range(0, grid_size - w + 1):
                for y in range(0, grid_size - h + 1):
                    output[count, :] = [2, x, y, w, h]
                    count += 1

    # generate feature type 3h
    for w in range(3, grid_size + 1, 3):
        for h in range(1, grid_size + 1):
            for x in range(0, grid_size - w + 1):
                for y in range(0, grid_size - h + 1):
                    output[count, :] = [3, x, y, w, h]
                    count += 1

    # generate feature type 3v
    for w in range(1, grid_size + 1):
        for h in range(3, grid_size + 1, 3):
            for x in range(0, grid_size - w + 1):
                for y in range(0, grid_size - h + 1):
                    output[count, :] = [4, x, y, w, h]
                    count += 1

    # generate feature type 4
    for w in range(2, grid_size + 1, 2):
        for h in range(2, grid_size + 1, 2):
            for x in range(0, grid_size - w + 1):
                for y in range(0, grid_size - h + 1):
                    output[count, :] = [5, x, y, w, h]
                    count += 1

    return output


@njit(parallel=True, cache=True)
def compute_features_all_imgs(fts: np.ndarray, int_imgs: np.ndarray,
                              nf: np.ndarray):
    # get number of features
    num_fts = fts.shape[0]

    # create output array
    output = np.zeros((num_fts, int_imgs.shape[0]), dtype=np.float32)

    # loop over all integral images
    for n in prange(int_imgs.shape[0]):
        # loop over all features
        for i in prange(num_fts):
            ft = fts[i, :]
            x, y, w, h = ft[1], ft[2], ft[3], ft[4]
            if ft[0] == 1:
                # feature type 2h
                hw = w // 2
                output[i, n] = 2 * (int_imgs[n, y + h, x + hw]
                                - int_imgs[n, y, x + hw]) + \
                        int_imgs[n, y, x] - int_imgs[n, y + h, x] + \
                        int_imgs[n, y, x + w] - int_imgs[n, y + h, x + w]

            elif ft[0] == 2:
                # feature type 2v
                hh = h // 2
                output[i, n] = 2 * (int_imgs[n, y + hh, x]
                                - int_imgs[n, y + hh, x + w]) + \
                        int_imgs[n, y, x+w] - int_imgs[n, y, x] + \
                        int_imgs[n, y + h, x + w] - int_imgs[n, y + h, x]

            elif ft[0] == 3:
                # feature type 3h
                tw = w // 3
                output[i, n] = 2 * (int_imgs[n, y + h, x + 2 * tw]) + \
                        2 * int_imgs[n, y, x + tw] - 2 * int_imgs[n, y + h, x + tw] - \
                        2 * int_imgs[n, y, x + 2 * tw] + \
                        int_imgs[n, y + h, x] - int_imgs[n, y, x] + \
                        int_imgs[n, y, x + w] - int_imgs[n, y + h, x + w]

            elif ft[0] == 4:
                # feature type 3v
                th = h // 3
                output[i, n] = 2 * (int_imgs[n, y + 2 * th, x + w]) + \
                        2 * int_imgs[n, y + th, x] - 2 * int_imgs[n, y + th, x + w] - \
                        2 * int_imgs[n, y + 2 * th, x] + \
                        int_imgs[n, y, x + w] - int_imgs[n, y, x] + \
                        int_imgs[n, y + h, x] - int_imgs[n, y + h, x + w]

            elif ft[0] == 5:
                # feature type 4
                hw = w // 2
                hh = h // 2
                output[i, n] = 4 * int_imgs[n, y + hh, x + hw] - \
                        2 * int_imgs[n, y, x + hw] - 2 * int_imgs[n, y + hh, x] - \
                        2 * int_imgs[n, y + hh, x + w] - 2 * int_imgs[n, y + h, x + hw] + \
                        int_imgs[n, y, x] + int_imgs[n, y + h, x] + \
                        int_imgs[n, y, x + w] + int_imgs[n, y + h, x + w]

            output[i, n] = 0 if nf[n] == 0 else output[i, n] / nf[n]

    return output