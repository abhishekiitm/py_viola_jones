import os
import time
import logging
import concurrent.futures
import multiprocessing
import tracemalloc
import gc
from functools import partial
from typing import NamedTuple

import numpy as np
import cv2
import pandas as pd
from numba import njit

import utils
import cascade_classifier as cclf

logger = logging.getLogger(__name__)


@njit(cache=True)
def _scan_image(wc_classifiers, wc_thresholds, wc_polarities, wc_alphas,
                  thresholds_stage, start_idx_stage, end_idx_stage, gray_img,
                  int_im, int_im_sq, window_start, H, W, default_grid_size,
                  fpr_estimate, using_cascade_clf):
    STRIDE_FACTOR = 3
    STRIDE = window_start // STRIDE_FACTOR
    max_size = max(
        int(
            np.ceil(H * W / STRIDE / STRIDE *
                    (fpr_estimate**len(start_idx_stage)))), 10)
    fp_imgs = np.zeros((max_size, window_start, window_start),
                       dtype=np.int32)  # store the false positive images
    windows = np.zeros((max_size, 4), dtype=np.int32)  # store the windows
    count = 0
    consumed = 0
    skip_x = False

    # create a list of 1000's for storing the feature values for all the weak classifiers in all the stages
    feat_vals = np.zeros(
        (max_size, len(wc_classifiers)), dtype=np.float32) + 1000.
    out_hypotheses = np.zeros((max_size, len(start_idx_stage)),
                              dtype=np.float32)
    out_nf = np.zeros((max_size, len(start_idx_stage)), dtype=np.float32)

    x, y = 0, 0
    while x < W - window_start + 1:
        while y < H - window_start + 1:

            # calculate mean and std dev of the window
            sum_im = int_im[y + window_start - 1, x + window_start - 1] + int_im[y+1, x+1] \
                - int_im[y + window_start-1, x+1] - int_im[y+1, x + window_start-1]
            area = (window_start - 2) * (window_start - 2)

            sum_im_sq = int_im_sq[y + window_start-1, x + window_start-1] + int_im_sq[y+1, x+1] \
                - int_im_sq[y + window_start-1, x+1] - int_im_sq[y+1, x + window_start-1]

            nf = np.sqrt(area * sum_im_sq - sum_im * sum_im)

            if not using_cascade_clf:
                # generating for first stage when there is no cascade classifier
                # std dev of the window is high enough, so add it to the list of candidate windows
                fp_imgs[count, :] = gray_img[y:y + window_start,
                                             x:x + window_start]
                windows[count, :] = np.array(
                    [x, y, x + window_start, y + window_start])
                count += 1
                consumed += 1
                y += STRIDE
                skip_x = True
                continue

            # std dev of the window is high enough and cascade classifier is present
            # predict using cascade classifier

            is_false_positive = True
            # loop through all the stages of the cascade classifier
            for i in range(len(start_idx_stage)):
                # select the indices of the start and end of the weak classifiers for the current stage
                start_idx = start_idx_stage[i]
                end_idx = end_idx_stage[i]

                sum_hypotheses = 0.
                # loop through all the weak classifiers of the current stage and sum the hypotheses and alphas
                for j in range(start_idx, end_idx + 1):
                    # select the weak classifier
                    feat = wc_classifiers[j]
                    ft_x, ft_y, ft_w, ft_h = feat[1], feat[2], feat[3], feat[4]

                    scale_x = ft_x
                    scale_y = ft_y
                    scale_w = ft_w
                    scale_h = ft_h

                    # calculate the feature value
                    if feat[0] == 1:
                        # feature type 2h
                        hw = scale_w // 2
                        feat_val = 2 * (int_im[y + scale_y + scale_h, x + scale_x + hw]
                                        - int_im[y + scale_y, x + scale_x + hw]) + \
                                int_im[y + scale_y, x + scale_x] - int_im[y + scale_y + scale_h, x + scale_x] + \
                                int_im[y + scale_y, x + scale_x + scale_w] - int_im[y + scale_y + scale_h, x + scale_x + scale_w]

                    elif feat[0] == 2:
                        # feature type 2v
                        hh = scale_h // 2
                        feat_val = 2 * (int_im[y + scale_y + hh, x + scale_x]
                                        - int_im[y + scale_y + hh, x + scale_x + scale_w]) + \
                                int_im[y + scale_y, x + scale_x+scale_w] - int_im[y + scale_y, x + scale_x] + \
                                int_im[y + scale_y + scale_h, x + scale_x + scale_w] - int_im[y + scale_y + scale_h, x + scale_x]

                    elif feat[0] == 3:
                        # feature type 3h
                        tw = scale_w // 3
                        feat_val = 2 * (int_im[y + scale_y + scale_h, x + scale_x + 2 * tw]) + \
                                2 * int_im[y + scale_y, x + scale_x + tw] - 2 * int_im[y + scale_y + scale_h, x + scale_x + tw] - \
                                2 * int_im[y + scale_y, x + scale_x + 2 * tw] + \
                                int_im[y + scale_y + scale_h, x + scale_x] - int_im[y + scale_y, x + scale_x] + \
                                int_im[y + scale_y, x + scale_x + scale_w] - int_im[y + scale_y + scale_h, x + scale_x + scale_w]

                    elif feat[0] == 4:
                        # feature type 3v
                        th = scale_h // 3
                        feat_val = 2 * (int_im[y + scale_y + 2 * th, x + scale_x + scale_w]) + \
                                2 * int_im[y + scale_y + th, x + scale_x] - 2 * int_im[y + scale_y + th, x + scale_x + scale_w] - \
                                2 * int_im[y + scale_y + 2 * th, x + scale_x] + \
                                int_im[y + scale_y, x + scale_x + scale_w] - int_im[y + scale_y, x + scale_x] + \
                                int_im[y + scale_y + scale_h, x + scale_x] - int_im[y + scale_y + scale_h, x + scale_x + scale_w]

                    elif feat[0] == 5:
                        # feature type 4
                        hw = scale_w // 2
                        hh = scale_h // 2
                        feat_val = 4 * int_im[y + scale_y + hh, x + scale_x + hw] - \
                                2 * int_im[y + scale_y, x + scale_x + hw] - 2 * int_im[y + scale_y + hh, x + scale_x] - \
                                2 * int_im[y + scale_y + hh, x + scale_x + scale_w] - 2 * int_im[y + scale_y + scale_h, x + scale_x + hw] + \
                                int_im[y + scale_y, x + scale_x] + int_im[y + scale_y + scale_h, x + scale_x] + \
                                int_im[y + scale_y, x + scale_x + scale_w] + int_im[y + scale_y + scale_h, x + scale_x + scale_w]

                    feat_vals[count, j] = feat_val
                    # normalize the feature value
                    feat_val = 0 if nf == 0 else feat_val / nf

                    vote = (np.sign((wc_polarities[j] * wc_thresholds[j]) -
                                    (wc_polarities[j] * feat_val)) + 1) // 2
                    sum_hypotheses += wc_alphas[j] * vote

                out_hypotheses[count, i] = sum_hypotheses
                if sum_hypotheses < thresholds_stage[i]:
                    # correctly predicted not a face by the cascade classifier, break out of the loop
                    is_false_positive = False
                    break

            consumed += 1
            # if predicted face, add to the list of bounding boxes
            if is_false_positive:
                fp_imgs[count, :] = gray_img[y:y + window_start,
                                             x:x + window_start]
                windows[count, :] = np.array(
                    [x, y, x + window_start, y + window_start])
                out_nf[count, :] = nf
                count += 1

                if count == max_size:
                    return fp_imgs[:count, :], windows[:count, :], feat_vals[:count, :], \
                        out_hypotheses[:count, :], out_nf[:count,:], consumed, max_size

                skip_x = True
                y += int(STRIDE * window_start / default_grid_size)
                continue

            # if not predicted face, move on to the next window
            y += 1

        # if skip_x is True, last row had a candidate window, so move by STRIDE
        x = x + STRIDE * window_start if skip_x else x + 1

        skip_x = False
        y = 0

    # search for candidate windows complete, return the list of candidate windows

    return fp_imgs[:count, :], windows[:count, :], feat_vals[:count, :], \
        out_hypotheses[:count, :], out_nf[:count, :], consumed, max_size


class DataReader:
    """
    Class for reading the data from the positive samples directory and negative samples directory
    """

    def __init__(self, args, clf):
        # check if positive samples directory, and negative samples directory exist
        if not os.path.exists(args.data_pos):
            raise Exception('Positive samples directory does not exist')

        if not os.path.exists(args.data_neg):
            raise Exception('Negative samples directory does not exist')

        self.args = args
        self.clf = clf

        # get the filepaths of the positive samples and negative samples
        self.pos_filepaths = self._get_filepaths(args.data_pos)
        self.neg_filepaths = self._get_filepaths(args.data_neg)

        # shuffle the negative filepaths with a fixed seed
        np.random.seed(0)
        np.random.shuffle(self.neg_filepaths)

        self.scale_factor = 1.414
        self.step_factor = 0.5

        # load the current negative sample index from disk
        self.last_neg_gen_idx_path = os.path.join(args.model,
                                                  'curr_neg_idx.txt')
        self._load_last_neg_gen_idx()
        # self.min_neg_batch_size = 100000
        self.bg_imgs_gray = []

        self.pos_data = tuple()
        self.curr_batch = tuple()

    def _get_filepaths(self, directory):
        """
        returns a list of filepaths of the files in the directory
        """
        filepaths = []
        for filename in sorted(os.listdir(directory)):
            filepath = os.path.join(directory, filename)
            if os.path.isfile(filepath):
                filepaths.append(filepath)
        return filepaths

    def _get_pos_paths(self):
        return self.pos_filepaths

    def _get_neg_paths(self):
        return self.neg_filepaths

    @staticmethod
    def _convert_image_to_grayscale(input_path):
        img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        return img

    def _load_image_grayscale_parallel_threading(self, img_paths):
        """
            returns a list of grayscale images of .png files in the folder
        """
        with concurrent.futures.ThreadPoolExecutor(
                max_workers=os.cpu_count()) as executor:
            imgs = list(
                executor.map(self._convert_image_to_grayscale, img_paths))
        return imgs

    @staticmethod
    def _preprocess_raw_images(imgs):
        """
        generates integral images, integral images of squares, and normalizing factors for each image
        """
        # take the square of the images
        imgs_sq = np.square(imgs, dtype=np.uint32)

        # calculate the integral image of each image
        imgs_int = np.array(
            [utils.to_integral_numba_uint32(img) for img in imgs])
        # calculate the integral image of the square of each image
        imgs_sq_int = np.array(
            [utils.to_integral_numba_uint32(img) for img in imgs_sq])

        # img = cv2.imread('/gscratch/emitlab/emitlab_projects/repos/traincascade/img.png', cv2.IMREAD_GRAYSCALE)
        # img_sq = np.square(img, dtype=np.uint16)
        # img_int = utils.to_integral(img)
        # img_sq_int = utils.to_integral(img_sq)
        # norm = utils.calc_norm(img_int, img_sq_int)

        # calculate the normalizing factor for each image
        imgs_nf = np.array([
            utils.calc_norm(img, img_sq)
            for img, img_sq in zip(imgs_int, imgs_sq_int)
        ],
                           dtype=np.float32)

        return imgs_int, imgs_sq_int, imgs_nf

    def _read_pos_raw_data(self):
        # use threading to read the positive samples in parallel
        start_time = time.time()
        imgs = self._load_image_grayscale_parallel_threading(
            self._get_pos_paths())
        logger.info('Time taken to read positive samples: {} seconds'.format(
            time.time() - start_time))

        # convert list of images to numpy array
        imgs = np.array(imgs)

        # preprocess the raw images
        imgs_int, imgs_sq_int, imgs_nf = self._preprocess_raw_images(imgs)

        # save the processed data
        self.pos_data = (imgs_int, imgs_sq_int, imgs_nf)

        return imgs_int, imgs_sq_int, imgs_nf

    def get_pos_data(self):
        """
        returns the processed data of all the positive samples - integral images, integral images of squares, and normalizing factors
        """
        return self.pos_data if self.pos_data else self._read_pos_raw_data()

    def _get_last_neg_gen_idx(self):
        """
        returns the last negative sample index and the last round of generation
        """
        return self.last_neg_gen_idx

    def _get_neg_filepath(self, idx):
        """
        returns the filepath of the negative sample with index idx
        """
        return self.neg_filepaths[idx]

    def _save_last_neg_gen_idx(self):
        """
        saves the current negative sample index to disk
        """
        stage = self.clf.get_stage()
        logger.info(
            f"Saving last_neg_gen_idx: {self.last_neg_gen_idx} to disk for stage {stage}"
        )
        with open(self.last_neg_gen_idx_path, 'w') as f:
            f.write(f"{self.last_neg_gen_idx}")

    def _load_last_neg_gen_idx(self):
        """
        loads the current negative sample index from disk
        """
        # if the stage is 0 OR if the file does not exist, set the curr neg sample index to 0
        if self.clf.get_stage() == 0 or not os.path.exists(
                self.last_neg_gen_idx_path):
            self.last_neg_gen_idx = 0
            return

        with open(self.last_neg_gen_idx_path, 'r') as f:
            lines = f.readlines()
        self.last_neg_gen_idx = int(lines[0])
        logger.info(
            f"Loaded last_neg_gen_idx: {self.last_neg_gen_idx} from disk")

    def _load_resize_img(self, filepath, idx):
        # load the image and convert it to grayscale
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

        # loop and created scaled images until the image is smaller than the default window size
        H, W = img.shape
        scale_factor = self.scale_factor

        # calculate the integral image of the image and the integral image of the square of the image
        img_sq = utils.square(img)
        int_im = utils.to_integral_numba(img)
        int_im_sq = utils.to_integral_numba(img_sq)

        out = [(img, int_im, int_im_sq, idx)]
        while True:
            H, W = int(H / scale_factor), int(W / scale_factor)
            if W < self.args.W or H < self.args.H: break
            img = cv2.resize(img, (W, H))

            img_sq = utils.square(img)
            int_im = utils.to_integral_numba(img)
            int_im_sq = utils.to_integral_numba(img_sq)

            # save the image to disk for debugging
            # cv2.imwrite(f'temp/check_img_{H}_{W}.png', img)

            out.append((img, int_im, int_im_sq, idx))

        return out

    def _load_bg_imgs(self):
        curr_file_idx = self._get_last_neg_gen_idx()
        BATCH_SIZE = 100
        start_idx = curr_file_idx
        end_idx = curr_file_idx + BATCH_SIZE

        # get the indices of the bg images
        bg_indices = list(
            np.arange(start_idx, end_idx) % len(self.neg_filepaths))

        # get the filepaths of the bg images
        bg_filepaths = [self._get_neg_filepath(idx) for idx in bg_indices]

        with concurrent.futures.ThreadPoolExecutor(
                max_workers=os.cpu_count()) as executor:
            out = list(
                executor.map(self._load_resize_img, bg_filepaths, bg_indices))

        # format the output
        img_gray = []
        img_int = []
        img_int_sq = []
        img_indices = []
        for sub_result in out:
            for img_gray_i, img_int_i, img_int_sq_i, img_idx_i in sub_result:
                img_gray.append(img_gray_i)
                img_int.append(img_int_i)
                img_int_sq.append(img_int_sq_i)
                img_indices.append(img_idx_i)

        # store the bg images
        self.bg_imgs_gray = img_gray
        self.bg_imgs_int = img_int
        self.bg_imgs_int_sq = img_int_sq
        self.bg_imgs_indices = img_indices

    def get_neg_data(self, num_samples):
        """
        returns a num_samples integral images, integral images of squares, and normalizing factors of negative samples
        """

        # if bg images are not loaded, load them, else use them
        if len(self.bg_imgs_gray) == 0:
            self._load_bg_imgs()
            self.bg_imgs_idx = 0

        count = 0
        consumed = 0

        # load the numpy array version of the cascade classifier
        wc_classifiers, wc_thresholds, wc_polarities, wc_alphas, thresholds_stage, start_idx_stage, end_idx_stage = self.clf.get_cascade_classifier(
        )

        # tracemalloc.start()

        neg_imgs_all = []
        windows_all = []
        H_all = []
        W_all = []
        bg_indices_all = []
        img_gray_all = []
        while count < num_samples:
            # check if there are enough bg images left
            if self.bg_imgs_idx >= len(self.bg_imgs_gray):
                # update the last_neg_gen_idx
                self.last_neg_gen_idx = (
                    self.bg_imgs_indices[self.bg_imgs_idx - 1] + 1) % len(
                        self.neg_filepaths)
                # load the next batch of bg images
                self._load_bg_imgs()
                self.bg_imgs_idx = 0

            # get the next image from the bg_imgs
            img_gray = self.bg_imgs_gray[self.bg_imgs_idx]
            int_im = self.bg_imgs_int[self.bg_imgs_idx]
            int_im_sq = self.bg_imgs_int_sq[self.bg_imgs_idx]
            self.bg_imgs_idx += 1

            # scan the image to get the hard negative imgs
            neg_imgs, windows, feat_vals_scan, hypotheses_scan, nf_scan, temp_consumed, max_size = _scan_image(
                wc_classifiers, wc_thresholds, wc_polarities, wc_alphas,
                thresholds_stage, start_idx_stage, end_idx_stage, img_gray,
                int_im, int_im_sq, self.args.W, img_gray.shape[0],
                img_gray.shape[1], self.args.W,
                self.args.maxFalseAlarmRate * 1.2,
                self.clf._get_stage() != 0)
            # neg_imgs, temp_consumed = ([np.ones(
            #     (self.args.W,
            #      self.args.W))], 1) if np.random.rand() < 0.1 else ([], 0)
            neg_imgs_all.extend(neg_imgs)
            windows_all.extend(windows)
            H_all.extend([img_gray.shape[0]] * len(neg_imgs))
            W_all.extend([img_gray.shape[1]] * len(neg_imgs))
            bg_indices_all.extend(
                [self.bg_imgs_indices[self.bg_imgs_idx - 1]] * len(neg_imgs))
            img_gray_all.extend([img_gray] * len(neg_imgs))
            # adjust consumed in the ratio of the false positives used
            if count + len(neg_imgs) > num_samples:
                ratio = (num_samples - count) / len(neg_imgs)
                temp_consumed = int(temp_consumed * ratio)
            consumed += temp_consumed
            count += len(neg_imgs)

        # update the last_neg_gen_idx
        self.last_neg_gen_idx = (self.bg_imgs_indices[self.bg_imgs_idx - 1] +
                                 1) % len(self.neg_filepaths)
        self._save_last_neg_gen_idx()

        # adjust the list as np array with the right dtype
        neg_imgs_all = np.array(neg_imgs_all[:num_samples], dtype=np.uint8)
        # neg_imgs_all = np.array(neg_imgs_all, dtype=np.uint8)

        # get the integral images, integral images of squares, and normalizing factors
        neg_int_all, neg_sq_int_all, neg_nf_all = self._preprocess_raw_images(
            neg_imgs_all)

        return neg_int_all, neg_sq_int_all, neg_nf_all, consumed


class imgStruct(NamedTuple):
    img_gray: np.ndarray
    img: np.ndarray


def load_images(img_paths):
    # load the images
    images = {}
    for img_path in img_paths:
        img = cv2.imread(img_path)
        # convert the image to grayscale
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        images[img_path] = imgStruct(img_gray=img_gray, img=img)

    return images


class TestDataReader:

    def __init__(self, args):
        self.args = args

        # check if test file exists
        if not os.path.exists(args.test_file):
            raise Exception('Test file does not exist')

        # load the test file using pandas
        self.test_df = pd.read_csv(args.test_file)

        # keep only the first 300 rows
        self.test_df = self.test_df[:300]

        unique_img_paths = self.test_df['abs_path'].unique()
        test_images = load_images(unique_img_paths)
        self.test_images = test_images

    def get_test_images(self):
        return self.test_images