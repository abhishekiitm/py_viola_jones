import os
import json
import logging
import time
from typing import NamedTuple, Tuple, List, Callable

import numpy as np
from numba import prange, njit
import pandas as pd

import data_reader
import features_viola_haar as ft

logger = logging.getLogger(__name__)


class ThresholdPolarity(NamedTuple):
    threshold: float
    polarity: float


class ClassifierResult(NamedTuple):
    threshold: float
    polarity: int
    classification_error: float
    classifier: Callable[[np.ndarray], float]


class WeakClassifier(NamedTuple):
    threshold: float
    polarity: int
    alpha: float
    classifier: Callable[[np.ndarray], float]


@njit(parallel=True, cache=True)
def compute_cascade_classifier(xis, nf, wc_classifiers, wc_thresholds,
                               wc_polarities, wc_alphas, thresholds_stage,
                               start_idx_stage, end_idx_stage):
    """
    Takes in a list of 24x24 images and returns a list of 0s and 1s indicating whether the image contains a face or not.
    """

    # compute the cascade classifier on each of the integral images
    preds = np.zeros(xis.shape[0], dtype=np.int32)

    # create a list of 1000's for storing the feature values for all the weak classifiers in all the stages
    feat_vals = np.zeros(
        (xis.shape[0], len(wc_classifiers)), dtype=np.float32) + 1000.
    # create a list of zeros for storing the sum_hypotheses for all the stages
    out_hypotheses = np.zeros((xis.shape[0], len(start_idx_stage)),
                              dtype=np.float32)

    for n in prange(xis.shape[0]):
        is_positive = True

        # loop through all the stages of the cascade classifier
        for stage in range(len(start_idx_stage)):
            # select the indices of the start and end of the weak classifiers for the current stage
            start_idx = start_idx_stage[stage]
            end_idx = end_idx_stage[stage]

            sum_hypotheses = 0.
            # loop through all the weak classifiers of the current stage and sum the hypotheses and alphas
            for j in range(start_idx, end_idx + 1):
                # select the weak classifier
                feat = wc_classifiers[j]
                x, y, w, h = feat[1], feat[2], feat[3], feat[4]

                # compute the feature value
                if feat[0] == 1:
                    # feature type 2h
                    hw = w // 2
                    feat_val = 2 * (xis[n, y + h, x + hw]
                                    - xis[n, y, x + hw]) + \
                            xis[n, y, x] - xis[n, y + h, x] + \
                            xis[n, y, x + w] - xis[n, y + h, x + w]

                elif feat[0] == 2:
                    # feature type 2v
                    hh = h // 2
                    feat_val = 2 * (xis[n, y + hh, x]
                                    - xis[n, y + hh, x + w]) + \
                            xis[n, y, x+w] - xis[n, y, x] + \
                            xis[n, y + h, x + w] - xis[n, y + h, x]

                elif feat[0] == 3:
                    # feature type 3h
                    tw = w // 3
                    feat_val = 2 * (xis[n, y + h, x + 2 * tw]) + \
                            2 * xis[n, y, x + tw] - 2 * xis[n, y + h, x + tw] - \
                            2 * xis[n, y, x + 2 * tw] + \
                            xis[n, y + h, x] - xis[n, y, x] + \
                            xis[n, y, x + w] - xis[n, y + h, x + w]

                elif feat[0] == 4:
                    # feature type 3v
                    th = h // 3
                    feat_val = 2 * (xis[n, y + 2 * th, x + w]) + \
                            2 * xis[n, y + th, x] - 2 * xis[n, y + th, x + w] - \
                            2 * xis[n, y + 2 * th, x] + \
                            xis[n, y, x + w] - xis[n, y, x] + \
                            xis[n, y + h, x] - xis[n, y + h, x + w]

                elif feat[0] == 5:
                    # feature type 4
                    hw = w // 2
                    hh = h // 2
                    feat_val = 4 * xis[n, y + hh, x + hw] - \
                            2 * xis[n, y, x + hw] - 2 * xis[n, y + hh, x] - \
                            2 * xis[n, y + hh, x + w] - 2 * xis[n, y + h, x + hw] + \
                            xis[n, y, x] + xis[n, y + h, x] + \
                            xis[n, y, x + w] + xis[n, y + h, x + w]

                # store the feature value
                feat_vals[n, j] = feat_val
                # normalize the feature value
                feat_val = 0 if nf[n] == 0 else feat_val / nf[n]

                vote = (np.sign((wc_polarities[j] * wc_thresholds[j]) -
                                (wc_polarities[j] * feat_val)) + 1) // 2
                sum_hypotheses += wc_alphas[j] * vote

            # store the sum_hypotheses for the current stage
            out_hypotheses[n, stage] = sum_hypotheses

            # compute the prediction for the current stage
            if sum_hypotheses < thresholds_stage[stage]:
                # predicted not a face by the cascade classifier, break out of the loop
                is_positive = False
                break

        if is_positive: preds[n] = 1

    return preds, feat_vals, out_hypotheses


@njit(parallel=True, cache=True)
def argsort_features(computed_features: np.ndarray) -> np.ndarray:
    # create zero array for storing the indices
    sorted_indices = np.zeros_like(computed_features, dtype=np.int32)

    # loop over all features and store the indices of the sorted features
    for i in prange(computed_features.shape[0]):
        sorted_indices[i] = np.argsort(computed_features[i])

    return sorted_indices


@njit(cache=True)
def normalize_weights(w: np.ndarray) -> np.ndarray:
    return w / w.sum()


@njit(parallel=True, cache=True)
def argmin(x):
    return np.argmin(x)


@njit(cache=True)
def build_running_sums(
        ys: np.ndarray,
        ws: np.ndarray) -> Tuple[float, float, List[float], List[float]]:
    s_minus, s_plus = 0., 0.
    t_minus, t_plus = 0., 0.
    s_minuses, s_pluses = np.zeros_like(ws), np.zeros_like(ws)

    # for y, w in zip(ys, ws):
    for i in prange(len(ys)):
        y = ys[i]
        w = ws[i]
        if y < .5:
            s_minus += w
            t_minus += w
        else:
            s_plus += w
            t_plus += w
        s_minuses[i] = s_minus
        s_pluses[i] = s_plus
    return t_minus, t_plus, s_minuses, s_pluses


@njit(cache=True)
def build_running_sums_2(
        ys: np.ndarray,
        ws: np.ndarray) -> Tuple[float, float, List[float], List[float]]:
    s_minus, s_plus = 0., 0.
    t_minus, t_plus = 0., 0.
    s_minuses, s_pluses = np.zeros(len(ys) + 1), np.zeros(len(ys) + 1)

    # for y, w in zip(ys, ws):
    for i in prange(len(ys)):
        y = ys[i]
        w = ws[i]
        if y < .5:
            s_minus += w
            t_minus += w
        else:
            s_plus += w
            t_plus += w
        s_minuses[i + 1] = s_minus
        s_pluses[i + 1] = s_plus
    return t_minus, t_plus, s_minuses, s_pluses


@njit(cache=True)
def find_best_threshold(zs: np.ndarray, t_minus: float, t_plus: float,
                        s_minuses: List[float], s_pluses: List[float],
                        weight_pos: float) -> ThresholdPolarity:
    min_e = float(1e7)
    min_z, polarity = 0.0, 0.0
    for z, s_m, s_p in zip(zs, s_minuses, s_pluses):
        error_1 = weight_pos * s_p + (t_minus - s_m)
        error_2 = s_m + weight_pos * (t_plus - s_p)
        if error_1 < min_e:
            min_e = error_1
            min_z = z
            polarity = -1
        elif error_2 < min_e:
            min_e = error_2
            min_z = z
            polarity = 1
    return ThresholdPolarity(threshold=min_z, polarity=polarity)


@njit(cache=True)
def find_best_threshold_2(zs: np.ndarray, t_minus: float, t_plus: float,
                          s_minuses: List[float], s_pluses: List[float],
                          weight_pos: float) -> ThresholdPolarity:
    min_e = float(1e7)
    EPS = 1e-7
    min_z, polarity = 0.0, 0.0
    # for z, s_m, s_p in zip(zs, s_minuses, s_pluses):
    for i in prange(1, len(zs)):
        z = zs[i]
        # only update if z is sufficiently larger than the previous z
        if z < zs[i - 1] + EPS: continue
        s_m = s_minuses[i]
        s_p = s_pluses[i]
        error_1 = weight_pos * s_p + (t_minus - s_m)
        error_2 = s_m + weight_pos * (t_plus - s_p)
        if error_1 < min_e:
            min_e = error_1
            # min_z = z - EPS / 2
            min_z = z
            polarity = -1
        elif error_2 < min_e:
            min_e = error_2
            # min_z = z + EPS / 2
            min_z = z
            polarity = 1
    return ThresholdPolarity(threshold=min_z, polarity=polarity)


@njit(cache=True)
def determine_threshold_polarity_new(
        ys: np.ndarray,
        ws: np.ndarray,
        zs: np.ndarray,
        weight_pos: float = 1) -> ThresholdPolarity:

    # Determine the best threshold: build running sums
    # t_minus, t_plus, s_minuses, s_pluses = build_running_sums(ys, ws)
    t_minus, t_plus, s_minuses, s_pluses = build_running_sums_2(ys, ws)

    # Determine the best threshold: select optimal threshold.
    # return find_best_threshold(zs, t_minus, t_plus, s_minuses, s_pluses,
    #                            weight_pos)
    return find_best_threshold_2(zs, t_minus, t_plus, s_minuses, s_pluses,
                                 weight_pos)


@njit(parallel=True, cache=True)
def evaluate_features(computed_features, sorted_indices, ys, ws):
    thresholds = np.zeros(computed_features.shape[0], dtype=np.float32)
    polarities = np.zeros(computed_features.shape[0], dtype=np.int32)
    classification_errors = np.zeros(computed_features.shape[0],
                                     dtype=np.float32)

    for i in prange(computed_features.shape[0]):
        zi = computed_features[i]
        yi = ys
        wi = ws
        p = sorted_indices[i]

        # arrange by the sorted indices
        zi, yi, wi = zi[p], yi[p], wi[p]

        # Determine the best threshold:
        result = determine_threshold_polarity_new(yi, wi, zi)

        # Determine the classification error
        classification_error = 0.
        for z, y, w in zip(zi, yi, wi):
            h = (np.sign((result.polarity * result.threshold) -
                         (result.polarity * z)) + 1) // 2
            # h_list.append(h)
            classification_error += w * np.abs(h - y)

        thresholds[i] = result.threshold
        polarities[i] = result.polarity
        classification_errors[i] = classification_error

    return thresholds, polarities, classification_errors


@njit(cache=True)
def update_weights(ws: np.ndarray, y: np.ndarray, ft: np.ndarray, beta: float,
                   threshold: float, polarity: int) -> np.ndarray:
    for n in range(len(ft)):
        feat_val = ft[n]
        h_i = (np.sign((polarity * threshold) -
                       (polarity * feat_val)) + 1) // 2
        e_i = np.abs(h_i - y[n])
        ws[n] = ws[n] * np.power(beta, 1 - e_i)

    return ws


@njit(cache=True)
def calc_threshold_stage(wc_alphas, wc_thresholds, wc_polarities, wc_features,
                         num_pos, min_hit_rate):
    # first compute the cascade classifier output for the training data set
    zs = np.zeros(wc_features.shape[1], dtype=np.float32)

    # loop through each sample in the training data set
    for n in range(wc_features.shape[1]):
        # loop through all the weak classifiers of the current stage and sum the hypotheses
        sum_hypotheses = 0.
        for j in range(len(wc_alphas)):
            feat_val = wc_features[j][n]
            vote = (np.sign((wc_polarities[j] * wc_thresholds[j]) -
                            (wc_polarities[j] * feat_val)) + 1) // 2
            sum_hypotheses += wc_alphas[j] * vote

        zs[n] = sum_hypotheses

    # first num_pos samples are positive, rest are negative
    # to calculate the threshold, we need to sort the zs for only the positive samples
    zs_pos_sorted = np.sort(zs[:num_pos])

    # calculate the threshold for the current stage
    threshold_idx = int(np.floor((1 - min_hit_rate) * num_pos))
    threshold_stage = zs_pos_sorted[threshold_idx]

    # calculate the hit rate and false alarm rate for the current stage
    hit_rate = np.sum(zs[:num_pos] >= threshold_stage) / num_pos
    false_alarm_rate = np.sum(zs[num_pos:] >= threshold_stage) / (zs.shape[0] -
                                                                  num_pos)

    return threshold_stage, hit_rate, false_alarm_rate


class CascadeClassifier:

    def __init__(self, args):
        self.args = args
        if args.train:
            self._initialize_train_args()
        else:
            self._initialize_inference_args()

    def _initialize_train_args(self):
        args = self.args
        # check if model directory exists
        if not os.path.exists(args.model):
            raise Exception('Model directory does not exist')

        # check if width and height are positive integers
        if args.W <= 0 or args.H <= 0:
            raise Exception('Width and height must be positive integers')

        # check if minimum hit rate and maximum false alarm rate are between 0 and 1
        if args.minHitRate < 0 or args.minHitRate > 1:
            raise Exception('Minimum hit rate must be between 0 and 1')

        # check if number of positive samples, number of negative samples, and number of stages are positive integers
        if args.numPos <= 0 or args.numNeg <= 0 or args.numStages <= 0:
            raise Exception(
                'Number of positive samples, number of negative samples, and number of stages must be positive integers'
            )

        # check if acceptance ratio break value is between 0 and 1
        if args.acceptanceRatioBreakValue < 0 or args.acceptanceRatioBreakValue > 1:
            raise Exception(
                'Acceptance ratio break value must be between 0 and 1')

        self.args = args

        # if restartPreviousTraining is True, delete all files in the model directory
        if not args.reUsePreviousTraining:
            for filename in os.listdir(args.model):
                os.remove(os.path.join(args.model, filename))

        # load the saved cascade classifier if it exists in the model directory
        # if it does not exist, a new cascade classifier is created
        self.load_classifier()

        # initialize the data reader
        self.data_reader = data_reader.DataReader(args, self)

    def _initialize_inference_args(self):
        args = self.args
        # check if the image exists
        if not os.path.exists(args.image):
            raise Exception('Image path does not exist')

        # check if model directory exists
        if not os.path.exists(args.model):
            raise Exception('Model directory does not exist')

        # check if the model directory contains cascade_clf_{stage}.json files
        cascade_clf_filenames = [
            filename for filename in os.listdir(args.model) if
            filename.startswith("cascade_clf_") and filename.endswith(".json")
        ]
        if len(cascade_clf_filenames) == 0:
            raise Exception(
                'Model directory does not contain any cascade classifier')

        # check if width and height are positive integers
        if args.W <= 0 or args.H <= 0:
            raise Exception('Width and height must be positive integers')

        # check if stride is a positive integer
        if args.stride <= 0:
            raise Exception('Stride must be a positive integer')

        # check if scale is a positive float
        if args.scale <= 0:
            raise Exception('Scale must be a positive float')

        # check if minNeighbors is a positive integer
        if args.minNeighbors <= 0:
            raise Exception('minNeighbors must be a positive integer')

        # load the saved cascade classifier
        self.load_classifier()

    def _get_max_stage_saved_classifiers(self):
        # get all cascade_clf_{stage}.json files in the model directory
        cascade_clf_filenames = [
            filename for filename in os.listdir(self.args.model) if
            filename.startswith("cascade_clf_") and filename.endswith(".json")
        ]
        # extract number between cascade_clf_ and .json
        stage_nums = [
            int(filename[12:-5]) for filename in cascade_clf_filenames
        ]

        return max(stage_nums) if len(stage_nums) > 0 else 0

    def load_classifier(self):
        """
        Load the saved cascade classifier if it exists in the model directory
        """
        # check for "cascade_clf_{stage}.json" files in the model directory
        stage = self._get_max_stage_saved_classifiers()
        if stage == 0:
            self.wc_classifiers_all = []
            self.wc_thresholds_all = []
            self.wc_polarities_all = []
            self.wc_alphas_all = []
            self.thresholds_stage = []
            self.start_idx_stage = []
            self.end_idx_stage = []
            return

        cascade_clf_filename = os.path.join(self.args.model,
                                            f"cascade_clf_{stage}.json")
        with open(cascade_clf_filename, 'r') as ip:
            in_dict = json.load(ip)

        self.wc_classifiers_all = in_dict["wc_classifiers"]
        self.wc_thresholds_all = in_dict["wc_thresholds"]
        self.wc_polarities_all = in_dict["wc_polarities"]
        self.wc_alphas_all = in_dict["wc_alphas"]
        self.thresholds_stage = in_dict["thresholds_stage"]
        self.start_idx_stage = in_dict["start_idx_stage"]
        self.end_idx_stage = in_dict["end_idx_stage"]

    def save_classifier(self):
        out_dict = {}
        out_dict["wc_classifiers"] = np.array(self.wc_classifiers_all).tolist()
        out_dict["wc_thresholds"] = np.array(self.wc_thresholds_all).tolist()
        out_dict["wc_polarities"] = np.array(self.wc_polarities_all).tolist()
        out_dict["wc_alphas"] = np.array(self.wc_alphas_all).tolist()
        out_dict["thresholds_stage"] = np.array(self.thresholds_stage).tolist()
        out_dict["start_idx_stage"] = np.array(self.start_idx_stage).tolist()
        out_dict["end_idx_stage"] = np.array(self.end_idx_stage).tolist()

        out_filename = os.path.join(self.args.model,
                                    f"cascade_clf_{self._get_stage()}.json")
        with open(out_filename, 'w') as op:
            json.dump(out_dict, op, indent=2)

    def get_cascade_classifier(self, stage=None):
        if stage is None: stage = self._get_stage()

        if stage == 0:
            return np.array(
                [np.array([0, 0, 0, 0, 0], dtype=np.int32)],
                dtype=np.int32), np.array([0.0], dtype=np.float32), np.array(
                    [0], dtype=np.int32), np.array(
                        [0.0], dtype=np.float32), np.array(
                            [0], dtype=np.float32), np.array(
                                [0], dtype=np.int32), np.array([0],
                                                               dtype=np.int32)

        end_idx = self.end_idx_stage[stage - 1]
        wc_classifiers = np.array(self.wc_classifiers_all[:end_idx + 1],
                                  dtype=np.int32)
        wc_thresholds = np.array(self.wc_thresholds_all[:end_idx + 1],
                                 dtype=np.float32)
        wc_polarities = np.array(self.wc_polarities_all[:end_idx + 1],
                                 dtype=np.int32)
        wc_alphas = np.array(self.wc_alphas_all[:end_idx + 1],
                             dtype=np.float32)
        thresholds_stage = np.array(self.thresholds_stage[:stage],
                                    dtype=np.float32)
        start_idx_stage = np.array(self.start_idx_stage[:stage],
                                   dtype=np.int32)
        end_idx_stage = np.array(self.end_idx_stage[:stage], dtype=np.int32)

        return wc_classifiers, wc_thresholds, wc_polarities, wc_alphas, thresholds_stage, start_idx_stage, end_idx_stage

    def get_stage(self):
        return self._get_stage()

    def _get_stage(self):
        """
        Get the size of the current cascade classifier
        """
        return len(self.thresholds_stage)

    def _filter_incorrect_preds(self,
                                img_int,
                                img_int_sq,
                                img_nf,
                                keep_pred=0):
        """
        Remove false negatives / false positives from the data set
        """

        # if stage is 0, there are no false negatives
        if self._get_stage() == 0: return img_int, img_int_sq, img_nf

        # get the latest cascade classifier
        wc_classifiers, wc_thresholds, wc_polarities, wc_alphas, thresholds_stage, start_idx_stage, end_idx_stage = self.get_cascade_classifier(
        )

        # compute the cascade classifier output for the positive samples
        preds, _, _ = compute_cascade_classifier(
            img_int, img_nf, wc_classifiers, wc_thresholds, wc_polarities,
            wc_alphas, thresholds_stage, start_idx_stage, end_idx_stage)

        # filter out the predictions that are equal to keep_pred
        img_int = img_int[preds == keep_pred]
        img_int_sq = img_int_sq[preds == keep_pred]
        img_nf = img_nf[preds == keep_pred]

        return img_int, img_int_sq, img_nf

    def update_training_data(self):
        """
        Update the training data set that will be used to train the current stage 
        includes removing hard negatives correctly classified by the current cascade classifier
        as well as removing positives that are not correctly classified by the current cascade classifier
        """
        # get the positive samples and negative samples and the samples consumed while getting them

        pos_int, pos_int_sq, pos_nf = self.data_reader.get_pos_data()
        pos_initial = len(pos_int)
        pos_int, pos_int_sq, pos_nf = self._filter_incorrect_preds(pos_int,
                                                                   pos_int_sq,
                                                                   pos_nf,
                                                                   keep_pred=1)
        pos_removed = pos_initial - len(pos_int)

        # keep only the first numPos positive samples
        pos_int = pos_int[:self.args.numPos]
        pos_int_sq = pos_int_sq[:self.args.numPos]
        pos_nf = pos_nf[:self.args.numPos]

        neg_int_all, neg_int_sq_all, neg_nf_all, neg_consumed = self.data_reader.get_neg_data(
            self.args.numNeg)

        neg_sample_count = len(neg_int_all)
        self.acceptance_ratio = 0 if neg_consumed == 0 else neg_sample_count / neg_consumed

        logger.info(
            f"Pos count: {len(pos_int)} : Pos removed ratio: {pos_removed / pos_initial}"
        )
        logger.info(
            f"Neg count: {len(neg_int_all)} : Neg consumed: {neg_consumed}")
        logger.info(f"Acceptance ratio: {self.acceptance_ratio}")

        # check if no of positive samples is equal to numPos and no of negative samples is equal to numNeg
        if len(pos_int) != self.args.numPos or len(
                neg_int_all) != self.args.numNeg:
            raise Exception(
                f"No of pos samples: {len(pos_int)} and no of neg samples: {len(neg_int_all)} is equal to numPos: {self.args.numPos} and numNeg: {self.args.numNeg}"
            )

        # concatenate pos_int and neg_int_all to get the training data set for the current stage
        self.training_data = (
            np.concatenate((pos_int, neg_int_all), axis=0),
            np.concatenate((pos_int_sq, neg_int_sq_all), axis=0),
            np.concatenate((pos_nf, neg_nf_all), axis=0),
            np.concatenate((np.ones(len(pos_int)), np.zeros(len(neg_int_all))),
                           axis=0),
        )

    @staticmethod
    def _initialize_weights(y):
        m = len(y[y == 0])
        n = len(y[y == 1])
        ws = np.zeros_like(y, dtype=np.float32)
        ws[y == 0] = 1 / (2 * m)
        ws[y == 1] = 1 / (2 * n)
        return ws

    def add_weak_classifier(self, wc_classifiers, wc_thresholds, wc_polarities,
                            wc_alphas, threshold):
        start_idx = len(self.wc_classifiers_all)
        end_idx = start_idx + len(wc_classifiers) - 1
        self.wc_classifiers_all.extend(wc_classifiers)
        self.wc_thresholds_all.extend(wc_thresholds)
        self.wc_polarities_all.extend(wc_polarities)
        self.wc_alphas_all.extend(wc_alphas)
        self.thresholds_stage.append(threshold)
        self.start_idx_stage.append(start_idx)
        self.end_idx_stage.append(end_idx)

    def _train_stage(self):
        """
        Train the current stage of the cascade classifier using AdaBoost
        """

        # Step 0: Initialize the variables for the current stage
        weak_classifiers = []
        wc_thresholds = []
        wc_polarities = []
        wc_classifiers = []
        wc_alphas = []
        wc_features = []
        num_weak_clfs = 0

        # Step 1: Get the training data set for the current stage
        img_int, img_int_sq, nf, y = self.training_data

        # Step 2: Initialize the weights for the training data set
        ws = self._initialize_weights(y)

        # Step 3: Precalculate the feature values for the training data set and get argsort them
        fts = ft.compute_features_all_imgs(self.features, img_int, nf)
        sorted_indices = argsort_features(fts)

        logger.info(f"Training stage {self._get_stage()}")
        logger.info(f"N   |   HR     |   FAR")
        # Step 4: Start training the weak classifiers one by one (AdaBoost) until the maximum false alarm rate is reached
        while True:  # change this condition later to halt adding weak classifiers
            # Step 4.1: Normalize the weights
            ws = normalize_weights(ws)

            # Step 4.2: Evaluate the error of each weak classifier
            thresholds, polarities, classification_errors = evaluate_features(
                fts, sorted_indices, y, ws)

            # Step 4.3: Select the weak classifier with the lowest error
            best_idx = argmin(classification_errors)
            best = ClassifierResult(
                threshold=thresholds[best_idx],
                polarity=polarities[best_idx],
                classifier=best_idx,
                classification_error=classification_errors[best_idx])

            # Step 4.4: Compute the alpha, beta, and new weights
            beta = best.classification_error / (1 - best.classification_error)
            # if the error is 0, set beta to 0.00001
            if beta == 0: beta = 0.00001
            alpha = np.log(1. / beta)

            # Step 4.5: Build the new weak classifier
            weak_clf = WeakClassifier(
                threshold=best.threshold,
                polarity=best.polarity,
                classifier=self.features[best.classifier],
                alpha=alpha)

            # Step 4.6: Register the new weak classifier
            weak_classifiers.append(weak_clf)
            wc_thresholds.append(best.threshold)
            wc_polarities.append(best.polarity)
            wc_classifiers.append(self.features[best.classifier])
            wc_alphas.append(alpha)
            wc_features.append(fts[best.classifier])
            num_weak_clfs += 1

            # Step 4.7: Calculate the threshold for the current stage and check if the maximum false alarm rate is reached
            threshold_stage, hit_rate, false_alarm_rate = calc_threshold_stage(
                np.array(wc_alphas), np.array(wc_thresholds),
                np.array(wc_polarities), np.array(wc_features),
                self.args.numPos, self.args.minHitRate)
            # format the hit rate and false alarm rate to be displayed as percentages with 1 decimal places and two digits before the decimal point
            logger.info(
                f"{num_weak_clfs}   |   {hit_rate * 100:.1f}%  |   {false_alarm_rate * 100:.1f}%"
            )
            # stop training the current stage if false alarm rate is less than the maximum false alarm rate or if the maximum number of weak classifiers is reached
            if false_alarm_rate < self.args.maxFalseAlarmRate or num_weak_clfs >= self.args.maxWeakCount:
                break

            # Step 4.8: Update the weights
            ws = update_weights(ws, y, fts[best.classifier], beta,
                                best.threshold, best.polarity)

        # Step 5: Add the weak classifiers to the cascade classifier
        self.add_weak_classifier(wc_classifiers, wc_thresholds, wc_polarities,
                                 wc_alphas, threshold_stage)

    def initialize_features(self):
        """
        Initialize the haar features which will be used for training
        """
        self.features = ft.generate_haar_features(self.args.W)
        # self.features = self.features[::20]

    def train(self):

        # get the start stage
        start_stage = self._get_stage()

        # initialize the features
        self.initialize_features()

        data_gen_times = []
        train_times = []
        # loop through the stages
        for stage in range(start_stage, self.args.numStages):
            # code for training each stage of the cascade classifier
            # each stage is a strong classifier made up of weak classifiers (decision stumps) trained using AdaBoost

            start_time = time.time()
            # Step 1: Update the training data set that will be used to train the current stage (hard negative mining)
            self.update_training_data()
            data_gen_time = time.time() - start_time
            # return
            logger.info(f"Training dataset for stage {stage} updated")

            # Step 2: Check if training termination conditions are met ie. if the acceptance ratio is less than the acceptance ratio break value
            if 0 < self.acceptance_ratio < self.args.acceptanceRatioBreakValue:
                logger.info(
                    f"Acceptance ratio {self.acceptance_ratio} is less than the acceptance ratio break value {self.args.acceptanceRatioBreakValue}.\nTerminating training."
                )
                break

            # Step 3: Train the weak classifiers using AdaBoost
            self._train_stage()
            logger.info(f"Stage {stage} trained")
            train_time = time.time() - start_time - data_gen_time

            # Step 4: Save the cascade classifier
            self.save_classifier()
            logger.info(f"Cascade classifier for stage {stage} saved")
            logger.info("-" * 50)
            logger.info("-" * 50)

            # Append the training time and data generation time
            data_gen_times.append(data_gen_time)
            train_times.append(train_time)

        # Save the training times and data generation times
        stages = list(range(start_stage, self.args.numStages))[:len(
            data_gen_times)]
        timing_df = pd.DataFrame({
            "stage": stages,
            "data_gen_time": data_gen_times,
            "train_time": train_times
        })

        timing_df.to_csv(os.path.join(self.args.results_dir,
                                      "timing_baseline.csv"),
                                      index=False)
        