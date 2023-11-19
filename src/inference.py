import argparse

import cv2
import numpy as np
from numba import njit

import nms
import utils
import cascade_classifier


@njit(cache=True)
def _scan_image_infer(wc_classifiers, wc_thresholds, wc_polarities, wc_alphas,
                      thresholds_stage, start_idx_stage, end_idx_stage, int_im,
                      int_im_sq, window_start, H, W):
    STRIDE = 1
    max_size = 500
    # fp_imgs = np.zeros((max_size, window_start, window_start),
    #                    dtype=np.int32)  # store the false positive images
    windows = np.zeros((max_size, 4), dtype=np.int32)  # store the windows
    count = 0
    consumed = 0

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

            is_positive = True
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
                    is_positive = False
                    break

            consumed += 1
            # if predicted face, add to the list of bounding boxes
            if is_positive:
                windows[count, :] = np.array(
                    [x, y, x + window_start, y + window_start])
                out_nf[count, :] = nf
                count += 1

                if count == max_size:
                    return windows[:count, :], feat_vals[:count, :], \
                        out_hypotheses[:count, :], out_nf[:count,:], consumed, max_size

            # if not predicted face, move on to the next window
            y += STRIDE

        # move to the next row of windows
        x += STRIDE
        y = 0

    # search for candidate windows complete, return the list of candidate windows

    return windows[:count, :], feat_vals[:count, :], \
        out_hypotheses[:count, :], out_nf[:count, :], consumed, max_size


def get_all_pos_boxes(img, wc_classifiers, wc_thresholds, wc_polarities,
                          wc_alphas, thresholds_stage, start_idx_stage,
                          end_idx_stage, grid_size, min_obj_size, max_obj_size,
                          scale_factor):
    # first get the scaled versions of the img
    H, W = img.shape
    orig_H, orig_W = H, W
    orig_img = img.copy()

    # calculate the integral image of the image and the integral image of the square of the image
    img_sq = utils.square(img)
    int_im = utils.to_integral_numba(img)
    int_im_sq = utils.to_integral_numba(img_sq)

    scaled_imgs = [(img, int_im, int_im_sq)]
    while True:
        H, W = int(H / scale_factor), int(W / scale_factor)
        if W < grid_size or H < grid_size: break
        img = cv2.resize(img, (W, H))

        img_sq = utils.square(img)
        int_im = utils.to_integral_numba(img)
        int_im_sq = utils.to_integral_numba(img_sq)

        # save the image to disk for debugging
        # cv2.imwrite(f'temp/check_img_{H}_{W}.png', img)

        scaled_imgs.append((img, int_im, int_im_sq))

    final_bboxes = []
    final_bboxes = np.empty((0, 4), dtype=np.int32)
    j = 0
    for img, int_im, int_im_sq in scaled_imgs:
        boxes, feat_vals, out_hypotheses, out_nf, consumed, max_size = _scan_image_infer(
            wc_classifiers, wc_thresholds, wc_polarities, wc_alphas,
            thresholds_stage, start_idx_stage, end_idx_stage, int_im,
            int_im_sq, grid_size, img.shape[0], img.shape[1])
        # adjust the bounding boxes to the original image size
        for i in range(len(boxes)):
            # # add the bounding box to the img
            # cv2.rectangle(img, (boxes[i, 0], boxes[i, 1]),
            #               (boxes[i, 2], boxes[i, 3]), (0, 255, 0), 1)
            scale = orig_H / img.shape[0]
            boxes[i, :] = (boxes[i, :] * scale).astype(np.int32)
            # # add the scaled bounding box to the original image
            # cv2.rectangle(orig_img, (boxes[i, 0], boxes[i, 1]),
            #               (boxes[i, 2], boxes[i, 3]), (0, 255, 0), 1)

        final_bboxes = np.vstack((final_bboxes, boxes))

        # # save the image to disk for debugging
        # cv2.imwrite(f'temp/new_scan/img_{j}.png', img)
        # cv2.imwrite(f'temp/new_scan/orig_img_{j}.png', orig_img)
        j += 1

    return final_bboxes


def get_predictions(cascade_clf, img, stage, grid_size, iou_threshold, scale,
                    min_obj_size, max_obj_size, minNeighbors):
    """get the predictions for the given stage"""

    # load the numpy array version of the cascade classifier
    wc_classifiers, wc_thresholds, wc_polarities, wc_alphas, thresholds_stage, start_idx_stage, end_idx_stage = cascade_clf.get_cascade_classifier(
        stage)

    all_bboxes = get_all_pos_boxes(img, wc_classifiers, wc_thresholds,
                                       wc_polarities, wc_alphas,
                                       thresholds_stage, start_idx_stage,
                                       end_idx_stage, grid_size, min_obj_size,
                                       max_obj_size, 1.25)

    # remove overlapping bounding boxes
    aggregated_bboxes = nms.nms_viola_jones(all_bboxes.copy(), 0.6, minNeighbors)
    final_boxes = nms.merge_overlapping_boxes(aggregated_bboxes, 0.5)

    return all_bboxes, aggregated_bboxes, final_boxes


if __name__ == "__main__":
    # Parse the command line arguments
    parser = argparse.ArgumentParser(
        description='Benchmark a cascade classifier')
    parser.add_argument(
        '-model',
        type=str,
        required=True,
        help='Directory containing the trained cascade classifier')
    parser.add_argument('-W',
                        type=int,
                        default=24,
                        help='Width of the samples')
    parser.add_argument('-H',
                        type=int,
                        default=24,
                        help='Height of the samples')
    parser.add_argument('-stride',
                        type=int,
                        default=1,
                        help='Step size for the sliding window')
    parser.add_argument('-scale',
                        type=float,
                        default=1.25,
                        help='Scale factor for the sliding window')
    parser.add_argument('-minNeighbors',
                        type=int,
                        default=3,
                        help='Minimum number of neighbors for a detection')
    parser.add_argument('-stage',
                        type=int,
                        default=-1,
                        help='Stage of the cascade classifier to use')
    parser.add_argument('-min_obj_size',
                        type=int,
                        default=24,
                        help='Minimum size of the objects to detect')
    parser.add_argument('-max_obj_size',
                        type=int,
                        default=2000,
                        help='Maximum size of the objects to detect')
    parser.add_argument('-image',
                        type=str,
                        default=None,
                        help='Image to run the inference on')
    
    args = parser.parse_args()
    args.train = False

    # load the cascade classifier
    cascade_clf = cascade_classifier.CascadeClassifier(args)

    # load the img to run the inference on
    img = cv2.imread(args.image)

    # convert the img to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # get the predictions for the given stage
    all_bboxes, agg_bboxes, final_bboxes = get_predictions(
            cascade_clf, img_gray, args.stage, args.H, 0.5, args.scale, 
            args.min_obj_size, args.max_obj_size, args.minNeighbors)
    
    # draw the bounding boxes on the image
    for bbox in final_bboxes:
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0),
                      2)
        
    # save the image to disk
    cv2.imwrite('temp/infer.jpg', img)
