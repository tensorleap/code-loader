from typing import Tuple, List

import numpy as np
import tensorflow as tf  # type: ignore
from numpy.typing import NDArray

from code_loader.helpers.detection.utils import xywh_to_xyxy_format, jaccard

DEFAULT_FEATURE_MAPS = ((80, 80), (40, 40), (20, 20))
DEFAULT_BOX_SIZES = (((10, 13), (16, 30), (33, 23)),
                     ((30, 61), (62, 45), (59, 119)),
                     ((116, 90), (156, 198), (373, 326)))  # tiny fd


def decode_bboxes(loc_pred: tf.Tensor, priors: tf.Tensor, variances: int = 1) -> NDArray[np.float32]:
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc_pred (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors, 4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions as follows:
        x_pred = x_anchor + x_anchor*x_delta_pred
        y_pred = y_anchor + y_anchor*y_delta_pred
        w_pred = exp(w_delta_pred*var)*w_anchor
        h_pred = exp(h_delta_pred*var)*h_anchor
    """
    # MAX_CLIP = 4.135166556742356
    # log_preds = tf.clip_by_value(loc_pred[:, 2:], clip_value_max=MAX_CLIP, clip_value_min=-np.inf)
    boxes = tf.concat([
        priors[:, :2] + loc_pred[:, :2],
        priors[:, 2:] * (loc_pred[:, 2:] * variances) ** 2
    ], 1)
    return xywh_to_xyxy_format(boxes)


def encode_bboxes(matched: tf.Tensor, priors: tf.Tensor, variances: Tuple[int, int]) -> tf.Tensor:
    """
    This encodes a [X,Y,W,H] GT into a list of matches priors s.t.

    :param matched: Tensor - matched GT for each prior [N_PRIORS,4]
    :param priors: Tensor - Priors [N_priors,4]
    :param variances: Variances used
    :return:
        encoded bounding box gt Tensor
    """
    g_cxcy = (matched[:, :2] - priors[:, :2]) / (variances[0])
    g_wh = tf.math.sqrt((matched[:, 2:]) / priors[:, 2:]) / variances[1]
    return tf.concat([g_cxcy, g_wh], 1)


def scale_loc_prediction(loc_pred: List[tf.Tensor], decoded: bool = False, image_size: float = 640.,
                         strides: Tuple[int, int, int] = (8, 16, 32)) -> \
        List[tf.Tensor]:
    new_loc_pred = [None] * len(loc_pred)
    if decoded:
        new_loc_pred = [loc / image_size for loc in loc_pred]
    else:
        for i in range(len(loc_pred)):
            new_loc_pred[i] = tf.concat([(strides[i] * (2 * tf.sigmoid(loc_pred[i][..., :2]) - 0.5)) / image_size,
                                         2 * tf.sigmoid(loc_pred[i][..., 2:])], axis=-1)
    return new_loc_pred


def reshape_output_list(keras_output: tf.Tensor, image_size: int, priors: int = 3,
                        feature_maps: Tuple[Tuple[int, int], ...] = ((80, 80), (40, 40), (20, 20)),
                        decoded: bool = False) -> \
        Tuple[List[tf.Tensor], List[tf.Tensor]]:
    """
    reshape the mode's output to two lists sized [NUM_FEATURES] following detectron2 convention.
    class_list item: (BATCH_SIZE, NUM_ANCHORS, CLASSES)
    loc_list item:  (BATCH_SIZE, NUM_ANCHORS, 4)
    """
    num_features = len(feature_maps)
    j = 0
    loc_pred_list = []
    class_pred_list = []
    for k in range(num_features):
        # add classes prediction
        num_elements = feature_maps[k][0] * feature_maps[k][1] * priors
        loc_pred_list.append(keras_output[:, j:j + num_elements, :4])
        class_pred_list.append(keras_output[:, j:j + num_elements, 4:])
        j += num_elements
    loc_pred_list = scale_loc_prediction(loc_pred_list, decoded, image_size=image_size)
    return class_pred_list, loc_pred_list


def match(threshold: float, truths: tf.Tensor, priors: tf.Tensor
          , labels: tf.Tensor, background_label: int,
          max_match_per_gt: int) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Matches between the GT and the anchors
    :param threshold:
    :param truths: (N_truths,4) - (X,Y,W,H)
    :param priors: (N_priors,4) - (X,Y,W,H)
    :param variances:
    :param labels: (N_truths) GT class
    :param background_label: int - the background label
    :return: loc (Tensor) [Npriors, 4], pred_label (Tensor) [Npriors]
    """
    # compute jaccard and best prior overlap and truth overlap
    if truths.shape[0] == 0:  # NO GT
        matches = tf.zeros_like(priors)
        return matches, tf.ones(priors.shape[0], dtype=tf.int32) * background_label
    overlaps = jaccard(xywh_to_xyxy_format(truths), xywh_to_xyxy_format(priors))  # (N_TRUTHS, N_PRIORS)
    best_prior_idx = tf.math.argmax(overlaps, axis=1)  # (NTRUTHS,)
    best_truth_overlap = tf.math.reduce_max(overlaps, axis=0, keepdims=True) # (1, N_PRIORS)
    best_truth_idx = tf.math.argmax(overlaps, axis=0)  # (N_PRIORS,)
    # keep only best k priors per GT
    # create an overlap vector that only fits in the specific
    # TODO fix the top_k in a way that will represent actual predictions
    k_truth_value, k_truth_idx = tf.nn.top_k(overlaps, k=min(max_match_per_gt, overlaps.shape[1]))
    unique_best_priors = tf.unique(tf.reshape(k_truth_idx, -1))[0]
    unique_values_priors = tf.gather(best_truth_overlap[0, ...], unique_best_priors)
    filtered_overlaps = tf.zeros_like(best_truth_overlap)[0,...]
    filtered_overlaps = tf.expand_dims(tf.tensor_scatter_nd_update(tensor=filtered_overlaps,
                                indices=tf.expand_dims(unique_best_priors, axis=1),
                                updates=unique_values_priors), axis=0)
    # FOR EACH GT, replace the value of the best fitting prior with the GT INDEX
    # THIS RATES ALL GT ACCORDING TO WHICH RESULT IN HIGHEST JACACRD
    matches = tf.gather(params=truths, indices=best_truth_idx)  # GT for each PRIOR (N_PRIOR, 4)
    pred_label = tf.gather(params=labels, indices=best_truth_idx)  # THIS IS THE BEST LABELS
    pred_label = tf.where(condition=filtered_overlaps < threshold, x=background_label, y=tf.cast(pred_label,
                                                                                                  tf.int32))  # eliminates low threshold
    pred_label = tf.squeeze(pred_label)  # (Nprior)
    return matches, pred_label  # decoded_gt, decoded_pred, ignore
