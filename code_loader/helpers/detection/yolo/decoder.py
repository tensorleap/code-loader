from typing import List

import numpy as np
import tensorflow as tf  # type: ignore
from numpy.typing import NDArray

from code_loader.helpers.detection.utils import xywh_to_xyxy_format
from code_loader.helpers.detection.yolo.utils import decode_bboxes


class Decoder:
    """At test time, Detect is the final layer of SSD.
    Consists of 4 steps:
    Bounding Boxes Decoding,
    Confidence Thresholding,
    Non-Max Suppression,
    Top-K Filtering.
    """

    def __init__(self, num_classes: int, background_label: int, top_k: int, conf_thresh: float, nms_thresh: float,
                 max_bb_per_layer: int = 20, max_bb: int = 20):
        self.num_classes = num_classes
        self.background_label = background_label
        self.top_k = top_k
        self.max_bb_per_layer = max_bb_per_layer
        self.max_bb = max_bb
        # Parameters used in nms.
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh

    def __call__(self, loc_data: List[tf.Tensor], conf_data: List[tf.Tensor], prior_data: List[NDArray[np.float32]],
                 from_logits: bool = True, decoded: bool = False) -> List[NDArray[np.float32]]:
        """
        Args:
            loc_data: (tensor) Location preds from loc layers
                Shape: [batch, num_priors*4]
            conf_data: (tensor) Shape: Confidence preds from confidence layers
                Shape: [batch*num_priors, num_classes]
            prior_data: (tensor) Prior boxes and variances? from priorbox layers
                Shape: [1, num_priors, 4]
        """
        # loc_data: (batch_size, num_priors, 4)
        # conf_data: (batch_size, num_priors, num_classes)
        MAX_RETURNS = 20
        MAX_CANDIDATES_PER_LAYER = 20
        classes_num = conf_data[0].shape[-1]-1
        conf_preds = [tf.transpose(a=layer_conf, perm=[0, 2, 1]) for layer_conf in
                      conf_data]  # (batch_size, num_classes, num_priors)
        outputs = []
        for i in range(tf.shape(loc_data[0])[0]):
            loc = [loc_e[i, ...] for loc_e in loc_data]
            conf = [conf_e[i, ...] for conf_e in conf_preds]
            if from_logits:
                conf = [tf.math.sigmoid(layer_conf) for layer_conf in conf]
            class_selections: List[List[tf.Tensor]] = [[] for j in range(classes_num)]
            for l_loc, l_conf, l_prior in zip(loc, conf, prior_data):
                object_conf = l_conf[0, ...]
                if classes_num > 1:
                    l_conf = l_conf[1:, ...]*object_conf
                else:
                    l_conf = object_conf[np.newaxis, ...]
                mask = object_conf > self.conf_thresh
                classes = tf.argmax(l_conf, axis=0)
                max_scores = tf.reduce_max(l_conf, axis=0)
                # mask = max_scores > self.conf_thresh
                non_zero_indices = tf.where(mask)[:, 0]
                if len(non_zero_indices) != 0:
                    scores_masked = max_scores[mask]
                    if len(scores_masked) > self.max_bb_per_layer:
                        best_scores, best_indices = tf.math.top_k(scores_masked, k=self.max_bb_per_layer)
                    else:
                        best_scores = scores_masked
                        best_indices = np.arange(len(scores_masked))
                    topk_indices = tf.gather(non_zero_indices, best_indices)
                    selected_loc = tf.gather(l_loc, topk_indices, 1)
                    selected_scores = best_scores
                    if not decoded:
                        selected_prior = tf.gather(l_prior, topk_indices, 1)
                        selected_decoded = decode_bboxes(selected_loc.numpy(), selected_prior.numpy())  # (num_priors, 4)  (xmin, ymin, xmax, ymax) - THIS WORKS
                    else:
                        selected_decoded = xywh_to_xyxy_format(selected_loc.numpy())
                    selected_classes = tf.gather(classes, topk_indices)
                    for k in range(len(selected_classes)):
                        class_selections[selected_classes[k]].append(
                            (selected_scores[k], *selected_decoded[k, :], selected_classes[k]))
            final_preds = []
            for i in range(classes_num):
                if len(class_selections[i]) > 0:
                    np_selection: NDArray[np.float32] = np.array(class_selections[i])
                    boxes = np_selection[:, 1:5]
                    scores = np_selection[:, 0]
                    selected_indices = tf.image.non_max_suppression(boxes=boxes,
                                                                    scores=scores,
                                                                    max_output_size=self.top_k,
                                                                    iou_threshold=self.nms_thresh)
                    final_preds.append(np_selection[selected_indices, :])
            if any(class_selections):
                # choose best MAX_RETURNS/#detected class
                chosen_list = []
                remainder_list = []
                max_per_class = max(int(self.max_bb/len(final_preds)), 1)
                for l in range(len(final_preds)):
                    chosen_list.append(final_preds[l][:max_per_class, ...])
                    remains = final_preds[l][max_per_class:, ...]
                    if remains.shape[0] > 0:
                        remainder_list.append(remains)
                predictions: NDArray[np.float32] = np.concatenate(chosen_list, axis=0)
                predictions = predictions[:self.max_bb, ...]                       # this selects top 20 objects
                matched_obj = predictions.shape[0]
                missing_amount = self.max_bb-matched_obj
                if len(remainder_list) > 0 and predictions.shape[0] < self.max_bb:
                    remainder_np: NDArray[np.float32] = np.concatenate(remainder_list, axis=0)
                    if remainder_np.shape[0] > missing_amount:
                        top_indices = np.argpartition(remainder_np[:, 0], -missing_amount)[-missing_amount:]
                        remainder_np = remainder_np[top_indices, ...]
                    predictions = np.concatenate([predictions, remainder_np], axis=0)
                outputs.append(predictions)
            else:
                outputs.append(np.empty(0, dtype=float))
        return outputs
