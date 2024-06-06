import tensorflow as tf
from typing import Dict
import numpy as np
from utils import calculate_iou


def detection_metrics(
    gt_bboxes: tf.Tensor,
    pred_bboxes: tf.Tensor,
    label_id_to_name: Dict[int, str],
    background_label: int,
    threshold: float = 0.5,
    class_agnostic: bool = False,
) -> Dict[str, tf.Tensor]:
    # Assuming gt_bboxes in format [n_objects, (x1, y1, x2, y2, label)]
    # Assuming pred_bboxes in format [n_predictions, (x1, y1, x2, y2, confidence, label)]
    # Initialize dictionaries to store counts

    batch_size = len(gt_bboxes)
    results = generate_results_dict(batch_size, label_id_to_name)

    # If there are no predictions, mark all ground truth bounding boxes as false negatives
    if len(pred_bboxes) == 0:
        for i in range(len(gt_bboxes)):
            gt_bboxes = gt_bboxes[i]
            for j in range(len(gt_bboxes)):
                gt_bbox = gt_bboxes[j]
                gt_label = gt_bbox[-1]
                results[f"{label_id_to_name[gt_label]}_fn"][i] += 1
                results["sample_fn"][i] += 1
    else:
        # Iterate over predicted bounding boxes for each batch
        for i in range(len(pred_bboxes)):
            batch_pred_bboxes = pred_bboxes[i]
            gt_bboxes = gt_bboxes[i]
            gt_bboxes = gt_bboxes[gt_bboxes[:, 4] != background_label]

            # Iterate over predicted bounding boxes
            for pred_bbox in batch_pred_bboxes:
                pred_label = int(pred_bbox[-1])
                pred_coords = pred_bbox[:4]

                # Find matching ground truth bounding boxes
                matching_gt_bboxes = gt_bboxes[(gt_bboxes[:, -1] == pred_label)]

                # If there are no matching ground truth bounding boxes, consider it a false positive
                if len(matching_gt_bboxes) == 0:
                    results[f"{label_id_to_name[pred_label]}_fp"][i] += 1
                    results["sample_fp"][i] += 1
                    continue

                # Calculate IOU with all matching ground truth bounding boxes
                iou = calculate_iou(
                    np.expand_dims(pred_coords, 0), matching_gt_bboxes[:, :4]
                )
                max_iou = np.max(iou)
                max_iou_idx = np.argmax(iou)

                # If IOU is above threshold, consider it a true positive
                if max_iou >= threshold:
                    results[f"{label_id_to_name[pred_label]}_tp"][i] += 1
                    results["sample_tp"][i] += 1
                    # Remove matched ground truth bounding box to avoid double counting
                    gt_bboxes = np.delete(gt_bboxes, max_iou_idx, axis=0)
                else:
                    # If IOU is below threshold, consider it a false positive
                    results[f"{label_id_to_name[pred_label]}_fp"][i] += 1
                    results["sample_fp"][i] += 1

            # Any remaining ground truth bounding boxes are false negatives
            for gt_bbox in gt_bboxes:
                gt_label = int(gt_bbox[-1])
                results[f"{label_id_to_name[gt_label]}_fn"][i] += 1
                results["sample_fn"][i] += 1

    if class_agnostic:
        results = {
            "sample_fn": results["sample_fn"],
            "sample_fp": results["sample_fp"],
            "sample_tp": results["sample_tp"],
        }
    results = {k: tf.convert_to_tensor(v, dtype=tf.int32) for k, v in results.items()}
    return results


def generate_results_dict(
    batch_size: int, label_id_to_name: Dict[int, str]
) -> Dict[str, np.ndarray]:
    results = {}
    for class_name in label_id_to_name.values():
        results[f"{class_name}_fn"] = np.zeros((batch_size,), dtype=int)
        results[f"{class_name}_fp"] = np.zeros((batch_size,), dtype=int)
        results[f"{class_name}_tp"] = np.zeros((batch_size,), dtype=int)

    results["sample_fn"] = np.zeros((batch_size,), dtype=int)
    results["sample_fp"] = np.zeros((batch_size,), dtype=int)
    results["sample_tp"] = np.zeros((batch_size,), dtype=int)
    return results
