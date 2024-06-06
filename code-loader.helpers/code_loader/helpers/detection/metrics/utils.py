from typing import Union
import numpy as np
import tensorflow as tf


# Helper function to calculate IOU
def calculate_iou(
    box1: Union[np.ndarray, tf.Tensor], box2: Union[np.ndarray, tf.Tensor]
) -> np.ndarray:
    x1 = np.maximum(box1[:, 0], box2[:, 0])
    y1 = np.maximum(box1[:, 1], box2[:, 1])
    x2 = np.minimum(box1[:, 2], box2[:, 2])
    y2 = np.minimum(box1[:, 3], box2[:, 3])

    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    area_box1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area_box2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    union = area_box1 + area_box2 - intersection

    iou = intersection / np.maximum(union, 1e-8)
    return iou
