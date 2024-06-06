from typing import Union
import numpy as np
import tensorflow as tf


def calculate_iou(
    box1: Union[np.ndarray, tf.Tensor], box2: Union[np.ndarray, tf.Tensor]
) -> np.ndarray:
    """
    Calculate the Intersection over Union (IoU) for pairs of bounding boxes.

    The IoU is a measure of the overlap between two bounding boxes. It is defined as the area of the intersection
    divided by the area of the union of the two boxes. This function works with both NumPy arrays and TensorFlow tensors.

    Args:
        box1 (Union[np.ndarray, tf.Tensor]): An array of shape (N, 4) containing N bounding boxes.
            Each box is represented by four coordinates [x1, y1, x2, y2].
        box2 (Union[np.ndarray, tf.Tensor]): An array of shape (N, 4) containing N bounding boxes.
            Each box is represented by four coordinates [x1, y1, x2, y2].

    Returns:
        np.ndarray: An array of shape (N,) containing the IoU for each pair of bounding boxes.

    Example:
        >>> box1 = np.array([[0, 0, 2, 2], [1, 1, 3, 3]])
        >>> box2 = np.array([[1, 1, 3, 3], [2, 2, 4, 4]])
        >>> iou = calculate_iou(box1, box2)
        >>> print(iou)
        array([0.14, 0. ])
    """
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
