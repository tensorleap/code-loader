from enum import Enum
from typing import List, Tuple

import tensorflow as tf  # type: ignore
from keras import backend as K  # type: ignore
from keras.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error  # type: ignore
from keras.metrics import mean_squared_logarithmic_error, categorical_accuracy
from keras import metrics
from tensorflow.python.ops import array_ops, confusion_matrix, math_ops  # type: ignore

from code_loader.contract.datasetclasses import ConfusionMatrixElement
from code_loader.contract.enums import ConfusionMatrixValue


class Metric(Enum):
    MeanSquaredError = 'MeanSquaredError'
    MeanSquaredLogarithmicError = 'MeanSquaredLogarithmicError'
    MeanAbsoluteError = 'MeanAbsoluteError'
    MeanAbsolutePercentageError = 'MeanAbsolutePercentageError'
    Accuracy = 'Accuracy'
    BinaryAccuracy = 'BinaryAccuracy'
    MeanIOU = 'MeanIOU'
    ConfusionMatrixClassification = 'ConfusionMatrixClassification'


def _inner_mean_iou(args: Tuple[tf.Tensor, tf.Tensor, int]) -> tf.Tensor:
    y_true, y_pred, num_labels = args
    if y_pred.shape.ndims > 1:
        y_pred = array_ops.reshape(y_pred, [-1])

    if y_true.shape.ndims > 1:
        y_true = array_ops.reshape(y_true, [-1])

    # Accumulate the prediction to current confusion matrix.
    _dtype = "float32"
    current_cm = confusion_matrix.confusion_matrix(y_true, y_pred, num_labels)
    sum_over_row = math_ops.cast(math_ops.reduce_sum(current_cm, axis=0), dtype=_dtype)
    sum_over_col = math_ops.cast(math_ops.reduce_sum(current_cm, axis=1), dtype=_dtype)
    true_positives = math_ops.cast(array_ops.tensor_diag_part(current_cm), dtype=_dtype)

    # sum_over_row + sum_over_col = 2 * true_positives + false_positives + false_negatives.
    denominator = sum_over_row + sum_over_col - true_positives

    # The mean is only computed over classes that appear in the label or prediction tensor.
    # If the denominator is 0 we need to ignore the class.
    num_valid_entries = math_ops.reduce_sum(math_ops.cast(math_ops.not_equal(denominator, 0), dtype=_dtype))

    iou = math_ops.div_no_nan(true_positives, denominator)
    mean_iou = math_ops.div_no_nan(math_ops.reduce_sum(iou, name='mean_iou'), num_valid_entries)
    mean_iou = tf.expand_dims(mean_iou, 0)
    return mean_iou


def batch_mean_iou(ground_truth: tf.Tensor, prediction: tf.Tensor) -> tf.Tensor:
    num_labels = prediction.shape[-1]
    ground_truth, prediction = argmax_and_fix_gt(ground_truth, prediction)
    mean_iou_result = tf.vectorized_map(_inner_mean_iou, (ground_truth, prediction, num_labels))
    mean_iou_result = tf.transpose(mean_iou_result)
    return mean_iou_result


def reduced_categorical_accuracy(ground_truth: tf.Tensor, prediction: tf.Tensor) -> tf.Tensor:
    per_pixel_accuracy = categorical_accuracy(ground_truth, prediction)
    reduce_axis = tf.range(1, len(per_pixel_accuracy.shape))
    accuracy = tf.reduce_mean(per_pixel_accuracy, axis=reduce_axis)
    return accuracy


def mean_squared_error_dimension_reduced(ground_truth: tf.Tensor, prediction: tf.Tensor) -> tf.Tensor:
    ground_truth, prediction = flatten_non_batch_dims(ground_truth, prediction)
    return mean_squared_error(ground_truth, prediction)


def mean_squared_logarithmic_error_dimension_reduced(ground_truth: tf.Tensor, prediction: tf.Tensor) -> tf.Tensor:
    ground_truth, prediction = flatten_non_batch_dims(ground_truth, prediction)
    return mean_squared_logarithmic_error(ground_truth, prediction)


def mean_absolute_error_dimension_reduced(ground_truth: tf.Tensor, prediction: tf.Tensor) -> tf.Tensor:
    ground_truth, prediction = flatten_non_batch_dims(ground_truth, prediction)
    return mean_absolute_error(ground_truth, prediction)


def binary_accuracy(ground_truth: tf.Tensor, prediction: tf.Tensor) -> tf.Tensor:
    return metrics.binary_accuracy(ground_truth, prediction)


def mean_absolute_percentage_error_dimension_reduced(ground_truth: tf.Tensor, prediction: tf.Tensor) -> tf.Tensor:
    ground_truth, prediction = flatten_non_batch_dims(ground_truth, prediction)
    return mean_absolute_percentage_error(ground_truth, prediction)


def flatten_non_batch_dims(ground_truth: tf.Tensor, prediction: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    batch_size = ground_truth.shape[0]
    ground_truth = K.reshape(ground_truth, (batch_size, -1))
    prediction = K.reshape(prediction, (batch_size, -1))
    return ground_truth, prediction


def argmax_and_fix_gt(ground_truth: tf.Tensor, prediction: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    # checking sparsity and converting all softmax and one-hots to its argmax value
    prediction = tf.argmax(prediction, axis=-1)

    if ground_truth.shape[-1] == 1:
        ground_truth = tf.squeeze(ground_truth, axis=-1)

    if is_gt_to_argmax(prediction, ground_truth):
        ground_truth = tf.argmax(ground_truth, axis=-1)

    return ground_truth, prediction


def is_gt_to_argmax(prediction: tf.Tensor, ground_truth: tf.Tensor) -> bool:
    if len(prediction.shape) < len(ground_truth.shape):
        return True
    return False


def confusion_matrix_classification_metric(ground_truth: tf.Tensor, prediction: tf.Tensor) -> List[
    List[ConfusionMatrixElement]]:
    num_labels = prediction.shape[-1]
    labels = [str(i) for i in range(num_labels)]
    if len(labels) == 1:
        labels = ['0', '1']
        ground_truth = tf.concat([1 - ground_truth, ground_truth], axis=1)
        prediction = tf.concat([1 - prediction, prediction], axis=1)

    ret = []
    for batch_i in range(ground_truth.shape[0]):
        one_hot_vec = list(ground_truth[batch_i])
        pred_vec = list(prediction[batch_i])
        confusion_matrix_elements = []
        for i, label in enumerate(labels):
            expected_outcome = ConfusionMatrixValue.Positive if int(
                one_hot_vec[i]) == 1 else ConfusionMatrixValue.Negative
            cm_element = ConfusionMatrixElement(label, expected_outcome, float(pred_vec[i]))
            confusion_matrix_elements.append(cm_element)
        ret.append(confusion_matrix_elements)
    return ret


metrics_names_to_functions = {
    Metric.MeanSquaredError.name: mean_squared_error_dimension_reduced,
    Metric.MeanSquaredLogarithmicError.name: mean_squared_logarithmic_error_dimension_reduced,
    Metric.MeanAbsoluteError.name: mean_absolute_error_dimension_reduced,
    Metric.MeanAbsolutePercentageError.name: mean_absolute_percentage_error_dimension_reduced,
    Metric.Accuracy.name: reduced_categorical_accuracy,
    Metric.BinaryAccuracy.name: binary_accuracy,
    Metric.ConfusionMatrixClassification.name: confusion_matrix_classification_metric,
    Metric.MeanIOU.name: batch_mean_iou
}
