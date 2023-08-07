from typing import List

import numpy as np
import numpy.typing as npt
import tensorflow as tf  # type: ignore

from code_loader import leap_binder
from code_loader.contract.datasetclasses import PreprocessResponse, ConfusionMatrixElement
from code_loader.contract.enums import DatasetMetadataType, LeapDataType, ConfusionMatrixValue
from code_loader.contract.visualizer_classes import LeapText


def get_length(data):
    if data is None:
        length = None
    elif type(data) is dict and 'length' in data:
        length = data['length']
    elif type(data) is not dict:
        length = len(data)
    else:
        length = None

    return length


def prepare_data() -> List[PreprocessResponse]:
    a = [0] * 4
    b = [0] * 2
    c = [0]
    return [PreprocessResponse(length=get_length(a), data=np.array(a)),
            PreprocessResponse(length=get_length(b), data=np.array(b)),
            PreprocessResponse(length=get_length(c), data=np.array(c))]


def input_normal_input_subset_1_10(idx, samples):
    samples = samples.data
    batch_x = samples[idx: idx + 1]
    batch_images = []
    for x in batch_x:
        batch_images.append(x)
    return batch_images[0]


def ground_truth_output_times_20(idx, samples):
    samples = samples.data
    batch_x = samples[idx: idx + 1]
    batch_images = []
    for x in batch_x:
        batch_images.append(x * 20)
    return batch_images[0]


def metadata_x(idx, samples):
    samples = samples.data
    batch_x = samples[idx: idx + 1]
    batch_metadata = []
    for _ in batch_x:
        batch_metadata.append(0)
    return batch_metadata[0]


def metadata_y(idx, samples):
    samples = samples.data
    batch_x = samples[idx: idx + 1]
    batch_metadata = []
    for _ in batch_x:
        batch_metadata.append("fake_string")
    return batch_metadata[0]


def raw_custom_visualizer_func(data: npt.NDArray) -> LeapText:
    return LeapText([str(data)])


leap_binder.set_visualizer(function=raw_custom_visualizer_func, name='stub_visualizer',
                           visualizer_type=LeapDataType.Text)

leap_binder.set_preprocess(function=prepare_data)

leap_binder.set_input(function=input_normal_input_subset_1_10, name='normal_input_subset_1_10')

leap_binder.set_ground_truth(function=ground_truth_output_times_20, name='output_times_20')

leap_binder.set_metadata(function=metadata_x, name='x')

leap_binder.set_metadata(function=metadata_y, name='y')


def custom_metric(pred, gt):
    return pred - gt


leap_binder.add_custom_metric(custom_metric, "custom_metric")

leap_binder.add_prediction(name='pred_type1', labels=['yes', 'no'])


def custom_confusion_metric(gt_one_hot_encoding: tf.Tensor, pred_probabilities: tf.Tensor) -> List[
    List[ConfusionMatrixElement]]:
    num_labels = pred_probabilities.shape[-1]
    labels = [str(i) for i in range(num_labels)]
    if len(labels) == 1:
        labels = ['0', '1']
        gt_one_hot_encoding = tf.concat([1 - gt_one_hot_encoding, gt_one_hot_encoding], axis=1)
        pred_probabilities = tf.concat([1 - pred_probabilities, pred_probabilities], axis=1)

    ret = []
    for batch_i in range(gt_one_hot_encoding.shape[0]):
        one_hot_vec = list(gt_one_hot_encoding[batch_i])
        pred_vec = list(pred_probabilities[batch_i])
        confusion_matrix_elements = []
        for i, label in enumerate(labels):
            expected_outcome = ConfusionMatrixValue.Positive if int(
                one_hot_vec[i]) == 1 else ConfusionMatrixValue.Negative
            cm_element = ConfusionMatrixElement(label, expected_outcome, float(pred_vec[i]))
            confusion_matrix_elements.append(cm_element)
        ret.append(confusion_matrix_elements)
    return ret


leap_binder.add_custom_metric(custom_confusion_metric, "custom_confusion_metric")
