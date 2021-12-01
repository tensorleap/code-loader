from typing import List

import numpy as np  # type: ignore

from code_loader import dataset_binder
from code_loader.contract.datasetclasses import SubsetResponse
from code_loader.contract.enums import DatasetInputType, DatasetOutputType, DatasetMetadataType


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


def subset_test_subset_1_10() -> List[SubsetResponse]:
    a = [0] * 4
    b = [0] * 2
    c = [0]
    return [SubsetResponse(length=get_length(a), data=np.array(a)),
            SubsetResponse(length=get_length(b), data=np.array(b)),
            SubsetResponse(length=get_length(c), data=np.array(c))]


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
    batch_metafata = []
    for x in batch_x:
        batch_metafata.append(0)
    return batch_metafata[0]


def metadata_y(idx, samples):
    samples = samples.data
    batch_x = samples[idx: idx + 1]
    batch_metafata = []
    for x in batch_x:
        batch_metafata.append("fake_string")
    return batch_metafata[0]


dataset_binder.set_subset(subset_test_subset_1_10, 'test_subset_1_10')

dataset_binder.set_input(input_normal_input_subset_1_10, 'test_subset_1_10', DatasetInputType.Numeric,
                         'normal_input_subset_1_10')

dataset_binder.set_ground_truth(ground_truth_output_times_20, 'test_subset_1_10', DatasetOutputType.Numeric,
                                'output_times_20',
                                labels=None, masked_input=None)

dataset_binder.set_metadata(metadata_x, 'test_subset_1_10', DatasetMetadataType.int, 'x')

dataset_binder.set_metadata(metadata_y, 'test_subset_1_10', DatasetMetadataType.string, 'y')
