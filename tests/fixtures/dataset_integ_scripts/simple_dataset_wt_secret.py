import os
from typing import List

import numpy as np  # type: ignore

from code_loader import leap_binder
from code_loader.contract.datasetclasses import PreprocessResponse
from code_loader.contract.enums import DatasetMetadataType
from code_loader.visualizers.default_visualizers import DefaultVisualizer

SECRET = os.getenv("SECRET")
print(SECRET)


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
    a = [1] * 8
    b = [1] * 2
    c = [1]
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


leap_binder.set_preprocess(function=prepare_data)

leap_binder.set_input(function=input_normal_input_subset_1_10, name='normal_input_subset_1_10')

leap_binder.set_ground_truth(function=ground_truth_output_times_20, name='output_times_20')

leap_binder.set_metadata(function=metadata_x, metadata_type=DatasetMetadataType.int, name='x')

leap_binder.set_metadata(function=metadata_y, metadata_type=DatasetMetadataType.string, name='y')
