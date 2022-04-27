from typing import List

import numpy as np  # type: ignore

from code_loader import leap_binder
from code_loader.contract.datasetclasses import PreprocessResponse
from code_loader.contract.enums import DatasetMetadataType
from code_loader.decoders.default_decoders import DefaultDecoder

input_name = 'normal_input_subset_1_10'
word_to_index_value = "mock word_to_index"


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


def subset_test_subset_1_10() -> List[PreprocessResponse]:
    a = [0] * 4
    b = [0] * 2
    c = [0]
    leap_binder.cache_container["word_to_index"][input_name] = word_to_index_value
    return [PreprocessResponse(length=get_length(a), data=np.array(a)),
            PreprocessResponse(length=get_length(b), data=np.array(b)),
            PreprocessResponse(length=get_length(c), data=np.array(c))]


def input_normal_input_subset_1_10(idx, samples):
    samples = samples.data
    batch_x = samples[idx: idx + 1]
    batch_images = []
    for x in batch_x:
        batch_images.append(x)
    return leap_binder.cache_container["word_to_index"][input_name]


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


leap_binder.set_preprocess(subset_test_subset_1_10)

leap_binder.set_input(input_normal_input_subset_1_10, input_name)

leap_binder.set_ground_truth(ground_truth_output_times_20, 'output_times_20')

leap_binder.set_metadata(metadata_x, DatasetMetadataType.int, 'x')

leap_binder.set_metadata(metadata_y, DatasetMetadataType.string, 'y')
