from typing import List

import numpy as np

from code_loader import dataset_binder
from code_loader.contract.datasetclasses import PreprocessResponse
from code_loader.contract.enums import DatasetMetadataType, DatasetOutputType, DatasetInputType
from code_loader.decoders.simple_decoders import GraphDecoder


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


def input_negative_input_subset_11_20(idx, samples):
    samples = samples.data
    batch_x = samples[idx: idx + 1]
    batch_images = []
    for x in batch_x:
        batch_images.append(x * (-1))
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


dataset_binder.set_preprocess(prepare_data)

graph_decoder = GraphDecoder()
dataset_binder.set_decoder(graph_decoder)

dataset_binder.set_input(input_normal_input_subset_1_10, 'normal_input_subset_1_10',
                         DatasetInputType.Numeric, graph_decoder.name)

dataset_binder.set_input(input_negative_input_subset_11_20, 'negative_input_subset_11_20',
                         DatasetInputType.Numeric, graph_decoder.name)

dataset_binder.set_ground_truth(ground_truth_output_times_20, 'output_times_20',
                                DatasetOutputType.Numeric, graph_decoder.name, labels=None, masked_input=None)

dataset_binder.set_metadata(metadata_x, 'test_subset_1_10', DatasetMetadataType.int, 'x')

dataset_binder.set_metadata(metadata_y, 'test_subset_1_10', DatasetMetadataType.string, 'y')
