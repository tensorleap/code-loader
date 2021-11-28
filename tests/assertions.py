from typing import Dict, List

import numpy as np  # type: ignore

from code_loader import dataset_binder
from grappa import should  # type: ignore

from code_loader.contract.datasetclasses import SubsetResponse, DatasetSample
from code_loader.datasetloader import DatasetLoader


def assert_dataset_binder_is_valid():
    setup_container = dataset_binder.setup_container

    len(setup_container.subsets) | should.be.higher.than(0)
    len(setup_container.inputs) | should.be.higher.than(0)
    len(setup_container.ground_truths) | should.be.higher.than(0)
    len(setup_container.metadata) | should.be.higher.than(0)


def assert_subsets_is_valid(subsets: Dict[str, List[SubsetResponse]]):
    for subset_name, list_subset_response in subsets.items():
        subset_name | should.be.type(str)
        for subset_response in list_subset_response:
            subset_response | should.be.type(SubsetResponse)


def assert_encoder_is_valid(encoder_result: Dict[str, np.ndarray]):
    for encoder_name, encoder_data in encoder_result.items():
        encoder_name | should.be.type(str)
        type(encoder_data) | should.be.type(type(np.ndarray))


def assert_sample_is_valid(sample: DatasetSample):
    sample | should.be.type(DatasetSample)
    assert_encoder_is_valid(sample.inputs)
    assert_encoder_is_valid(sample.gt)
    assert_encoder_is_valid(sample.metadata)


def assert_secret_exists(dataset_loader: DatasetLoader):
    dataset_loader.global_variables | should.have.key('SECRET').that.should.be.type(str)
