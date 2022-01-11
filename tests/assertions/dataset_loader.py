from typing import Dict, Any

import numpy as np  # type: ignore

from code_loader import dataset_binder
from grappa import should  # type: ignore

from code_loader.contract.datasetclasses import DatasetSample, PreprocessHandler


def assert_dataset_binder_is_valid() -> None:
    setup_container = dataset_binder.setup_container

    setup_container.preprocess | should.be.type(PreprocessHandler)
    len(setup_container.inputs) | should.be.higher.than(0)
    len(setup_container.ground_truths) | should.be.higher.than(0)
    len(setup_container.metadata) | should.be.higher.than(0)
    len(setup_container.connections) | should.be.higher.than(0)


def assert_encoder_is_valid(encoder_result: Dict[str, np.ndarray]) -> None:
    for encoder_name, encoder_data in encoder_result.items():
        encoder_name | should.be.type(str)
        type(encoder_data) | should.be.type(type(np.ndarray))

def assert_metadata_encoder_is_valid(encoder_result: Dict[str, Union[str, int, bool, float]]) -> None:
    for encoder_name, encoder_data in encoder_result.items():
        encoder_name | should.be.type(str)
        type(encoder_data) | should.be.type(type(np.ndarray))


def assert_sample_is_valid(sample: DatasetSample) -> None:
    sample | should.be.type(DatasetSample)
    assert_encoder_is_valid(sample.inputs)
    assert_encoder_is_valid(sample.gt)
    assert_metadata_encoder_is_valid(sample.metadata)


def assert_word_to_index_in_cache_container(expected_key: str, expected_value: Any) -> None:
    dataset_binder.cache_container | should.have.key("word_to_index")
    dataset_binder.cache_container["word_to_index"][expected_key] | should.be.equal(expected_value)


def assert_input_has_value(actual_input: Any, expected_input: Any) -> None:
    actual_input | should.be.equal(expected_input)
