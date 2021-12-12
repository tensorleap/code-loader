from typing import Dict, List, Any, Union

import numpy as np  # type: ignore
from grappa import should  # type: ignore

from code_loader import dataset_binder
from code_loader.contract.datasetclasses import SubsetResponse, DatasetSample, SectionEnum


def assert_dataset_binder_is_valid() -> None:
    setup_container = dataset_binder.setup_container

    len(setup_container.subsets) | should.be.higher.than(0)
    len(setup_container.inputs) | should.be.higher.than(0)
    len(setup_container.ground_truths) | should.be.higher.than(0)
    len(setup_container.metadata) | should.be.higher.than(0)


def assert_subsets_is_valid(subsets: Dict[str, List[SubsetResponse]]) -> None:
    for subset_name, list_subset_response in subsets.items():
        subset_name | should.be.type(str)
        for subset_response in list_subset_response:
            subset_response | should.be.type(SubsetResponse)


def assert_encoder_is_valid(encoder_result: Dict[str, np.ndarray]) -> None:
    for encoder_name, encoder_data in encoder_result.items():
        encoder_name | should.be.type(str)
        type(encoder_data) | should.be.type(type(np.ndarray))


def assert_sample_identity(sample_identity: Dict[str, Union[str, int]]):
    sample_identity["index"] | should.be.type(int)
    sample_identity["subset"] | should.be.type(str)
    sample_identity["state"] | should.be.type(str)


def assert_sample_is_valid(sample: DatasetSample):
    assert_encoder_is_valid(sample[SectionEnum.inputs.name])
    assert_encoder_is_valid(sample[SectionEnum.ground_truths.name])
    assert_encoder_is_valid(sample[SectionEnum.metadata.name])
    assert_sample_identity(sample[SectionEnum.sample_identity.name])


def assert_word_to_index_in_cache_container(expected_key: str, expected_value: Any) -> None:
    dataset_binder.cache_container | should.have.key("word_to_index")
    dataset_binder.cache_container["word_to_index"][expected_key] | should.be.equal(expected_value)


def assert_input_has_value(actual_input: Any, expected_input: Any) -> None:
    actual_input | should.be.equal(expected_input)
