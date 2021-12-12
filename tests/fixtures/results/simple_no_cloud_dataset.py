import pytest

from code_loader.contract.enums import DatasetInputType, DatasetMetadataType, DatasetOutputType
from code_loader.contract.responsedataclasses import DatasetSetup, DatasetInputInstance, DatasetMetadataInstance, \
    DatasetOutputInstance, DatasetIntegParseResult, DatasetTestResultPayload, DatasetPreprocess


@pytest.fixture
def no_cloud_dataset_loader_expected_result() -> DatasetIntegParseResult:
    expected_setup = DatasetSetup(
        inputs=[
            DatasetInputInstance(name='normal_input_subset_1_10', shape=[1],
                                 type=DatasetInputType.Numeric)],
        metadata=[
            DatasetMetadataInstance(name='x', type=DatasetMetadataType.int),
            DatasetMetadataInstance(name='y', type=DatasetMetadataType.string)],
        outputs=[
            DatasetOutputInstance(name='output_times_20', shape=[1],
                                  type=DatasetOutputType.Numeric, masked_input=None, labels=None)],
        preprocess=DatasetPreprocess(training_length=4, validation_length=2, test_length=1))

    expected_payloads = [
        DatasetTestResultPayload(name='preprocess', display={
            'training': '[0 0 0 0]',
            'validation': '[0 0]',
            'test': '[0]'}, is_passed=True, shape=None),
        DatasetTestResultPayload(name='normal_input_subset_1_10',
                                 display={'training': '0', 'validation': '0',
                                          'test': '0'}, is_passed=True, shape=[1]),
        DatasetTestResultPayload(name='output_times_20',
                                 display={'training': '0', 'validation': '0',
                                          'test': '0'}, is_passed=True, shape=[1]),
        DatasetTestResultPayload(name='x', display={'training': '0', 'validation': '0',
                                                    'test': '0'}, is_passed=True,
                                 shape=[1]),
        DatasetTestResultPayload(name='y', display={
            'training': 'fake_string',
            'validation': 'fake_string',
            'test': 'fake_string'}, is_passed=True, shape=[1])]

    expected_result = DatasetIntegParseResult(expected_payloads, is_valid=True, setup=expected_setup,
                                              general_error=None)

    return expected_result
