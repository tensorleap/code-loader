import pytest

from code_loader.contract.enums import DatasetMetadataType, LeapDataType
from code_loader.contract.responsedataclasses import DatasetSetup, DatasetInputInstance, DatasetMetadataInstance, \
    DatasetOutputInstance, DatasetIntegParseResult, DatasetTestResultPayload, DatasetPreprocess, VisualizerInstance


@pytest.fixture
def no_cloud_dataset_loader_expected_result() -> DatasetIntegParseResult:
    expected_setup = DatasetSetup(
        inputs=[
            DatasetInputInstance(name='normal_input_subset_1_10', shape=[1])
        ],
        metadata=[
            DatasetMetadataInstance(name='x', type=DatasetMetadataType.int),
            DatasetMetadataInstance(name='y', type=DatasetMetadataType.string)],
        outputs=[
            DatasetOutputInstance(name='output_times_20', shape=[1])
        ],
        preprocess=DatasetPreprocess(training_length=4, validation_length=2, test_length=1),
        visualizers=[
            VisualizerInstance(name='Image', type=LeapDataType.Image, arg_names=['data']),
            VisualizerInstance(name='Graph', type=LeapDataType.Graph, arg_names=['data']),
            VisualizerInstance(name='RawData', type=LeapDataType.Text, arg_names=['data']),
            VisualizerInstance(name='HorizontalBar', type=LeapDataType.HorizontalBar, arg_names=['data']),
            VisualizerInstance(name='Text', type=LeapDataType.Text, arg_names=['data']),
            VisualizerInstance(name='ImageMask', type=LeapDataType.ImageMask, arg_names=['mask', 'image']),
            VisualizerInstance(name='TextMask', type=LeapDataType.TextMask, arg_names=['mask', 'text_data'])
        ],
        prediction_types=[],
        custom_loss_names=[]
    )

    expected_payloads = [
        DatasetTestResultPayload(name='preprocess', display={
            'training': '',
            'validation': '',
            'test': ''}, is_passed=True, shape=None),
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
                                              general_error=None, is_valid_for_model=False)

    return expected_result
