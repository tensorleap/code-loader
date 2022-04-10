import pytest

from code_loader.contract.enums import DatasetMetadataType, LeapDataType, Metric
from code_loader.contract.responsedataclasses import DatasetSetup, DatasetInputInstance, DatasetMetadataInstance, \
    DatasetOutputInstance, DatasetIntegParseResult, DatasetTestResultPayload, DatasetPreprocess, DecoderInstance, \
    PredictionTypeInstance


@pytest.fixture
def no_cloud_wt_decoder_dataset_loader_expected_result() -> DatasetIntegParseResult:
    expected_setup = DatasetSetup(
        inputs=[
            DatasetInputInstance(name='normal_input_subset_1_10', shape=[1])],
        metadata=[
            DatasetMetadataInstance(name='x', type=DatasetMetadataType.int),
            DatasetMetadataInstance(name='y', type=DatasetMetadataType.string)],
        outputs=[
            DatasetOutputInstance(name='output_times_20', shape=[1])
        ],
        preprocess=DatasetPreprocess(training_length=4, validation_length=2, test_length=1),
        decoders=[
            DecoderInstance(name='Image', type=LeapDataType.Image, arg_names=['data']),
            DecoderInstance(name='Graph', type=LeapDataType.Graph, arg_names=['data']),
            DecoderInstance(name='Numeric', type=LeapDataType.Numeric, arg_names=['data']),
            DecoderInstance(name='HorizontalBar', type=LeapDataType.HorizontalBar, arg_names=['data']),
            DecoderInstance(name='Text', type=LeapDataType.Text, arg_names=['data']),
            DecoderInstance(name='ImageMask', type=LeapDataType.ImageMask, arg_names=['mask', 'image']),
            DecoderInstance(name='TextMask', type=LeapDataType.TextMask, arg_names=['mask', 'text_data']),
            DecoderInstance(name='stub_decoder', type=LeapDataType.Numeric, arg_names=['data'])
        ],
        prediction_types=[
            PredictionTypeInstance('pred_type1', ['yes', 'no'], [Metric.MeanAbsoluteError], ["custom_metric"])],
        custom_loss_names=[]
    )

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
