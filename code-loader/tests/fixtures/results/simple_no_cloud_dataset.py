import pytest

from code_loader.contract.enums import DatasetMetadataType, LeapDataType
from code_loader.contract.responsedataclasses import DatasetSetup, DatasetInputInstance, DatasetMetadataInstance, \
    DatasetOutputInstance, DatasetIntegParseResult, DatasetTestResultPayload, DatasetPreprocess, VisualizerInstance, \
    MetricInstance


@pytest.fixture
def no_cloud_dataset_loader_expected_result() -> DatasetIntegParseResult:
    expected_setup = DatasetSetup(
        inputs=[
            DatasetInputInstance(name='normal_input_subset_1_10', shape=[1])
        ],
        metadata=[
            DatasetMetadataInstance(name='z_x', type=DatasetMetadataType.float),
            DatasetMetadataInstance(name='z_y', type=DatasetMetadataType.string)],
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
        metrics=[
            MetricInstance(name='MeanSquaredError', arg_names=['ground_truth', 'prediction']),
            MetricInstance(name='MeanSquaredLogarithmicError', arg_names=['ground_truth', 'prediction']),
            MetricInstance(name='MeanAbsoluteError', arg_names=['ground_truth', 'prediction']),
            MetricInstance(name='MeanAbsolutePercentageError', arg_names=['ground_truth', 'prediction']),
            MetricInstance(name='Accuracy', arg_names=['ground_truth', 'prediction']),
            MetricInstance(name='BinaryAccuracy', arg_names=['ground_truth', 'prediction']),
            MetricInstance(name='ConfusionMatrixClassification', arg_names=['ground_truth', 'prediction']),
            MetricInstance(name='MeanIOU', arg_names=['ground_truth', 'prediction'])
        ],
        prediction_types=[],
        custom_losses=[]
    )

    expected_payloads = [
        DatasetTestResultPayload(name='preprocess', display={}, is_passed=True, shape=None),
        DatasetTestResultPayload(name='normal_input_subset_1_10',
                                 display={}, is_passed=True, shape=[1]),
        DatasetTestResultPayload(name='output_times_20',
                                 display={}, is_passed=True, shape=[1]),
        DatasetTestResultPayload(name='z_x', display={}, is_passed=True,
                                 shape=[1]),
        DatasetTestResultPayload(name='z_y', display={}, is_passed=True, shape=[1])]

    expected_result = DatasetIntegParseResult(expected_payloads, is_valid=True, setup=expected_setup,
                                              general_error=None, is_valid_for_model=False,
                                              print_log="test\n")

    return expected_result
