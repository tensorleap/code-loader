import pytest

from code_loader.contract.enums import DatasetMetadataType, LeapDataType
from code_loader.contract.responsedataclasses import DatasetSetup, DatasetInputInstance, DatasetMetadataInstance, \
    DatasetOutputInstance, DatasetIntegParseResult, DatasetTestResultPayload, DatasetPreprocess, VisualizerInstance, \
    PredictionTypeInstance, MetricInstance

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
    visualizers=[
        VisualizerInstance(name='Image', type=LeapDataType.Image, arg_names=['data']),
        VisualizerInstance(name='Graph', type=LeapDataType.Graph, arg_names=['data']),
        VisualizerInstance(name='RawData', type=LeapDataType.Text, arg_names=['data']),
        VisualizerInstance(name='HorizontalBar', type=LeapDataType.HorizontalBar, arg_names=['data']),
        VisualizerInstance(name='Text', type=LeapDataType.Text, arg_names=['data']),
        VisualizerInstance(name='ImageMask', type=LeapDataType.ImageMask, arg_names=['mask', 'image']),
        VisualizerInstance(name='TextMask', type=LeapDataType.TextMask, arg_names=['mask', 'text_data']),
        VisualizerInstance(name='stub_visualizer', type=LeapDataType.Text, arg_names=['data'])

    ],
    metrics=[
        MetricInstance(name='MeanSquaredError', arg_names=['ground_truth', 'prediction']),
        MetricInstance(name='MeanSquaredLogarithmicError', arg_names=['ground_truth', 'prediction']),
        MetricInstance(name='MeanAbsoluteError', arg_names=['ground_truth', 'prediction']),
        MetricInstance(name='MeanAbsolutePercentageError', arg_names=['ground_truth', 'prediction']),
        MetricInstance(name='Accuracy', arg_names=['ground_truth', 'prediction']),
        MetricInstance(name='BinaryAccuracy', arg_names=['ground_truth', 'prediction']),
        MetricInstance(name='ConfusionMatrixClassification', arg_names=['ground_truth', 'prediction']),
        MetricInstance(name='MeanIOU', arg_names=['ground_truth', 'prediction']),
        MetricInstance(name='custom_metric', arg_names=['pred', 'gt']),
        MetricInstance(name='custom_confusion_metric', arg_names=['gt_one_hot_encoding', 'pred_probabilities']),
    ],
    prediction_types=[
        PredictionTypeInstance('pred_type1', ['yes', 'no'])],
    custom_losses=[]
)


@pytest.fixture
def no_cloud_wt_visualizer_dataset_loader_expected_result() -> DatasetIntegParseResult:
    expected_payloads = [
        DatasetTestResultPayload(name='preprocess', display={}, is_passed=True, shape=None),
        DatasetTestResultPayload(name='normal_input_subset_1_10',
                                 display={}, is_passed=True, shape=[1]),
        DatasetTestResultPayload(name='output_times_20',
                                 display={}, is_passed=True, shape=[1]),
        DatasetTestResultPayload(name='x', display={}, is_passed=True,
                                 shape=[1]),
        DatasetTestResultPayload(name='y', display={}, is_passed=True, shape=[1])]

    expected_result = DatasetIntegParseResult(expected_payloads, is_valid=True, setup=expected_setup,
                                              general_error=None, is_valid_for_model=False)

    return expected_result
