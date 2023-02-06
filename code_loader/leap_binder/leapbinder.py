import inspect
from typing import Callable, List, Optional, Dict, Any, Type, Union

import numpy as np
import numpy.typing as npt
import tensorflow as tf  # type: ignore
from typeguard import typechecked

from code_loader.contract.datasetclasses import SectionCallableInterface, InputHandler, \
    GroundTruthHandler, MetadataHandler, DatasetIntegrationSetup, VisualizerHandler, PreprocessResponse, \
    PreprocessHandler, VisualizerCallableInterface, CustomLossHandler, CustomCallableInterface, PredictionTypeHandler, \
    MetadataSectionCallableInterface, UnlabeledDataPreprocessHandler, CustomLayerHandler, MetricHandler, \
    ConfusionMatrixCallableInterface, CustomCallableInterfaceMultiArgs, ConfusionMatrixCallableInterfaceMultiArgs
from code_loader.contract.enums import DatasetMetadataType, LeapDataType, MetricEnum
from code_loader.metrics.default_metrics import metrics_names_to_functions
from code_loader.utils import to_numpy_return_wrapper
from code_loader.visualizers.default_visualizers import DefaultVisualizer, \
    default_graph_visualizer, \
    default_image_visualizer, default_horizontal_bar_visualizer, default_word_visualizer, \
    default_image_mask_visualizer, default_text_mask_visualizer, default_raw_data_visualizer


class LeapBinder:
    def __init__(self) -> None:
        self.setup_container = DatasetIntegrationSetup()
        self.cache_container: Dict[str, Any] = {"word_to_index": {}}
        self._visualizer_names: List[str] = list()
        self._encoder_names: List[str] = list()
        self._extend_with_default_visualizers()
        self._add_default_metrics()

    def _add_default_metrics(self) -> None:
        for metric_name, metric_function in metrics_names_to_functions.items():
            self.add_custom_metric(function=metric_function, name=metric_name)

    def _extend_with_default_visualizers(self) -> None:
        self.set_visualizer(function=default_image_visualizer, name=DefaultVisualizer.Image.value,
                            visualizer_type=LeapDataType.Image)
        self.set_visualizer(function=default_graph_visualizer, name=DefaultVisualizer.Graph.value,
                            visualizer_type=LeapDataType.Graph)
        self.set_visualizer(function=default_raw_data_visualizer, name=DefaultVisualizer.RawData.value,
                            visualizer_type=LeapDataType.Text)
        self.set_visualizer(function=default_horizontal_bar_visualizer, name=DefaultVisualizer.HorizontalBar.value,
                            visualizer_type=LeapDataType.HorizontalBar)
        self.set_visualizer(function=default_word_visualizer, name=DefaultVisualizer.Text.value,
                            visualizer_type=LeapDataType.Text)
        self.set_visualizer(function=default_image_mask_visualizer, name=DefaultVisualizer.ImageMask.value,
                            visualizer_type=LeapDataType.ImageMask)
        self.set_visualizer(function=default_text_mask_visualizer, name=DefaultVisualizer.TextMask.value,
                            visualizer_type=LeapDataType.TextMask)

    @typechecked
    def set_visualizer(self, function: VisualizerCallableInterface,
                       name: str,
                       visualizer_type: LeapDataType,
                       heatmap_visualizer: Optional[Callable[..., npt.NDArray[np.float32]]] = None) -> None:
        arg_names = inspect.getfullargspec(function)[0]
        if heatmap_visualizer:
            if arg_names != inspect.getfullargspec(heatmap_visualizer)[0]:
                raise Exception(
                    f'The argument names of the heatmap visualizer callback must match the visualizer callback '
                    f'{str(arg_names)}')
        self.setup_container.visualizers.append(
            VisualizerHandler(name, function, visualizer_type, arg_names, heatmap_visualizer))
        self._visualizer_names.append(name)

    @typechecked
    def set_preprocess(self, function: Callable[[], List[PreprocessResponse]]) -> None:
        self.setup_container.preprocess = PreprocessHandler(function)

    @typechecked
    def set_unlabeled_data_preprocess(self, function: Callable[[], PreprocessResponse]) -> None:
        self.setup_container.unlabeled_data_preprocess = UnlabeledDataPreprocessHandler(function)

    @typechecked
    def set_input(self, function: SectionCallableInterface, name: str) -> None:
        function = to_numpy_return_wrapper(function)
        self.setup_container.inputs.append(InputHandler(name, function))

        self._encoder_names.append(name)

    @typechecked
    def add_custom_loss(self, function: CustomCallableInterface, name: str) -> None:
        self.setup_container.custom_loss_handlers.append(CustomLossHandler(name, function))

    @typechecked
    def add_custom_metric(self,
                          function: Union[CustomCallableInterfaceMultiArgs, ConfusionMatrixCallableInterfaceMultiArgs],
                          name: str) -> None:
        arg_names = inspect.getfullargspec(function)[0]
        self.setup_container.metrics.append(MetricHandler(name, function, arg_names))

    @typechecked
    def add_prediction(self, name: str, labels: List[str], metrics: Optional[List[MetricEnum]] = None,
                       custom_metrics: Optional[
                           List[Union[CustomCallableInterface, ConfusionMatrixCallableInterface]]] = None) -> None:
        self.setup_container.prediction_types.append(PredictionTypeHandler(name, labels, metrics, custom_metrics))

    @typechecked
    def set_ground_truth(self, function: SectionCallableInterface, name: str) -> None:
        function = to_numpy_return_wrapper(function)
        self.setup_container.ground_truths.append(GroundTruthHandler(name, function))

        self._encoder_names.append(name)

    @typechecked
    def set_metadata(self, function: MetadataSectionCallableInterface, metadata_type: DatasetMetadataType,
                     name: str) -> None:
        self.setup_container.metadata.append(MetadataHandler(name, function, metadata_type))

    @typechecked
    def set_custom_layer(self, custom_layer: Type[tf.keras.layers.Layer], name: str) -> None:
        init_args = inspect.getfullargspec(custom_layer.__init__)[0][1:]
        call_args = inspect.getfullargspec(custom_layer.call)[0][1:]
        self.setup_container.custom_layers[name] = CustomLayerHandler(name, custom_layer, init_args, call_args)
