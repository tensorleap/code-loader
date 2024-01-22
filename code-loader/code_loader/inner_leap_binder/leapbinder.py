import inspect
from typing import Callable, List, Optional, Dict, Any, Type, Union

import numpy as np
import numpy.typing as npt

from code_loader.contract.datasetclasses import SectionCallableInterface, InputHandler, \
    GroundTruthHandler, MetadataHandler, DatasetIntegrationSetup, VisualizerHandler, PreprocessResponse, \
    PreprocessHandler, VisualizerCallableInterface, CustomLossHandler, CustomCallableInterface, PredictionTypeHandler, \
    MetadataSectionCallableInterface, UnlabeledDataPreprocessHandler, CustomLayerHandler, MetricHandler, \
    CustomCallableInterfaceMultiArgs, ConfusionMatrixCallableInterfaceMultiArgs, VisualizerCallableReturnType, \
    CustomMultipleReturnCallableInterfaceMultiArgs, DatasetBaseHandler, custom_latent_space_attribute
from code_loader.contract.enums import LeapDataType, DataStateEnum, DataStateType
from code_loader.contract.responsedataclasses import DatasetTestResultPayload
from code_loader.contract.visualizer_classes import map_leap_data_type_to_visualizer_class
from code_loader.utils import to_numpy_return_wrapper, get_shape
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

        if visualizer_type.value not in map_leap_data_type_to_visualizer_class:
            raise Exception(
                f'The visualizer_type is invalid. current visualizer_type: {visualizer_type}, '  # type: ignore[attr-defined]
                f'should be one of : {", ".join([arg.__name__ for arg in VisualizerCallableReturnType.__args__])}')

        func_annotations = function.__annotations__
        if "return" not in func_annotations:
            print(f"Tensorleap Warning: no return type hint for function {function.__name__}. Please configure the "
                  f"right return type. for more info on python type"
                  f" hints: "
                  f"https://docs.python.org/3/library/typing.html")
        else:
            return_type = func_annotations["return"]
            if return_type not in VisualizerCallableReturnType.__args__:  # type: ignore[attr-defined]
                raise Exception(
                    f'The return type of function {function.__name__} is invalid. current return type: {return_type}, '  # type: ignore[attr-defined]
                    f'should be one of : {", ".join([arg.__name__ for arg in VisualizerCallableReturnType.__args__])}')

            expected_return_type = map_leap_data_type_to_visualizer_class[visualizer_type.value]
            if not issubclass(return_type, expected_return_type):
                raise Exception(
                    f'The return type of function {function.__name__} is invalid. current return type: {return_type}, '
                    f'should be {expected_return_type}')

        self.setup_container.visualizers.append(
            VisualizerHandler(name, function, visualizer_type, arg_names, heatmap_visualizer))
        self._visualizer_names.append(name)

    def set_preprocess(self, function: Callable[[], List[PreprocessResponse]]) -> None:
        self.setup_container.preprocess = PreprocessHandler(function)

    def set_unlabeled_data_preprocess(self, function: Callable[[], PreprocessResponse]) -> None:
        self.setup_container.unlabeled_data_preprocess = UnlabeledDataPreprocessHandler(function)

    def set_input(self, function: SectionCallableInterface, name: str) -> None:
        function = to_numpy_return_wrapper(function)
        self.setup_container.inputs.append(InputHandler(name, function))

        self._encoder_names.append(name)

    def add_custom_loss(self, function: CustomCallableInterface, name: str) -> None:
        arg_names = inspect.getfullargspec(function)[0]
        self.setup_container.custom_loss_handlers.append(CustomLossHandler(name, function, arg_names))

    def add_custom_metric(self,
                          function: Union[CustomCallableInterfaceMultiArgs,
                          CustomMultipleReturnCallableInterfaceMultiArgs,
                          ConfusionMatrixCallableInterfaceMultiArgs],
                          name: str) -> None:
        arg_names = inspect.getfullargspec(function)[0]
        self.setup_container.metrics.append(MetricHandler(name, function, arg_names))

    def add_prediction(self, name: str, labels: List[str], channel_dim: int = -1) -> None:
        self.setup_container.prediction_types.append(PredictionTypeHandler(name, labels, channel_dim))

    def set_ground_truth(self, function: SectionCallableInterface, name: str) -> None:
        function = to_numpy_return_wrapper(function)
        self.setup_container.ground_truths.append(GroundTruthHandler(name, function))

        self._encoder_names.append(name)

    def set_metadata(self, function: MetadataSectionCallableInterface, name: str) -> None:
        self.setup_container.metadata.append(MetadataHandler(name, function))

    def set_custom_layer(self, custom_layer: Type[Any], name: str, inspect_layer: bool = False,
                         kernel_index: Optional[int] = None, use_custom_latent_space: bool = False) -> None:
        if inspect_layer and kernel_index is not None:
            custom_layer.kernel_index = kernel_index

        if use_custom_latent_space and not hasattr(custom_layer, custom_latent_space_attribute):
            raise Exception(f"{custom_latent_space_attribute} function has not been set for custom layer: {custom_layer.__name__}")

        init_args = inspect.getfullargspec(custom_layer.__init__)[0][1:]
        call_args = inspect.getfullargspec(custom_layer.call)[0][1:]
        self.setup_container.custom_layers[name] = CustomLayerHandler(name, custom_layer, init_args, call_args,
                                                                      use_custom_latent_space=use_custom_latent_space)

    def check_preprocess(self, preprocess_result: Dict[DataStateEnum, PreprocessResponse]) -> None:
        preprocess_handler = self.setup_container.preprocess
        unlabeled_preprocess_handler = self.setup_container.unlabeled_data_preprocess

        if preprocess_handler is None:
            raise Exception('None preprocess_handler')

        for state, preprocess_response in preprocess_result.items():
            if preprocess_response.length is None or preprocess_response.length <= 0:
                raise Exception('Invalid dataset length')
            if unlabeled_preprocess_handler is not None and state == DataStateEnum.unlabeled:
                unlabeled_preprocess_handler.data_length = preprocess_response.length
            else:
                state_type = DataStateType(state.name)
                preprocess_handler.data_length[state_type] = preprocess_response.length

    def get_preprocess_result(self) -> Dict[DataStateEnum, PreprocessResponse]:
        preprocess = self.setup_container.preprocess
        if preprocess is None:
            raise Exception("Please make sure you call the leap_binder.set_preprocess method")
        preprocess_results = preprocess.function()
        preprocess_result_dict = {
            DataStateEnum(i): preprocess_result
            for i, preprocess_result in enumerate(preprocess_results)
        }

        unlabeled_preprocess = self.setup_container.unlabeled_data_preprocess
        if unlabeled_preprocess is not None:
            preprocess_result_dict[DataStateEnum.unlabeled] = unlabeled_preprocess.function()

        return preprocess_result_dict

    def _get_all_dataset_base_handlers(self) -> List[Union[DatasetBaseHandler, MetadataHandler]]:
        all_dataset_base_handlers: List[Union[DatasetBaseHandler, MetadataHandler]] = []
        all_dataset_base_handlers.extend(self.setup_container.inputs)
        all_dataset_base_handlers.extend(self.setup_container.ground_truths)
        all_dataset_base_handlers.extend(self.setup_container.metadata)
        return all_dataset_base_handlers

    @staticmethod
    def check_handler(
            preprocess_response: PreprocessResponse, test_result: List[DatasetTestResultPayload],
            dataset_base_handler: Union[DatasetBaseHandler, MetadataHandler]) -> List[DatasetTestResultPayload]:
        raw_result = dataset_base_handler.function(0, preprocess_response)
        handler_type = 'metadata' if isinstance(dataset_base_handler, MetadataHandler) else None
        if isinstance(dataset_base_handler, MetadataHandler) and isinstance(raw_result, dict):
            metadata_test_result_payloads = [
                DatasetTestResultPayload(f'{dataset_base_handler.name}_{single_metadata_name}')
                for single_metadata_name, single_metadata_result in raw_result.items()
            ]
            for i, (single_metadata_name, single_metadata_result) in enumerate(raw_result.items()):
                metadata_test_result = metadata_test_result_payloads[i]
                assert isinstance(single_metadata_result, (float, int, str, bool))
                result_shape = get_shape(single_metadata_result)
                metadata_test_result.shape = result_shape
                metadata_test_result.raw_result = single_metadata_result
                metadata_test_result.handler_type = handler_type
            test_result = metadata_test_result_payloads
        else:
            assert not isinstance(raw_result, dict)
            result_shape = get_shape(raw_result)
            test_result[0].shape = result_shape
            test_result[0].raw_result = raw_result
            test_result[0].handler_type = handler_type

            # setting shape in setup for all encoders
            if isinstance(dataset_base_handler, (InputHandler, GroundTruthHandler)):
                dataset_base_handler.shape = result_shape
        return test_result

    def check_handlers(self, preprocess_result: Dict[DataStateEnum, PreprocessResponse]) -> None:
        dataset_base_handlers: List[Union[DatasetBaseHandler, MetadataHandler]] = self._get_all_dataset_base_handlers()
        for dataset_base_handler in dataset_base_handlers:
            test_result = [DatasetTestResultPayload(dataset_base_handler.name)]
            for state, preprocess_response in preprocess_result.items():
                if state == DataStateEnum.unlabeled and isinstance(dataset_base_handler, GroundTruthHandler):
                    continue
                self.check_handler(preprocess_response, test_result, dataset_base_handler)

    def check(self) -> None:
        preprocess_result = self.get_preprocess_result()
        self.check_preprocess(preprocess_result)
        self.check_handlers(preprocess_result)
        print("Successful!")
