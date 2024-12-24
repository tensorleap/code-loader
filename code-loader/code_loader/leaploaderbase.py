# mypy: ignore-errors

from abc import abstractmethod

from typing import Dict, List, Union, Type, Optional

import numpy as np
import numpy.typing as npt

from code_loader.contract.datasetclasses import DatasetSample, LeapData, \
    PredictionTypeHandler, CustomLayerHandler, VisualizerHandlerData, MetricHandlerData, MetricCallableReturnType, \
    CustomLossHandlerData
from code_loader.contract.enums import DataStateEnum
from code_loader.contract.responsedataclasses import DatasetIntegParseResult, DatasetTestResultPayload, \
    DatasetSetup, ModelSetup


class LeapLoaderBase:
    def __init__(self, code_path: str, code_entry_name: str):
        self.code_entry_name = code_entry_name
        self.code_path = code_path

    @abstractmethod
    def metric_by_name(self) -> Dict[str, MetricHandlerData]:
        pass

    @abstractmethod
    def visualizer_by_name(self) -> Dict[str, VisualizerHandlerData]:
        pass

    @abstractmethod
    def custom_loss_by_name(self) -> Dict[str, CustomLossHandlerData]:
        pass

    @abstractmethod
    def custom_layers(self) -> Dict[str, CustomLayerHandler]:
        pass

    @abstractmethod
    def prediction_type_by_name(self) -> Dict[str, PredictionTypeHandler]:
        pass

    @abstractmethod
    def get_sample(self, state: DataStateEnum, sample_id: Union[int, str]) -> DatasetSample:
        pass

    @abstractmethod
    def check_dataset(self) -> DatasetIntegParseResult:
        pass

    @abstractmethod
    def run_visualizer(self, visualizer_name: str, input_tensors_by_arg_name: Dict[str, npt.NDArray[np.float32]]) -> LeapData:
        pass

    @abstractmethod
    def run_metric(self, metric_name: str,
                   input_tensors_by_arg_name: Dict[str, npt.NDArray[np.float32]]) -> MetricCallableReturnType:
        pass

    @abstractmethod
    def run_custom_loss(self, custom_loss_name: str,
                        input_tensors_by_arg_name: Dict[str, npt.NDArray[np.float32]]):
        pass

    @abstractmethod
    def run_heatmap_visualizer(self, visualizer_name: str, input_tensors_by_arg_name: Dict[str, npt.NDArray[np.float32]]
                               ) -> npt.NDArray[np.float32]:
        pass

    @abstractmethod
    def get_dataset_setup_response(self, handlers_test_payloads: List[DatasetTestResultPayload]) -> DatasetSetup:
        pass

    @abstractmethod
    def get_model_setup_response(self) -> ModelSetup:
        pass

    @abstractmethod
    def get_preprocess_sample_ids(
            self, update_unlabeled_preprocess=False) -> Dict[DataStateEnum, Union[List[int], List[str]]]:
        pass

    @abstractmethod
    def get_sample_id_type(self) -> Type:
        pass

    @abstractmethod
    def get_heatmap_visualizer_raw_vis_input_arg_name(self, visualizer_name: str) -> Optional[str]:
        pass

    def is_custom_latent_space(self) -> bool:
        if not self.code_entry_name or not self.code_path:
            return False
        custom_layers = self.custom_layers()
        return any(layer.use_custom_latent_space for layer in custom_layers.values())
