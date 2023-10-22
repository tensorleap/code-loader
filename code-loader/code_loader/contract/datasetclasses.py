from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Dict, Union, Type

import numpy as np
import numpy.typing as npt

from code_loader.contract.enums import DataStateType, DataStateEnum, LeapDataType, ConfusionMatrixValue
from code_loader.contract.visualizer_classes import LeapImage, LeapText, LeapGraph, LeapHorizontalBar, \
    LeapTextMask, LeapImageMask, LeapImageWithBBox


@dataclass
class PreprocessResponse:
    length: int
    data: Any


SectionCallableInterface = Callable[[int, PreprocessResponse], npt.NDArray[np.float32]]

MetadataSectionCallableInterface = Union[
    Callable[[int, PreprocessResponse], int],
    Callable[[int, PreprocessResponse], Dict[str, int]],
    Callable[[int, PreprocessResponse], str],
    Callable[[int, PreprocessResponse], Dict[str, str]],
    Callable[[int, PreprocessResponse], bool],
    Callable[[int, PreprocessResponse], Dict[str, bool]],
    Callable[[int, PreprocessResponse], float],
    Callable[[int, PreprocessResponse], Dict[str, float]]
]


@dataclass
class PreprocessHandler:
    function: Callable[[], List[PreprocessResponse]]
    data_length: Dict[DataStateType, int] = field(default_factory=dict)


@dataclass
class UnlabeledDataPreprocessHandler:
    function: Callable[[], PreprocessResponse]
    data_length: int = 0


VisualizerCallableInterface = Union[
    Callable[..., LeapImage],
    Callable[..., LeapText],
    Callable[..., LeapGraph],
    Callable[..., LeapHorizontalBar],
    Callable[..., LeapImageMask],
    Callable[..., LeapTextMask],
    Callable[..., LeapImageWithBBox]
]

VisualizerCallableReturnType = Union[LeapImage, LeapText, LeapGraph, LeapHorizontalBar,
                                     LeapImageMask, LeapTextMask, LeapImageWithBBox]

CustomCallableInterface = Callable[..., Any]


@dataclass
class ConfusionMatrixElement:
    label: str
    expected_outcome: ConfusionMatrixValue
    predicted_probability: float
    id: str = ''


ConfusionMatrixCallableInterface = Callable[[Any, Any], List[List[ConfusionMatrixElement]]]

CustomCallableInterfaceMultiArgs = Callable[..., Any]
CustomMultipleReturnCallableInterfaceMultiArgs = Callable[..., Dict[str, Any]]
ConfusionMatrixCallableInterfaceMultiArgs = Callable[..., List[List[ConfusionMatrixElement]]]
MetricCallableReturnType = Union[Any, List[List[ConfusionMatrixElement]]]


@dataclass
class CustomLossHandler:
    name: str
    function: CustomCallableInterface
    arg_names: List[str]


@dataclass
class MetricHandler:
    name: str
    function: Union[CustomCallableInterfaceMultiArgs, ConfusionMatrixCallableInterfaceMultiArgs]
    arg_names: List[str]


@dataclass
class VisualizerHandler:
    name: str
    function: VisualizerCallableInterface
    type: LeapDataType
    arg_names: List[str]
    heatmap_function: Optional[Callable[..., npt.NDArray[np.float32]]] = None


@dataclass
class DatasetBaseHandler:
    name: str
    function: SectionCallableInterface


@dataclass
class InputHandler(DatasetBaseHandler):
    shape: Optional[List[int]] = None


@dataclass
class GroundTruthHandler(DatasetBaseHandler):
    shape: Optional[List[int]] = None


@dataclass
class MetadataHandler:
    name: str
    function: MetadataSectionCallableInterface


@dataclass
class PredictionTypeHandler:
    name: str
    labels: List[str]


@dataclass
class CustomLayerHandler:
    name: str
    layer: Type[Any]
    init_arg_names: List[str]
    call_arg_names: List[str]


@dataclass
class DatasetIntegrationSetup:
    preprocess: Optional[PreprocessHandler] = None
    unlabeled_data_preprocess: Optional[UnlabeledDataPreprocessHandler] = None
    visualizers: List[VisualizerHandler] = field(default_factory=list)
    inputs: List[InputHandler] = field(default_factory=list)
    ground_truths: List[GroundTruthHandler] = field(default_factory=list)
    metadata: List[MetadataHandler] = field(default_factory=list)
    prediction_types: List[PredictionTypeHandler] = field(default_factory=list)
    custom_loss_handlers: List[CustomLossHandler] = field(default_factory=list)
    metrics: List[MetricHandler] = field(default_factory=list)
    custom_layers: Dict[str, CustomLayerHandler] = field(default_factory=dict)


@dataclass
class DatasetSample:
    inputs: Dict[str, npt.NDArray[np.float32]]
    gt: Optional[Dict[str, npt.NDArray[np.float32]]]
    metadata: Dict[str, Union[str, int, bool, float]]
    index: int
    state: DataStateEnum
