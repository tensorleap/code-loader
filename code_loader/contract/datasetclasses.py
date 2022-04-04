from typing import Any, Callable, List, Optional, Dict, Union

import numpy as np
import numpy.typing as npt
from dataclasses import dataclass, field

from code_loader.contract.decoder_classes import LeapImage, LeapText, LeapNumeric, LeapGraph, LeapHorizontalBar, \
    LeapTextMask, LeapImageMask
from code_loader.contract.enums import DataStateType, DatasetMetadataType, \
    DataStateEnum, LeapDataType
from code_loader.contract.responsedataclasses import PredictionTypeInstance


@dataclass
class PreprocessResponse:
    length: int
    data: Any


SectionCallableInterface = Callable[[int, PreprocessResponse], npt.NDArray[np.float32]]


@dataclass
class PreprocessHandler:
    function: Callable[[], List[PreprocessResponse]]
    data_length: Dict[DataStateType, int] = field(default_factory=dict)


DecoderCallableInterface = Union[
    Callable[..., LeapImage],
    Callable[..., LeapNumeric],
    Callable[..., LeapText],
    Callable[..., LeapGraph],
    Callable[..., LeapHorizontalBar],
    Callable[..., LeapImageMask],
    Callable[..., LeapTextMask],
]

DecoderCallableReturnType = Union[LeapImage, LeapNumeric, LeapText,
                                  LeapGraph, LeapHorizontalBar, LeapImageMask, LeapTextMask]


@dataclass
class DecoderHandler:
    name: str
    function: DecoderCallableInterface
    type: LeapDataType
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
class MetadataHandler(DatasetBaseHandler):
    type: DatasetMetadataType


@dataclass
class DatasetIntegrationSetup:
    preprocess: Optional[PreprocessHandler] = None
    decoders: List[DecoderHandler] = field(default_factory=list)
    inputs: List[InputHandler] = field(default_factory=list)
    ground_truths: List[GroundTruthHandler] = field(default_factory=list)
    metadata: List[MetadataHandler] = field(default_factory=list)
    prediction_types: List[PredictionTypeInstance] = field(default_factory=list)


@dataclass
class DatasetSample:
    inputs: Dict[str, npt.NDArray[np.float32]]
    gt: Dict[str, npt.NDArray[np.float32]]
    metadata: Dict[str, npt.NDArray[np.float32]]
    index: int
    state: DataStateEnum
