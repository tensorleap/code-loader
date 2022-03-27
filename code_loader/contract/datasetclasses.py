from typing import Any, Callable, List, Optional, Dict, Union
from typing_extensions import Protocol

import numpy as np  # type: ignore
from dataclasses import dataclass, field

from code_loader.contract.decoder_classes import LeapImage, LeapText, LeapNumeric, LeapGraph, LeapHorizontalBar, \
    LeapTextMask, LeapImageMask
from code_loader.contract.enums import DataStateType, DatasetMetadataType, \
    DataStateEnum, LeapDataType


@dataclass
class PreprocessResponse:
    length: int
    data: Any


SectionCallableInterface = Callable[[int, PreprocessResponse], np.ndarray]


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
    heatmap_function: Optional[Callable[..., np.array]] = None


@dataclass
class DatasetBaseHandler:
    name: str
    function: SectionCallableInterface


@dataclass
class InputHandler(DatasetBaseHandler):
    shape: Optional[List[int]] = None


@dataclass
class ConnectionInstance:
    decoder_name: str
    encoder_names: List[str]


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


@dataclass
class DatasetSample:
    inputs: Dict[str, np.ndarray]
    gt: Dict[str, np.ndarray]
    metadata: Dict[str, np.ndarray]
    index: int
    state: DataStateEnum
