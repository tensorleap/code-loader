from typing import Any, Callable, List, Optional, Dict, Union, Type

import numpy as np  # type: ignore
from dataclasses import dataclass, field

from code_loader.contract.decoder_classes import LeapImage, LeapText, LeapNumeric, LeapGraph, LeapHorizontalBar, \
    LeapMask
from code_loader.contract.enums import DataStateType, DatasetInputType, DatasetOutputType, DatasetMetadataType, \
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
    Callable[[np.array], LeapImage],
    Callable[[np.array], LeapNumeric],
    Callable[[np.array], LeapText],
    Callable[[np.array], LeapGraph],
    Callable[[np.array], LeapHorizontalBar],
    Callable[[np.array], LeapMask],
]

DecoderCallableReturnType = Union[LeapImage, LeapNumeric, LeapText, LeapGraph, LeapHorizontalBar, LeapMask]


@dataclass
class DecoderHandler:
    name: str
    function: DecoderCallableInterface
    type: LeapDataType
    heatmap_function: Optional[Callable[[np.array], np.array]] = None


@dataclass
class DatasetBaseHandler:
    name: str
    function: SectionCallableInterface


@dataclass
class InputHandler(DatasetBaseHandler):
    decoder_name: str
    shape: Optional[List[int]] = None


@dataclass
class GroundTruthHandler(DatasetBaseHandler):
    decoder_name: str
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
