from typing import Any, Callable, List, Optional, Dict, Union

import numpy as np
import numpy.typing as npt
from dataclasses import dataclass, field

from code_loader.contract.decoder_classes import LeapImage, LeapText, LeapNumeric, LeapGraph, LeapHorizontalBar
from code_loader.contract.enums import DataStateType, DatasetInputType, DatasetOutputType, DatasetMetadataType, \
    DataStateEnum


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
]

DecoderCallableReturnType = Union[LeapImage, LeapNumeric, LeapText, LeapGraph, LeapHorizontalBar]


@dataclass
class DecoderHandler:
    name: str
    function: DecoderCallableInterface
    heatmap_function: Optional[Callable[[np.array], np.array]] = None


@dataclass
class DatasetBaseHandler:
    name: str
    function: SectionCallableInterface


@dataclass
class InputHandler(DatasetBaseHandler):
    type: DatasetInputType
    decoder_name: str
    shape: Optional[List[int]] = None


@dataclass
class GroundTruthHandler(DatasetBaseHandler):
    type: DatasetOutputType
    decoder_name: str
    masked_input: Optional[str] = None
    shape: Optional[List[int]] = None


@dataclass
class MetadataHandler:
    name: str
    function: MetadataSectionCallableInterface
    subset_name: str
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
    inputs: Dict[str, npt.NDArray[np.float32]]
    gt: Dict[str, npt.NDArray[np.float32]]
    metadata: Dict[str, Union[str, int, bool, float]]
    index: int
    state: DataStateEnum
