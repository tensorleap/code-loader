from typing import Any, Callable, List, Optional, Dict

import numpy as np  # type: ignore
from dataclasses import dataclass, field

from code_loader.contract.enums import DataStateType, DatasetInputType, DatasetOutputType, DatasetMetadataType, \
    DataStateEnum
from code_loader.decoders.decoder_base import DecoderBase


@dataclass
class PreprocessResponse:
    length: int
    data: Any


SectionCallableInterface = Callable[[int, PreprocessResponse], np.ndarray]


@dataclass
class PreprocessHandler:
    function: Callable[[], List[PreprocessResponse]]
    data_length: Dict[DataStateType, int] = field(default_factory=dict)


@dataclass
class DecoderHandler:
    decoder: DecoderBase


@dataclass
class DatasetBaseHandler:
    name: str
    function: SectionCallableInterface


@dataclass
class InputHandler(DatasetBaseHandler):
    type: DatasetInputType
    shape: Optional[List[int]]
    decoder_name: str


@dataclass
class GroundTruthHandler(DatasetBaseHandler):
    type: DatasetOutputType
    labels: Optional[List[str]]
    masked_input: Optional[str]
    shape: Optional[List[int]]
    decoder_name: str


@dataclass
class MetadataHandler(DatasetBaseHandler):
    type: DatasetMetadataType


@dataclass
class DatasetIntegrationSetup:
    preprocess: PreprocessHandler = None
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
