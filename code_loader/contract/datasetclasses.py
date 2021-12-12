from enum import Enum
from typing import Any, Callable, List, Optional, Dict, Union

import numpy as np  # type: ignore
from dataclasses import dataclass, field

from code_loader.contract.enums import DataStateType, DatasetInputType, DatasetOutputType, DatasetMetadataType


@dataclass
class SubsetResponse:
    length: int
    data: Any


SectionCallableInterface = Callable[[int, SubsetResponse], np.ndarray]


@dataclass
class SubsetHandler:
    function: Callable[[], List[SubsetResponse]]
    name: str
    data_length: Dict[DataStateType, int] = field(default_factory=dict)


@dataclass
class DatasetBaseHandler:
    name: str
    function: SectionCallableInterface
    subset_name: str


@dataclass
class InputHandler(DatasetBaseHandler):
    type: DatasetInputType
    shape: Optional[List[int]]


@dataclass
class GroundTruthHandler(DatasetBaseHandler):
    type: DatasetOutputType
    labels: Optional[List[str]]
    masked_input: Optional[str]
    shape: Optional[List[int]]


@dataclass
class MetadataHandler(DatasetBaseHandler):
    type: DatasetMetadataType


@dataclass
class DatasetIntegrationSetup:
    subsets: List[SubsetHandler] = field(default_factory=list)
    inputs: List[InputHandler] = field(default_factory=list)
    ground_truths: List[GroundTruthHandler] = field(default_factory=list)
    metadata: List[MetadataHandler] = field(default_factory=list)


class SectionEnum(Enum):
    inputs = "inputs"
    ground_truths = "ground_truths"
    metadata = "metadata"
    sample_identity = "sample_identity"
    subsets = "subsets"


DatasetSample = Dict[str, Dict[str, Union[np.ndarray, int, str]]]
