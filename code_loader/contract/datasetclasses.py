from typing import Any, Callable, List, Optional, Dict

import numpy as np
from dataclasses import dataclass, field

from code_loader.contract.enums import DataStateType, DatasetInputType, DatasetOutputType, DatasetMetadataType


@dataclass
class SubsetResponse:
    length: int
    data: Any


SectionCallableInterface = Callable[[int, SubsetResponse], np.ndarray]


@dataclass
class SubsetHandler:
    ratio: float
    function: Callable[[], List[SubsetResponse]]
    name: str
    data_length: Optional[Dict[DataStateType, int]] = field(default_factory=dict)


@dataclass
class DatasetBaseHandler:
    name: str
    function: SectionCallableInterface
    subset: str


@dataclass
class InputHandler(DatasetBaseHandler):
    type: DatasetInputType
    shape: Optional[List[int]]


@dataclass
class GroundTruthHandler(DatasetBaseHandler):
    type: DatasetOutputType
    labels: List[str]
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
