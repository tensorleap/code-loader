from typing import Any, Callable, List, Optional, Dict, Union

import numpy as np
import numpy.typing as npt
from dataclasses import dataclass, field

from code_loader.contract.enums import DataStateType, DatasetInputType, DatasetOutputType, DatasetMetadataType, \
    DataStateEnum


@dataclass
class SubsetResponse:
    length: int
    data: Any


SectionCallableInterface = Callable[[int, SubsetResponse], np.ndarray]
MetadataSectionCallableInterface = Union[
    Callable[[int, SubsetResponse], int],
    Callable[[int, SubsetResponse], str],
    Callable[[int, SubsetResponse], bool],
    Callable[[int, SubsetResponse], float]
]


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
    shape: Optional[List[int]] = None


@dataclass
class GroundTruthHandler(DatasetBaseHandler):
    type: DatasetOutputType
    labels: Optional[List[str]] = None
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
    subsets: List[SubsetHandler] = field(default_factory=list)
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
