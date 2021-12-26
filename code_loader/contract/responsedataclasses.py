from typing import List, Optional, Dict

from dataclasses import dataclass, field

from code_loader.contract.enums import DatasetInputType, DatasetMetadataType, DatasetOutputType


@dataclass
class DatasetPreprocess:
    training_length: int
    validation_length: int
    test_length: Optional[int] = None


@dataclass
class DatasetBaseSectionInstance:
    name: str


@dataclass
class DatasetInputInstance(DatasetBaseSectionInstance):
    shape: List[int]
    type: DatasetInputType
    decoder_name: str


@dataclass
class DatasetMetadataInstance(DatasetBaseSectionInstance):
    type: DatasetMetadataType


@dataclass
class DatasetOutputInstance(DatasetBaseSectionInstance):
    shape: List[int]
    type: DatasetOutputType
    decoder_name: str
    masked_input: Optional[str] = None


@dataclass
class DatasetSetup:
    preprocess: DatasetPreprocess
    inputs: List[DatasetInputInstance]
    metadata: List[DatasetMetadataInstance]
    outputs: List[DatasetOutputInstance]


@dataclass
class DatasetTestResultPayload:
    name: str
    display: Dict[str, str] = field(default_factory=dict)
    is_passed: bool = True
    shape: Optional[List[int]] = None


@dataclass
class DatasetIntegParseResult:
    payloads: List[DatasetTestResultPayload]
    is_valid: bool
    setup: Optional[DatasetSetup]
    general_error: Optional[str]
