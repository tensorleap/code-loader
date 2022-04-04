from typing import List, Optional, Dict

from dataclasses import dataclass, field

from code_loader.contract.enums import DatasetMetadataType, LeapDataType, Metric


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


@dataclass
class DatasetMetadataInstance(DatasetBaseSectionInstance):
    type: DatasetMetadataType


@dataclass
class DatasetOutputInstance(DatasetBaseSectionInstance):
    shape: List[int]


@dataclass
class DecoderInstance:
    name: str
    type: LeapDataType


@dataclass
class PredictionTypeInstance:
    name: str
    labels: List[str]
    metrics: List[Metric]


@dataclass
class DatasetSetup:
    preprocess: DatasetPreprocess
    inputs: List[DatasetInputInstance]
    metadata: List[DatasetMetadataInstance]
    outputs: List[DatasetOutputInstance]
    decoders: List[DecoderInstance]
    prediction_types: List[PredictionTypeInstance]


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
