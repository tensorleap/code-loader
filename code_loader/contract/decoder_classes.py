from typing import List

import numpy as np  # type: ignore
from dataclasses import dataclass

from code_loader.contract.enums import LeapDataType


@dataclass
class LeapImage:
    data: np.array
    type: LeapDataType = LeapDataType.Image


@dataclass
class LeapNumeric:
    data: np.array
    type: LeapDataType = LeapDataType.Numeric


@dataclass
class LeapGraph:
    data: np.array
    type: LeapDataType = LeapDataType.Numeric


@dataclass
class LeapText:
    data: List[str]
    type: LeapDataType = LeapDataType.Text
