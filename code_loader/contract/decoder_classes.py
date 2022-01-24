from typing import List

import numpy as np  # type: ignore
from dataclasses import dataclass

from code_loader.contract.enums import LeapDataType


@dataclass
class LeapImage:
    data: np.array
    type: LeapDataType = LeapDataType.Image


@dataclass
class LeapImageWithBBox:
    data: np.array
    bbox: np.array
    type: LeapDataType = LeapDataType.ImageWithBBox


@dataclass
class LeapNumeric:
    data: np.array
    type: LeapDataType = LeapDataType.Numeric


@dataclass
class LeapGraph:
    data: np.array
    type: LeapDataType = LeapDataType.Graph


@dataclass
class LeapText:
    data: List[str]
    type: LeapDataType = LeapDataType.Text


@dataclass
class LeapHorizontalBar:
    body: List[float]
    labels: List[str]
    type: LeapDataType = LeapDataType.HorizontalBar


@dataclass
class LeapImageMask:
    mask: np.array
    image: np.array
    labels: List[str]
    type: LeapDataType = LeapDataType.ImageMask


@dataclass
class LeapTextMask:
    mask: np.array
    text_array: np.array
    labels: List[str]
    type: LeapDataType = LeapDataType.TextMask
