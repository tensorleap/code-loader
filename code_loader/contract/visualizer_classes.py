from typing import List

import numpy as np
import numpy.typing as npt
from dataclasses import dataclass

from code_loader.contract.enums import LeapDataType


@dataclass
class LeapImage:
    data: npt.NDArray[np.float32]
    type: LeapDataType = LeapDataType.Image


@dataclass
class LeapImageWithBBox:
    data: npt.NDArray[np.float32]
    bbox: npt.NDArray[np.float32]
    type: LeapDataType = LeapDataType.ImageWithBBox


@dataclass
class LeapNumeric:
    data: npt.NDArray[np.float32]
    type: LeapDataType = LeapDataType.Numeric


@dataclass
class LeapGraph:
    data: npt.NDArray[np.float32]
    type: LeapDataType = LeapDataType.Graph


@dataclass
class LeapText:
    data: List[str]
    type: LeapDataType = LeapDataType.Text


@dataclass
class LeapHorizontalBar:
    body: npt.NDArray[np.float32]
    labels: List[str]
    type: LeapDataType = LeapDataType.HorizontalBar


@dataclass
class LeapImageMask:
    mask: npt.NDArray[np.float32]
    image: npt.NDArray[np.float32]
    labels: List[str]
    type: LeapDataType = LeapDataType.ImageMask


@dataclass
class LeapTextMask:
    mask: npt.NDArray[np.float32]
    text_array: npt.NDArray[np.float32]
    labels: List[str]
    type: LeapDataType = LeapDataType.TextMask
