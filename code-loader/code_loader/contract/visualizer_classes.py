from typing import List, Any, Union

import numpy as np
import numpy.typing as npt
from dataclasses import dataclass

from code_loader.contract.enums import LeapDataType
from code_loader.contract.responsedataclasses import BoundingBox


class LeapValidationError(Exception):
    pass


def validate_type(actual: Any, expected: Any, prefix_message: str = '') -> None:
    if not isinstance(expected, list):
        expected = [expected]
    if actual not in expected:
        if len(expected) == 1:
            raise LeapValidationError(
                f'{prefix_message}.\n'
                f'visualizer returned unexpected type. got {actual}, instead of {expected[0]}')
        else:
            raise LeapValidationError(
                f'{prefix_message}.\n'
                f'visualizer returned unexpected type. got {actual}, allowed is one of {expected}')


@dataclass
class LeapImage:
    """
    Visualizer representing an image for Tensorleap.

    Attributes:
    data (npt.NDArray[np.float32] | npt.NDArray[np.uint8]): The image data.
    type (LeapDataType): The data type, default is LeapDataType.Image.
    """
    data: Union[npt.NDArray[np.float32], npt.NDArray[np.uint8]]
    type: LeapDataType = LeapDataType.Image

    def __post_init__(self) -> None:
        validate_type(self.type, LeapDataType.Image)
        validate_type(type(self.data), np.ndarray)
        validate_type(self.data.dtype, [np.uint8, np.float32])
        validate_type(len(self.data.shape), 3, 'Image must be of shape 3')
        validate_type(self.data.shape[2], [1, 3], 'Image channel must be either 3(rgb) or 1(gray)')


@dataclass
class LeapImageWithBBox:
    """
    Visualizer representing an image with bounding boxes for Tensorleap, used for object detection tasks.

    Attributes:
    data (npt.NDArray[np.float32] | npt.NDArray[np.uint8]): The image data, shaped [H, W, 3] or [H, W, 1].
    bounding_boxes (List[BoundingBox]): List of Tensorleap bounding boxes objects in relative size to image size.
    type (LeapDataType): The data type, default is LeapDataType.ImageWithBBox.
    """
    data: Union[npt.NDArray[np.float32], npt.NDArray[np.uint8]]
    bounding_boxes: List[BoundingBox]
    type: LeapDataType = LeapDataType.ImageWithBBox

    def __post_init__(self) -> None:
        validate_type(self.type, LeapDataType.ImageWithBBox)
        validate_type(type(self.data), np.ndarray)
        validate_type(self.data.dtype, [np.uint8, np.float32])
        validate_type(len(self.data.shape), 3, 'Image must be of shape 3')
        validate_type(self.data.shape[2], [1, 3], 'Image channel must be either 3(rgb) or 1(gray)')


@dataclass
class LeapGraph:
    """
    Visualizer representing a line chart data for Tensorleap.

    Attributes:
    data (npt.NDArray[np.float32]): The array data, shaped [M, N] where M is the number of data points and N is the number of variables.
    type (LeapDataType): The data type, default is LeapDataType.Graph.
    """
    data: npt.NDArray[np.float32]
    type: LeapDataType = LeapDataType.Graph

    def __post_init__(self) -> None:
        validate_type(self.type, LeapDataType.Graph)
        validate_type(type(self.data), np.ndarray)
        validate_type(self.data.dtype, np.float32)
        validate_type(len(self.data.shape), 2, 'Graph must be of shape 2')


@dataclass
class LeapText:
    """
    Visualizer representing text data for Tensorleap.

    Attributes:
    data (List[str]): The text data, consisting of a list of text tokens. If the model requires fixed-length inputs,
    it is recommended to maintain the fixed length, using empty strings ('') instead of padding tokens ('PAD') e.g., ['I', 'ate', 'a', 'banana', '', '', '', ...]
    type (LeapDataType): The data type, default is LeapDataType.Text.
    """
    data: List[str]
    type: LeapDataType = LeapDataType.Text

    def __post_init__(self) -> None:
        validate_type(self.type, LeapDataType.Text)
        validate_type(type(self.data), list)
        for value in self.data:
            validate_type(type(value), str)


@dataclass
class LeapHorizontalBar:
    """
    Visualizer representing horizontal bar data for Tensorleap.
    For example, this can be used to visualize the model's prediction scores in a classification problem.

    Attributes:
    body (npt.NDArray[np.float32]): The data for the bar, shaped [C], where C is the number of data points.
    labels (List[str]): Labels for the horizontal bar; e.g., when visualizing the model's classification output, labels are the class names.
    Length of `body` should match the length of `labels`, C.
    type (LeapDataType): The data type, default is LeapDataType.HorizontalBar.
    """
    body: npt.NDArray[np.float32]
    labels: List[str]
    type: LeapDataType = LeapDataType.HorizontalBar

    def __post_init__(self) -> None:
        validate_type(self.type, LeapDataType.HorizontalBar)
        validate_type(type(self.body), np.ndarray)
        validate_type(self.body.dtype, np.float32)
        validate_type(len(self.body.shape), 1, 'HorizontalBar body must be of shape 1')

        validate_type(type(self.labels), list)
        for label in self.labels:
            validate_type(type(label), str)


@dataclass
class LeapImageMask:
    """
    Visualizer representing an image with a mask for Tensorleap.
    This can be used for tasks such as segmentation, and other applications where it is important to highlight specific regions within an image.

    Attributes:
    mask (npt.NDArray[np.uint8]): The mask data, shaped [H, W].
    image (npt.NDArray[np.float32] | npt.NDArray[np.uint8]): The image data, shaped [H, W, 3] or shaped [H, W, 1].
    labels (List[str]): Labels associated with the mask regions; e.g., class names for segmented objects. The length of `labels` should match the number of unique values in `mask`.
    type (LeapDataType): The data type, default is LeapDataType.ImageMask.
    """
    mask: npt.NDArray[np.uint8]
    image: Union[npt.NDArray[np.float32], npt.NDArray[np.uint8]]
    labels: List[str]
    type: LeapDataType = LeapDataType.ImageMask

    def __post_init__(self) -> None:
        validate_type(self.type, LeapDataType.ImageMask)
        validate_type(type(self.mask), np.ndarray)
        validate_type(self.mask.dtype, np.uint8)
        validate_type(len(self.mask.shape), 2, 'image mask must be of shape 2')
        validate_type(type(self.image), np.ndarray)
        validate_type(self.image.dtype, [np.uint8, np.float32])
        validate_type(len(self.image.shape), 3, 'Image must be of shape 3')
        validate_type(self.image.shape[2], [1, 3], 'Image channel must be either 3(rgb) or 1(gray)')
        validate_type(type(self.labels), list)
        for label in self.labels:
            validate_type(type(label), str)


@dataclass
class LeapTextMask:
    """
    Visualizer representing text data with a mask for Tensorleap.
    This can be used for tasks such as named entity recognition (NER), sentiment analysis, and other applications where it is important to highlight specific tokens or parts of the text.

    Attributes:
    mask (npt.NDArray[np.uint8]): The mask data, shaped [L].
    text (List[str]): The text data, consisting of a list of text tokens, length of L.
    labels (List[str]): Labels associated with the masked tokens; e.g., named entities or sentiment categories. The length of `labels` should match the number of unique values in `mask`.
    type (LeapDataType): The data type, default is LeapDataType.TextMask.
    """
    mask: npt.NDArray[np.uint8]
    text: List[str]
    labels: List[str]
    type: LeapDataType = LeapDataType.TextMask

    def __post_init__(self) -> None:
        validate_type(self.type, LeapDataType.TextMask)
        validate_type(type(self.mask), np.ndarray)
        validate_type(self.mask.dtype, np.uint8)
        validate_type(len(self.mask.shape), 1, 'text mask must be of shape 1')
        validate_type(type(self.text), list)
        for t in self.text:
            validate_type(type(t), str)
        validate_type(type(self.labels), list)
        for label in self.labels:
            validate_type(type(label), str)


@dataclass
class LeapImageWithHeatmap:
    """
    Visualizer representing an image with heatmaps for Tensorleap.
    This can be used for tasks such as highlighting important regions in an image, visualizing attention maps, and other applications where it is important to overlay heatmaps on images.

    Attributes:
    image (npt.NDArray[np.float32]): The image data, shaped [H, W, C], where C is the number of channels.
    heatmaps (npt.NDArray[np.float32]): The heatmap data, shaped [N, H, W], where N is the number of heatmaps.
    labels (List[str]): Labels associated with the heatmaps; e.g., feature names or attention regions. The length of `labels` should match the number of heatmaps, N.
    type (LeapDataType): The data type, default is LeapDataType.ImageWithHeatmap.
    """
    image: npt.NDArray[np.float32]
    heatmaps: npt.NDArray[np.float32]
    labels: List[str]
    type: LeapDataType = LeapDataType.ImageWithHeatmap

    def __post_init__(self) -> None:
        validate_type(self.type, LeapDataType.ImageWithHeatmap)
        validate_type(type(self.heatmaps), np.ndarray)
        validate_type(self.heatmaps.dtype, np.float32)
        validate_type(type(self.image), np.ndarray)
        validate_type(self.image.dtype, np.float32)
        validate_type(type(self.labels), list)
        for label in self.labels:
            validate_type(type(label), str)
        if self.heatmaps.shape[0] != len(self.labels):
            raise LeapValidationError(
                'Number of heatmaps and labels must be equal')


map_leap_data_type_to_visualizer_class = {
    LeapDataType.Image.value: LeapImage,
    LeapDataType.Graph.value: LeapGraph,
    LeapDataType.Text.value: LeapText,
    LeapDataType.HorizontalBar.value: LeapHorizontalBar,
    LeapDataType.ImageMask.value: LeapImageMask,
    LeapDataType.TextMask.value: LeapTextMask,
    LeapDataType.ImageWithBBox.value: LeapImageWithBBox,
    LeapDataType.ImageWithHeatmap.value: LeapImageWithHeatmap
}
