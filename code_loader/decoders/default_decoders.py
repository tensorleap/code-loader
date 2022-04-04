from enum import Enum
from typing import cast, List

import numpy as np
import numpy.typing as npt

from code_loader.contract.decoder_classes import LeapImage, LeapNumeric, LeapGraph, LeapHorizontalBar, LeapText, \
    LeapImageMask, LeapTextMask
from code_loader.utils import rescale_min_max


class DefaultDecoder(Enum):
    Image = 'Image'
    Graph = 'Graph'
    Numeric = 'Numeric'
    HorizontalBar = 'HorizontalBar'
    Text = 'Text'
    ImageMask = 'ImageMask'
    TextMask = 'TextMask'


def default_image_decoder(data: npt.NDArray[np.float32]) -> LeapImage:
    rescaled_data = rescale_min_max(data)
    return LeapImage(rescaled_data)


def default_graph_decoder(data: npt.NDArray[np.float32]) -> LeapGraph:
    return LeapGraph(data)


def default_numeric_decoder(data: npt.NDArray[np.float32]) -> LeapNumeric:
    return LeapNumeric(data)


def default_horizontal_bar_decoder(data: npt.NDArray[np.float32]) -> LeapHorizontalBar:
    labels = [str(index) for index in range(len(data))]
    return LeapHorizontalBar(data, labels)


def default_word_decoder(data: npt.NDArray[np.float32]) -> LeapText:
    if hasattr(data, 'tolist'):
        data = data.tolist()
    words = [str(index[0]) if type(index) is list else str(index) for index in data]
    return LeapText(words)


def default_image_mask_decoder(mask: npt.NDArray[np.float32], image: npt.NDArray[np.float32]) -> LeapImageMask:
    n_different_labels = mask.shape[-1]
    labels = [str(i) for i in range(n_different_labels)]

    if len(mask.shape) > 2:
        if mask.shape[-1] == 1:
            mask = np.squeeze(mask, axis=-1)
        else:
            mask = np.argmax(mask, axis=-1)

    return LeapImageMask(mask, image, labels)


def default_text_mask_decoder(mask: npt.NDArray[np.float32], text_data: npt.NDArray[np.float32]) -> LeapTextMask:
    n_different_labels = mask.shape[-1]
    labels = [str(i) for i in range(n_different_labels)]

    if len(mask.shape) > 1:
        if mask.shape[-1] == 1:
            mask = np.squeeze(mask, axis=-1)
        else:
            mask = np.argmax(mask, axis=-1)

    return LeapTextMask(mask, text_data, labels)



