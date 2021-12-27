from enum import Enum

import numpy as np  # type: ignore

from code_loader.contract.decoder_classes import LeapImage, LeapNumeric, LeapGraph, LeapHorizontalBar, LeapText, \
    LeapMask
from code_loader.utils import rescale_min_max


class DefaultDecoder(Enum):
    Image = 'Image'
    Graph = 'Graph'
    Numeric = 'Numeric'
    HorizontalBar = 'HorizontalBar'
    Text = 'Text'
    Mask = 'Mask'


def default_image_decoder(data: np.array) -> LeapImage:
    rescaled_data = rescale_min_max(data)
    return LeapImage(rescaled_data)


def default_graph_decoder(data: np.array) -> LeapGraph:
    return LeapGraph(data)


def default_numeric_decoder(data: np.array) -> LeapNumeric:
    return LeapNumeric(data)


def default_horizontal_bar_decoder(data: np.array) -> LeapHorizontalBar:
    if hasattr(data, 'numpy'):
        data = data.numpy()
    if hasattr(data, 'tolist'):
        data = data.tolist()
    labels = [str(index) for index in range(len(data))]
    return LeapHorizontalBar(data, labels)


def default_word_decoder(data: np.array) -> LeapText:
    if hasattr(data, 'numpy'):
        data = data.numpy()
    if hasattr(data, 'tolist'):
        data = data.tolist()
    words = [str(index[0]) if type(index) is list else str(index) for index in data]
    return LeapText(words)


def default_mask_decoder(mask: np.array) -> LeapMask:
    n_different_labels = mask.shape[-1]
    labels = [str(i) for i in range(n_different_labels)]
    return LeapMask(mask, labels)



