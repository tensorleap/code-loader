from enum import Enum

import numpy as np  # type: ignore

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


def default_image_mask_decoder(mask: np.array, image: np.array) -> LeapImageMask:
    n_different_labels = mask.shape[-1]
    labels = [str(i) for i in range(n_different_labels)]

    if hasattr(mask, 'numpy'):
        mask = mask.numpy()
    if hasattr(image, 'numpy'):
        image = image.numpy()
    if len(mask.shape) > 2:
        if mask.shape[-1] == 1:
            mask = np.squeeze(mask, axis=-1)
        else:
            mask = np.argmax(mask, axis=-1)

    return LeapImageMask(mask, image, labels)


def default_text_mask_decoder(mask: np.array, text_data: np.array) -> LeapTextMask:
    n_different_labels = mask.shape[-1]
    labels = [str(i) for i in range(n_different_labels)]

    if hasattr(mask, 'numpy'):
        mask = mask.numpy()
    if len(mask.shape) > 1:
        if mask.shape[-1] == 1:
            mask = np.squeeze(mask, axis=-1)
        else:
            mask = np.argmax(mask, axis=-1)

    return LeapTextMask(mask, text_data, labels)



