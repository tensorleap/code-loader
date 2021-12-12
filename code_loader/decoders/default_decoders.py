from enum import Enum

import numpy as np  # type: ignore

from code_loader.contract.decoder_classes import LeapImage, LeapNumeric, LeapGraph
from code_loader.utils import rescale_min_max


class DefaultDecoder(Enum):
    Image = 'Image'
    Graph = 'Graph'
    Numeric = 'Numeric'


def default_image_decoder(data: np.array) -> LeapImage:
    rescaled_data = rescale_min_max(data)
    return LeapImage(rescaled_data)


def default_graph_decoder(data: np.array) -> LeapGraph:
    return LeapGraph(data)


def default_numeric_decoder(data: np.array) -> LeapNumeric:
    return LeapNumeric(data)
