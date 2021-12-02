from typing import Optional

import numpy as np

from code_loader.decoders.decoder_base import DecoderBase


class ImageDecoder(DecoderBase):
    def __init__(self, name: Optional[str] = None):
        super().__init__(name)

    def decode(self, data: np.array) -> np.array:
        return data

    def decode_heatmap(self, normalized_data: np.array) -> np.array:
        return normalized_data


class GraphDecoder(DecoderBase):
    def __init__(self, name: Optional[str] = None):
        super().__init__(name)

    def decode(self, data: np.array) -> np.array:
        return data

    def decode_heatmap(self, normalized_data: np.array) -> np.array:
        return normalized_data
