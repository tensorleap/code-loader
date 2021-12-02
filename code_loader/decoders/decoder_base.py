from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class DecoderBase(ABC):
    def __init__(self, name: Optional[str] = None):
        self.name = name
        if self.name is None:
            self.name = self.__class__.__name__

    @abstractmethod
    def decode(self, data: np.array) -> np.array:
        pass

    @abstractmethod
    def decode_heatmap(self, normalized_data: np.array) -> np.array:
        pass
