from typing import Tuple, List
import numpy as np
from numpy.typing import NDArray


class Grid:
    def __init__(self, image_size: Tuple[int, int], feature_maps: Tuple[Tuple[int, int], ...],
                 box_sizes: Tuple[Tuple[float, ...], ...], strides: Tuple[int, ...],
                 offset: int):
        self.image_size = image_size
        self.feature_maps = feature_maps
        self.box_sizes = box_sizes
        self.strides = strides
        self.offset = offset
        self.anchors = self.generate_cell_anchors()

    def generate_cell_anchors(self) -> NDArray[float]:
        """
        This returns anchors, located at (0,0) sized according to to box_sizes.
        :return: np.ndarray of cell_anchors  (len(FEATURE_MAPS), number of anchors, 4) 4: X,Y,W,H
        """
        layer_anchors = []
        for layer_box_sizes in self.box_sizes:
            anchors = []
            for box_size in layer_box_sizes:
                x0, y0, w, h = 0., 0., box_size[0], box_size[1]
                anchors.append([x0, y0, w, h])
            layer_anchors.append(np.array(anchors))
        return np.stack(layer_anchors)

    def _create_grid_offsets(self, size: Tuple[int, int], stride: Tuple[int, int, int, int, int]):
        grid_height, grid_width = size
        shifts_x = np.arange(- self.offset * stride, (grid_width - self.offset) * stride, step=stride, dtype=np.float32)
        shifts_y = np.arange(- self.offset * stride, (grid_height - self.offset) * stride, step=stride,
                             dtype=np.float32)
        shift_x, shift_y = np.meshgrid(shifts_x, shifts_y)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        return shift_x, shift_y

    def generate_anchors(self) -> List[NDArray[float]]:
        """
        Returns:
            list[Tensor]: #featuremap tensors, each is (#locations x #cell_anchors) x 4
        """
        anchors = []
        buffers = self.anchors
        grid_sizes = self.feature_maps
        for size, stride, base_anchors in zip(grid_sizes, self.strides, buffers):
            shift_x, shift_y = self._create_grid_offsets(size, stride)
            shifts = np.stack((shift_x, shift_y, np.zeros_like(shift_x), np.zeros_like(shift_y)), axis=1)
            absolute_anchors = (shifts.reshape(-1, 1, 4) + base_anchors.reshape(1, -1, 4)).reshape(-1, 4)
            normalized_anchors = absolute_anchors / np.array([self.image_size[1], self.image_size[0],
                                                              self.image_size[1], self.image_size[0]])
            normalized_anchors = np.swapaxes(np.swapaxes(normalized_anchors.reshape((*size, len(base_anchors), 4)), 1, 2), 0, 1).reshape(-1,4)
            anchors.append(normalized_anchors)
        return anchors