import sys
from typing import List
from types import TracebackType
import numpy as np  # type: ignore

from code_loader.contract.datasetclasses import SectionCallableInterface, PreprocessResponse


def to_numpy_return_wrapper(encoder_function: SectionCallableInterface) -> SectionCallableInterface:
    def numpy_encoder_function(idx: int, samples: PreprocessResponse) -> np.ndarray:
        result = encoder_function(idx, samples)
        numpy_result = np.array(result)
        return numpy_result

    return numpy_encoder_function


def get_root_traceback(exc_tb: TracebackType) -> TracebackType:
    return_traceback = exc_tb
    while return_traceback.tb_next is not None:
        return_traceback = return_traceback.tb_next
    return return_traceback


def get_root_exception_line_number() -> int:
    exc_tb = sys.exc_info()[2]
    root_exception_line_number = -1
    if exc_tb is not None:
        root_traceback = get_root_traceback(exc_tb)
        root_exception_line_number = root_traceback.tb_lineno
    return root_exception_line_number


def get_shape(result: np.ndarray) -> List[int]:
    np_shape = result.shape
    # fix single result shape viewing
    if np_shape is ():
        np_shape = (1,)
    shape = list(np_shape)
    return shape


def rescale_min_max(image: np.ndarray) -> np.ndarray:
    image = image.astype('float32')
    image -= image.min()
    image /= (image.max() - image.min() + 1e-5)

    # rescale the values to range between 0 and 255
    image *= 255
    image = image.astype('uint8')

    return image


