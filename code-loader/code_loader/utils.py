import sys
from pathlib import Path
from types import TracebackType
from typing import List, Union, Tuple

import numpy as np
import numpy.typing as npt

from code_loader.contract.datasetclasses import SectionCallableInterface, PreprocessResponse


def to_numpy_return_wrapper(encoder_function: SectionCallableInterface) -> SectionCallableInterface:
    def numpy_encoder_function(idx: int, samples: PreprocessResponse) -> npt.NDArray[np.float32]:
        result = encoder_function(idx, samples)
        numpy_result: npt.NDArray[np.float32] = np.array(result)
        return numpy_result

    return numpy_encoder_function


def get_root_traceback(exc_tb: TracebackType) -> TracebackType:
    return_traceback = exc_tb
    while return_traceback.tb_next is not None:
        return_traceback = return_traceback.tb_next
    return return_traceback


def get_root_exception_file_and_line_number() -> Tuple[int, str]:
    root_exception = sys.exc_info()[1]
    assert root_exception is not None
    if root_exception.__context__ is not None:
        root_exception = root_exception.__context__
    traceback = root_exception.__traceback__

    root_exception_line_number = -1
    root_exception_file_name = ''
    if traceback is not None:
        root_traceback = get_root_traceback(traceback)
        root_exception_line_number = root_traceback.tb_lineno
        root_exception_file_name = Path(root_traceback.tb_frame.f_code.co_filename).name
    return root_exception_line_number, root_exception_file_name


def get_shape(result: Union[npt.NDArray[np.float32], str, float, int, bool]) -> List[int]:
    if not isinstance(result, np.ndarray):
        return [1]
    np_shape = result.shape
    # fix single result shape viewing
    if np_shape == ():
        np_shape = (1,)
    shape = list(np_shape)
    return shape


def rescale_min_max(image: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    image = image.astype('float32')
    image -= image.min()
    image /= (image.max() - image.min() + 1e-5)

    # rescale the values to range between 0 and 255
    image *= 255
    image = image.astype('uint8')

    return image


