import numpy as np
from typing import Optional
from numpy.typing import NDArray
import inspect


def validate_image(image: NDArray[np.float64], expected_channels: Optional[int] = None) -> None:
    """
        Validate the input image by checking its type, dimensions, and data type.

        Args:
            image (NDArray): The image to validate.
            expected_channels (int): The expected number of color channels (e.g., 1 for grayscale, 3 for RGB).

        Raises:
            NotImplementedError: If the input is not a NumPy array.
            ValueError: If the image does not have the expected number of dimensions or channels.
            Exception: If the image data type is not one of the allowed types.
        """

    caller_frame = inspect.stack()[1]
    caller_name = caller_frame.function

    if not isinstance(image, np.ndarray):
        raise NotImplementedError(
            f"Wrong input type sent to metadata {caller_name}: Expected numpy array Got {type(image)}.")

    if image.dtype.name != 'float64':
        raise Exception(
            f"Wrong input type sent to metadata {caller_name}: Expected dtype float64 Got {image.dtype.name}.")
    
    if image.ndim != 3:
        raise ValueError(f"Wrong input dimension sent to metadata {caller_name}: Expected 3D but Got {image.ndim}D.")


    if expected_channels and expected_channels != image.shape[-1]:
        raise ValueError(f"Wrong input dimension sent to metadata {caller_name}: Expected {expected_channels} channels, "
                         f"but Got {image.shape[-1]} channels.")