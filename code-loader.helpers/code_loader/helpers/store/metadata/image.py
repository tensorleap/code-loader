import numpy as np
import cv2
from typing import Dict, Any, Optional, Union
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

    if expected_channels and expected_channels == 3:
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"Wrong input dimension sent to metadata {caller_name}: Expected 3D with {expected_channels} "
                             f"channels, but Got {image.ndim}D with {image.shape[-1]} channels .")
    else:
        if image.ndim != 3:
            raise ValueError(f"Wrong input dimension sent to metadata {caller_name}: Expected 3D but Got {image.ndim}D.")


def rgb_channel_stats(image: NDArray[np.float64]) -> Dict[str, np.float64]:
    """
    Get an RGB image in shape (H,W,3) and return the mean and standard deviation for each of its color channels.

    Args: image (NDArray[np.float64]): An RGB image in shape (H,W,3) represented as a NumPy array.

    Returns:
        Dict[str, np.float64]: A dictionary containing:
            - 'mean_red': Mean brightness of the red channel.
            - 'mean_green': Mean brightness of the green channel.
            - 'mean_blue': Mean brightness of the blue channel.
            - 'brightness_std_red': Standard deviation of brightness for the red channel.
            - 'brightness_std_green': Standard deviation of brightness for the green channel.
            - 'brightness_std_blue': Standard deviation of brightness for the blue channel.

    Description:
        This function calculates the mean and standard deviation of the red, green, and blue channels of the input RGB image.
        The results are returned as a dictionary with keys corresponding to the statistical measures for each channel.
        All values are rounded to two decimal places.
    """
    validate_image(image, expected_channels=3)

    r, g, b = cv2.split(image)

    res = {'mean_red': np.round(np.array(r).mean(), 2),
           'mean_green': np.round(np.array(g).mean(), 2),
           'mean_blue': np.round(np.array(b).mean(), 2),
           'std_red': np.round(np.array(r).std(), 2),
           'std_green': np.round(np.array(g).std(), 2),
           'std_blue': np.round(np.array(b).std(), 2)}

    return res


def lab_channel_stats(image: NDArray[np.float64]) -> Dict[str, Any]:
    """
    Get an RGB image in shape (H,W,3) and return the mean of the 'a' and 'b' channels in the LAB color space.

    Args: image (NDArray[np.float64]): A RGB image in shape (H,W,3) represented as a NumPy array.

    Returns:
        Dict[str, Any]: A dictionary containing:
            - 'a_mean': Mean value of the 'a' channel.
            - 'b_mean': Mean value of the 'b' channel.

    Description:
        This function converts the input RGB image to the LAB color space and calculates the mean values
        of the 'a' and 'b' channels. The results are returned as a dictionary with keys corresponding
        to the mean values of the 'a' and 'b' channels. All values are rounded to two decimal places.
    """
    validate_image(image)

    img_lab = cv2.cvtColor(image.astype('float32'), cv2.COLOR_RGB2LAB)
    _, a, b = cv2.split(img_lab)

    mean_a = np.mean(img_lab[:, :, 1])
    mean_b = np.mean(img_lab[:, :, 2])

    res = {'a_mean': np.round(mean_a, 2),
           'b_mean': np.round(mean_b, 2)}
    return res


def detect_sharpness(image: NDArray[np.float64]) -> Dict[str, np.float64]:
    """
    Get an image in shape (H,W,C) and return a sharpness metric based on the gradient magnitude.

    Args: image (NDArray[np.float64]): A gray scale image represented as a NumPy array.

    Returns:
        Dict[str, np.float64]: A dictionary containing:
            - 'sharpness': The average gradient magnitude, representing the sharpness of the image.

    Description:
        This function computes the gradient magnitude using the Sobel operator in both the x and y directions.
        The sharpness metric is determined by calculating the mean of the gradient magnitude, which quantifies the
        overall sharpness of the image.
        The sharpness value is rounded to two decimal places before being returned.
    """
    validate_image(image)

    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    gradient_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    res = {'sharpness': np.round(np.mean(gradient_magnitude), 2)}

    return res
