import numpy as np
import cv2
from typing import Dict, Any
from numpy.typing import NDArray


def validate_image(image: NDArray, function_name: str, expected_channels: int, expected_type: bool = None,
                   allow_float64: bool = None):
    """
        Validate the input image by checking its type, dimensions, and data type.

        Args:
            image (NDArray): The image to validate.
            function_name (str): The name of the calling function for error reporting.
            expected_channels (int): The expected number of color channels (e.g., 1 for grayscale, 3 for RGB).
            expected_type (bool, optional): Whether to check for specific dtypes. Defaults to None, meaning no dtype check.
            allow_float64 (bool, optional): Whether to allow float64 dtype in addition to uint8 and float32. Defaults to False.

        Raises:
            NotImplementedError: If the input is not a NumPy array.
            ValueError: If the image does not have the expected number of dimensions or channels.
            Exception: If the image data type is not one of the allowed types.
        """
    if not isinstance(image, np.ndarray):
        raise NotImplementedError(
            f"Wrong input type send to metadata {function_name}: Expected numpy array Got {type(image)}")

    if expected_channels == 3:
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"Wrong input dimension send to metadata {function_name}: Expected 3 Got 1")
    elif expected_channels == 1:
        if not (image.ndim == 2 or (image.ndim == 3 and image.shape[-1] == 1)):
            raise ValueError(
                f"{function_name}: Expected grayscale input (2D array or 3D with 1 channel), but got {image.ndim}D with "
                f"shape {image.shape}")

    if expected_type is not None:
        allowed_dtypes = ['uint8', 'float32']
        if allow_float64:
            allowed_dtypes.append('float64')
        if image.dtype.name not in allowed_dtypes:
            raise Exception(
                f"Wrong input type send to metadata {function_name}: Expected dtype {allowed_dtypes} Got {image.dtype.name}")


def rgb_channel_stats(image: NDArray) -> Dict[str, Any]:
    """
    Get an RGB image in shape (H,W,3) and return the mean and standard deviation for each of its color channels.

    Args:
        image (NDArray): An RGB image in shape (H,W,3) represented as a NumPy array.

    Returns:
        Dict[str, Any]: A dictionary containing:
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
    validate_image(image, rgb_channel_stats.__name__, 3)

    r, g, b = cv2.split(image)

    res = {'mean_red': np.round(r.mean(), 2),
           'mean_green': np.round(g.mean(), 2),
           'mean_blue': np.round(b.mean(), 2),
           'std_red': np.round(r.std(), 2),
           'std_green': np.round(g.std(), 2),
           'std_blue': np.round(b.std(), 2)}

    return res


def lab_channel_stats(image: NDArray) -> Dict[str, Any]:
    """
    Get an RGB image in shape (H,W,3) and return the mean of the 'a' and 'b' channels in the LAB color space.

    Args:
        image (NDArray): An RGB image in shape (H,W,3) represented as a NumPy array with float32 or uint8 values.

    Returns:
        Dict[str, Any]: A dictionary containing:
            - 'a_mean': Mean value of the 'a' channel.
            - 'b_mean': Mean value of the 'b' channel.

    Description:
        This function converts the input RGB image to the LAB color space and calculates the mean values
        of the 'a' and 'b' channels. The results are returned as a dictionary with keys corresponding
        to the mean values of the 'a' and 'b' channels. All values are rounded to two decimal places.
    """
    validate_image(image, lab_channel_stats.__name__, expected_channels=3, expected_type=True, allow_float64=False)

    img_lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    _, a, b = cv2.split(img_lab)

    mean_a = np.mean(img_lab[:, :, 1])
    mean_b = np.mean(img_lab[:, :, 2])

    res = {'a_mean': np.round(mean_a, 2),
           'b_mean': np.round(mean_b, 2)}
    return res


def detect_sharpness(image: NDArray) -> Dict[str, Any]:
    """
    Get an RGB image in shape (H,W,3) and return a sharpness metric based on the gradient magnitude.

    Args:
        image (NDArray): An image represented as a NumPy array with float32, float64 or uint8 values.

    Returns:
        Dict[str, Any]: A dictionary containing:
            - 'sharpness': The average gradient magnitude, representing the sharpness of the image.

    Description:
        This function converts the input RGB image to grayscale and computes the gradient magnitude using
        the Sobel operator in both the x and y directions. The sharpness metric is determined by calculating
        the mean of the gradient magnitude, which quantifies the overall sharpness of the image.
        The sharpness value is rounded to two decimal places before being returned.
    """
    validate_image(image, lab_channel_stats.__name__, expected_channels=1, expected_type=True, allow_float64=True)

    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    gradient_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    res = {'sharpness': np.round(np.mean(gradient_magnitude), 2)}

    return res

