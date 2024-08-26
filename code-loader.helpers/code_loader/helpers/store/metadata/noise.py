import numpy as np
from numpy.typing import NDArray

import scipy.ndimage # type: ignore
import skimage # type: ignore

from code_loader.helpers.store.image import validate_image  # type: ignore

def get_abs_log_metadata(image: NDArray[np.float64], sigma=1) -> NDArray[np.float64]:
    """
    Gets an image returns the absolute value of a LOG (Laplacian of Gaussians).
    Can be used to detect non-flat areas
    Args:
        image: An image [H,W,C]
        sigma: The sigma of the Gassian filter

    Returns:
        The absolute value of the LOG in shape [H,W,C]
    """
    validate_image(image)
    log = scipy.ndimage.gaussian_laplace(image, sigma)
    return np.abs(log)

def estimate_noise_sigma(image: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Estimate the noise in an image using the sigma method
    Args: 
        image: [H, W, C]

    Returns:
        The estimated noise sigma
    """
    validate_image(image)
    sigma = skimage.restoration.estimate_sigma(image, average_sigmas=True, channel_axis=-1)
    
    return sigma

def estimate_noise_laplacian(image: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Estimate the noise in an image using the Laplacian method.
    Args:
        image: [H,W,C]
    Returns:
        float: Estimated noise level in the input image.
    """
    filtered_image = scipy.ndimage.laplace(image)
    sigma = np.mean(np.abs(filtered_image))
    return sigma

def estimate_noise(image: NDArray[np.float64], method: str = 'sigma') -> NDArray[np.float64]:
    """
    Estimate the noise in an image using a specified method.
    Args:
        image: Input image as a 2D or 3D numpy array.
        method: Method to use for noise estimation. Supported methods are 'sigma' and 'laplacian'.
    Returns:
        float: Estimated noise level in the input image.
    """
    validate_image(image)
    if method == 'sigma':
        return estimate_noise_sigma(image)
    elif method == 'laplacian':
        return estimate_noise_laplacian(image)
    else:
        raise ValueError(f"Unsupported noise estimation method: {method}")


def total_variation(image: NDArray[np.float64]) -> np.float64:
    """
    Calculate the total variation (TV) of an image.

    Total variation is a measure of the smoothness of an image. It is often used 
    in image processing to reduce noise while preserving edges.

    Args:
        image: A 2D or 3D array representing the image. For a 3D array,
                            the channels should be along the last dimension.

    Returns:
        float: The total variation of the image, which is the sum of the magnitudes 
               of the gradients across all pixels.
    """
    validate_image(image)
    grad = np.array(np.gradient(image))
    return np.sum(np.sqrt(np.sum(grad**2, axis=0)))