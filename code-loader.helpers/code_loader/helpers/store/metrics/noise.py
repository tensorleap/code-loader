import numpy as np
from numpy.typing import NDArray
from code_loader.helpers.store_helpers.validators import validate_image
from code_loader.helpers.store.noise import total_vairation

def total_vairation_diff(image_1: NDArray[np.float64], image_2: NDArray[np.float64]) -> float:
    """
    Calculate the total variation (TV) of an image.
    Args:
        image: [H,W,C]
    Returns:
        float: Total variation of the input image.
    """
    validate_image(image_1)
    validate_image(image_2)

    tv_diff = total_vairation(image_1) - total_vairation(image_2)
    
    return tv_diff