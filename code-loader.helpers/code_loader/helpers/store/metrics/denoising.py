import numpy as np
from numpy.typing import NDArray
from code_loader.helpers.store.metadata.image import validate_image, total_variation
from code_loader.helpers.store.utils import compute_magnitude_spectrum, radial_profile


def total_variation_diff(image_1: NDArray[np.float64], image_2: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Calculate the total variation (TV) of an image.
    Args:
        image: [B,H,W,C]
    Returns:
        float: Total variation of the input image.
    """
    if image_1.shape != image_2.shape:
        raise ValueError("Error in total_variation_diff metric. The input images must have the same shape.")
    if len(image_1.shape) != 4:
        raise ValueError(f"Error in total_variation_diff metric. The input images must have 4 dimensions (B,H,W,C) but instead got {image_1.shape}.")
    
    tv_diffs = []
    barch_size = image_1.shape[0]
    for i in range(barch_size):
        tv_diff = total_variation(image_1[i]) - total_variation(image_2[i])
        tv_diffs.append(tv_diff)
    
    return np.asarray(tv_diff).astype(np.float64)


def frequency_band_retention_score(noisy: NDArray[np.float64], denoised: NDArray[np.float64], pixel_size: np.float64,
                                   f_min: np.float64, f_max: np.float64) -> NDArray[np.float64]:
    """
    Calculate the frequency band retention score (FRS) for a batch of noisy and denoised images.

    The FRS measures the retention of power in a specific frequency band after denoising,
    indicating how much of the high-frequency content (such as details or textures) is preserved.

    Args:
        noisy (NDArray[np.float64]): A batch of noisy images represented as a 3D array (batch size, height, width).
        denoised (NDArray[np.float64]): A batch of denoised images corresponding to the noisy images, 
                                        represented as a 3D array (batch size, height, width).
        pixel_size (np.float64): The size of a pixel in real-world units, used for calculating the frequency.
        f_min (np.float64): The minimum frequency of the band of interest.
        f_max (np.float64): The maximum frequency of the band of interest.

    Returns:
        NDArray[np.float64]: A 1D array containing the frequency band retention scores for each image in the batch.
    """
    if noisy.shape != denoised.shape:
        raise ValueError("Error in frequency_band_retention_score metric. The input images must have the same shape.")
    if len(noisy.shape) != 3:
        raise ValueError(f"Error in frequency_band_retention_score metric. The input images must have 3 dimensions (B,H,W) but instead got {noisy.shape}.")
    
    batch_size = noisy.shape[0]
    frs_scores = np.zeros(batch_size)

    for i in range(batch_size):
        # Extract individual samples from the batch
        noisy_sample = noisy[i]
        denoised_sample = denoised[i]

        # Compute magnitude spectra
        ps_noisy = compute_magnitude_spectrum(noisy_sample)
        ps_denoised = compute_magnitude_spectrum(denoised_sample)

        # Compute radial profile
        freq, power_noisy = radial_profile(ps_noisy, pixel_size)
        _, power_denoised = radial_profile(ps_denoised, pixel_size)

        # Define frequency bands (relative frequencies)
        f_mask = ((freq < f_max).astype(int) * (freq > f_min).astype(int)).astype(bool)

        # Compute the ratio of high-frequency power
        power_noisy = np.sum(power_noisy[f_mask])
        power_denoised = np.sum(power_denoised[f_mask])

        frs = power_denoised / power_noisy if power_noisy != 0 else 0.0

        frs_scores[i] = frs

    return frs_scores