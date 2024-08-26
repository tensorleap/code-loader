import numpy as np
from numpy.typing import NDArray
from scipy import fftpack # type: ignore
from code_loader.helpers.store_helpers.spectral_analysis import compute_magnitude_spectrum, radial_profile

def quantify_frequency_content(image: NDArray[np.float64], pixel_size: np.float64, f_min: np.float64, f_max: np.float64) -> NDArray[np.float64]:
    """
    Quantify the energy in a specific frequency band of an image.

    Args:
        image: Input image as a 2D numpy array.
        pixel_size: Size of a pixel in meters.
        f_min: Lower frequency bound.
        f_max: Upper frequency bound.

    Returns:
        freq_energy_ratio: Ratio of energy in the specified frequency band to the total energy.
    """
    # Compute the magnitude spectrum of the input image
    magnitude_spectrum = compute_magnitude_spectrum(image)
    # Compute the radial profile of the magnitude spectrum
    freq, radial_prof = radial_profile(magnitude_spectrum, pixel_size)
    # Define frequency bands
    f_mask = ((freq < f_max).astype(int) * (freq > f_min).astype(int)).astype(bool)
    # Calculate the energy in frequency band
    freq_energy = np.sum(radial_prof[f_mask])
    # Total energy
    total_energy = np.sum(radial_prof)
    # Ratios of energy in the specified frequency band to the total energy
    freq_energy_ratio = freq_energy / total_energy

    return freq_energy_ratio