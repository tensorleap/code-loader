import numpy as np
from numpy.typing import NDArray
from scipy import fftpack
from code_loader.helpers.store_helpers.spectral_analysis import compute_magnitude_spectrum, radial_profile

def quantify_frequency_content(image: NDArray[np.float64], f_min: np.float64, f_max: np.float64) -> np.float64:
    """
    Quantify the energy in a specific frequency band of an image.

    Args:
        image: Input image as a 2D numpy array.
        f_min: Lower frequency bound.
        f_max: Upper frequency bound.

    Returns:
        freq_energy_ratio: Ratio of energy in the specified frequency band to the total energy.
    """
    magnitude_spectrum = compute_magnitude_spectrum(image)
    freq, radial_prof = radial_profile(magnitude_spectrum)

    f_mask = ((freq < f_max).astype(int) * (freq > f_min).astype(int)).astype(bool)

    # Calculate the energy in frequency band
    freq_energy = np.sum(radial_prof[f_mask])
    # Total energy
    total_energy = np.sum(radial_prof)
    # Ratios of energy in low and high frequencies
    freq_energy_ratio = freq_energy / total_energy

    return freq_energy_ratio