import numpy as np
from numpy.typing import NDArray
from scipy import fftpack


def quantify_frequency_content(image):
    """
    Quantify the amounts of low and high-frequency content in an image.

    Parameters:
    - image: 2D numpy array (input image)
    - cutoff_frequency: float (normalized cutoff frequency, between 0 and 0.5)

    Returns:
    - low_freq_energy_ratio: Ratio of low-frequency energy to total energy
    - high_freq_energy_ratio: Ratio of high-frequency energy to total energy
    """

    freq, radial_prof = frequency_analysis(image)

    hf_mask = ((freq < 0.45).astype(int) * (freq > 0.15).astype(int)).astype(bool)
    lf_mask = ((freq < 0.15).astype(int) * (freq > 0.04).astype(int)).astype(bool)

    # Calculate the energy in each frequency band
    low_freq_energy = np.sum(radial_prof[lf_mask])
    high_freq_energy = np.sum(radial_prof[hf_mask])
    # Total energy
    total_energy = np.sum(radial_prof)
    # Ratios of energy in low and high frequencies
    low_freq_energy_ratio = low_freq_energy / total_energy
    high_freq_energy_ratio = high_freq_energy / total_energy

    return low_freq_energy_ratio, high_freq_energy_ratio