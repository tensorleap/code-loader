from typing import Dict
import numpy as np
from numpy.typing import NDArray
from code_loader.helpers.store_helpers.spectral_analysis import compute_magnitude_spectrum, radial_profile  # type: ignore

def frequency_band_retention_score(noisy: NDArray[np.float64], denoised: NDArray[np.float64], 
                                   f_min: np.float64, f_max: np.float64) -> NDArray[np.float64]:

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
        center = tuple(map(lambda x: x // 2, ps_noisy.shape))
        freq, power_noisy = radial_profile(ps_noisy, center)
        _, power_denoised = radial_profile(ps_denoised, center)

        # Define frequency bands (relative frequencies)
        f_mask = ((freq < f_max).astype(int) * (freq > f_min).astype(int)).astype(bool)

        # Compute the ratio of high-frequency power
        power_noisy = np.sum(power_noisy[f_mask])
        power_denoised = np.sum(power_denoised[f_mask])

        frs = power_denoised / power_noisy if power_noisy != 0 else 0.0

        frs_scores[i] = frs

    return frs_scores