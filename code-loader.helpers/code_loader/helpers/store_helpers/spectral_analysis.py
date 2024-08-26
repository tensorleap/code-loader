from typing import Tuple
import numpy as np
from numpy.typing import NDArray
from scipy import fftpack  # type: ignore
from code_loader.helpers.store_helpers.validators import validate_image  # type: ignore


def compute_power_spectrum(image: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Compute the power spectrum of an input image using Fast Fourier Transform (FFT).

    This function applies a 2D FFT to the input image, shifts the zero frequency component
    to the center of the spectrum, and calculates the power spectrum.

    Args:
        image (numpy.ndarray): Input image as a 2D numpy array.

    Returns:
        power_spectrum: Power spectrum of the input image, represented as the squared magnitude
                        of the shifted Fourier transform.
    """
    validate_image(image, expected_channels=2)
    f = fftpack.fft2(image)
    fshift = fftpack.fftshift(f)
    power_spectrum = np.abs(fshift) ** 2
    return power_spectrum


def compute_magnitude_spectrum(image: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Compute the magnitude spectrum of an input image using Fast Fourier Transform (FFT).

    This function applies a 2D FFT to the input image, shifts the zero frequency component
    to the center of the spectrum, and calculates the magnitude spectrum in decibels.

    Args:
        image (numpy.ndarray): Input image as a 2D numpy array.

    Returns:
        magnitude_spectrum: Magnitude spectrum of the input image in decibels.
    """
    validate_image(image, expected_channels=2)
    f = fftpack.fft2(image)
    fshift = fftpack.fftshift(f)
    magnitude_spectrum = 20 * np.log10(np.abs(fshift))
    return magnitude_spectrum


def radial_profile(spectrum_array: NDArray[np.float64], pixel_size: np.float64) -> NDArray[np.float64]:
    """
    Compute the radial profile of a 2D spectrum.

    This function calculates the average values of the input data at different
    radial distances from a specified center point. It's often used in image
    processing and signal analysis to analyze radially symmetric patterns.

    Args:
        spectrum_array : 2D input array for which to compute the radial profile.
        pixel_size : Size of a pixel in meters.

    Returns:
        freq : Array of spatial frequencies in cycles per meter.
        radialprofile : Radial profile of the input data, averaged over each radial distance

    Note:
        The radial distances are rounded to the nearest integer, so the resolution
        of the profile depends on the size and scale of the input data.
    """
    y, x = np.indices((spectrum_array.shape[:2]))
    center = (spectrum_array.shape[0] // 2, spectrum_array.shape[1] // 2)
    r = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
    r = r.astype(np.int32)
    tbin = np.bincount(r.ravel(), spectrum_array.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / nr

    # Calculate spatial frequencies in cycles per meter
    nyquist_freq = 1 / (2 * pixel_size)  # Nyquist frequency in cycles per meter
    freq_step = nyquist_freq / (len(radialprofile) - 1)
    freq = np.arange(0, nyquist_freq + freq_step, freq_step)
    return freq, radialprofile
