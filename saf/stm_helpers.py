import numpy as np
from scipy.signal import find_peaks

from saf.filters import annulus_filter


def regress_annulus(r_min, r_max, array, cx, cy, sd=2, r_step=1):
    """Estimate the characteristic radius of a feature in an image using
    annulus filters.

    For each radius in the specified range, this function computes the overlap between
    an annulus filter (centered at cx, cy) and the input image array. It removes a
    linear background from the overlap signal and returns the radius at the first
    peak in the normalized result.

    Parameters:
        r_min (int): Minimum radius to consider.
        r_max (int): Maximum radius to consider.
        array (ndarray): 2D image array to analyze.
        cx (float): X-coordinate of the annulus center.
        cy (float): Y-coordinate of the annulus center.
        sd (int, optional): Standard deviation for the Gaussian filter. Defaults to 2.
        r_step (int, optional): Radius increment. Defaults to 1.

    Returns:
        float: Radius corresponding to the first peak in normalized overlap.

    Raises:
        ValueError: If no peak is found in the normalized overlap.
    """
    radii = np.arange(r_min, r_max, r_step)
    overlap_scores = []
    shape = array.shape
    for r in radii:
        annulus = annulus_filter(r_0=r, sd=sd, imshape=shape, cx=cx, cy=cy)
        overlap = (annulus * array).sum()
        overlap_scores.append(overlap)

    # Linear fit to normalize overlap score
    m, b = np.polyfit(radii, overlap_scores, 1)
    y_fit = m * np.array(radii) + b
    normalized_overlap = np.array(overlap_scores) - y_fit

    # Find peaks in normalized overlap
    peaks, _ = find_peaks(normalized_overlap)

    if len(peaks) > 0:
        first_peak_idx = peaks[0]
        first_peak_r = radii[first_peak_idx]
    else:
        print(
            f"No peaks detected in the normalized overlap. Setting r={r_min}."
        )
        first_peak_r = r_min
    # Compute overlap array for the first peak radius
    annulus = annulus_filter(
        r_0=first_peak_r, sd=sd, imshape=shape, cx=cx, cy=cy
    )
    overlap_array = annulus * array
    return first_peak_r, overlap_array
