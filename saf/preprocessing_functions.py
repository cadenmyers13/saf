import numpy as np
from scipy.ndimage import gaussian_filter


def mask_and_blur_annulus(
    array,
    inner_radius,
    outer_radius,
    sigma=0.3,
    x_shift=0,
    y_shift=0,
    blur=True,
):
    """Masks data in the shape of an annulus based off inner_radius and
    outer_radius if `blur=True`, gaussian blur is applied to masked `array`."""
    array_shape = array.shape
    x, y = np.meshgrid(np.arange(array_shape[0]), np.arange(array_shape[1]))
    radius = np.sqrt(
        (x - (array_shape[0] - x_shift) // 2) ** 2
        + (y - (array_shape[1] - y_shift) // 2) ** 2
    )
    mask1 = radius <= inner_radius
    mask2 = radius >= outer_radius
    masked_data = array.copy()
    masked_data[mask1] = 0
    masked_data2 = masked_data.copy()
    masked_data2[mask2] = 0
    if blur:
        blurred_data = gaussian_filter(masked_data2, sigma=sigma)
        return blurred_data
    else:
        return masked_data2


def normalize_min_max(data):
    """Min max normalization of an array."""
    try:
        array = data.copy()
    except AttributeError:
        array = np.array(data, copy=True)
    if array.size == 0:
        raise ValueError("Cannot normalize an empty array.")
    array_min = np.min(array)
    array_max = np.max(array)
    if array_max == array_min:
        return np.zeros_like(array)
    norm_array = (array - array_min) / (array_max - array_min)
    return norm_array
