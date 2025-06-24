import numpy as np
from scipy.ndimage import gaussian_filter


import numpy as np
from scipy.ndimage import gaussian_filter

def mask_and_blur_annulus(
    array,
    inner_radius=14,  # default values for SkL case
    outer_radius=30,  # default values for SkL case
    sigma=0.3,
    x_shift=0,
    y_shift=0,
    blur=False,
):
    """Apply an annular mask and optional Gaussian blur to a 2D image or a stack of 2D images.
    
    Parameters
    ----------
    array : np.ndarray
        2D image or 3D stack of images (shape: [height, width] or [n_images, height, width])
    inner_radius : float
        Inner radius of the annulus.
    outer_radius : float
        Outer radius of the annulus.
    sigma : float
        Standard deviation for Gaussian kernel if blur=True.
    x_shift : int
        Horizontal shift for center of annulus.
    y_shift : int
        Vertical shift for center of annulus.
    blur : bool
        Whether to apply Gaussian blur after masking.

    Returns
    -------
    np.ndarray
        Masked (and optionally blurred) array of the same shape as input.
    """
    # Handle single image as a stack of one
    is_single_image = (array.ndim == 2)
    if is_single_image:
        array = array[np.newaxis, ...]  # Add image stack dimension

    n_images, height, width = array.shape
    x, y = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    radius = np.sqrt(
        (x - (height - x_shift) // 2) ** 2 +
        (y - (width - y_shift) // 2) ** 2
    )
    mask = (radius > inner_radius) & (radius < outer_radius)

    result = np.empty_like(array)
    for i in range(n_images):
        masked = np.where(mask, array[i], 0)
        if blur:
            masked = gaussian_filter(masked, sigma=sigma)
        result[i] = masked

    return result[0] if is_single_image else result



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
