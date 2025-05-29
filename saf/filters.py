import numpy as np


def find_k_value(resolution, n_folds):
    """Finds k given resolution(deg).

    This assumes FWHM_experiment is approx equal to FWHM_filter
    """
    res_rad = np.deg2rad(resolution)
    k = np.log(1 / 2) / (np.log(np.cos((n_folds / 4 * res_rad)) ** 2))
    # print('k=', k)
    return k


def n_fold_filter(k, offset, n_folds, imshape, cx=0, cy=0):
    """
    Generate an n-fold symmetric filter based on the input parameters.
    This function creates a filter that applies an n-fold rotational symmetry
    to a 2D image grid. The filter is computed using trigonometric functions
    and logarithmic scaling, and it is centered around a specified point.
    Parameters:
        k (float): A scaling factor that controls the sharpness of the filter.
        offset (float): An angular offset (in radians) applied to the symmetry.
        n_folds (int): The number of symmetric folds to apply.
        imshape (tuple): A tuple (h, w) representing the height and width of the image.
        cx (int, optional): The x-coordinate of the center of rotation. Defaults to 0.
        cy (int, optional): The y-coordinate of the center of rotation. Defaults to 0.
    Returns:
        numpy.ndarray: A 2D array of the same shape as the input image,
        representing the n-fold symmetric filter.
    Notes:
        - The filter is computed in polar coordinates relative to the center
          of the image grid.
        - The `k` parameter controls the steepness of the filter's response,
          with higher values resulting in sharper transitions.
        - The `offset` parameter allows for rotation of the symmetry pattern.
    """

    h, w = imshape
    x_grid, y_grid = np.meshgrid(np.arange(w), np.arange(h))
    x_rel = x_grid - cx
    y_rel = y_grid - cy

    theta = np.arctan2(y_rel, x_rel)
    return np.exp(k * np.log(np.cos(n_folds / 2 * (offset + theta)) ** 2))


def gaussian_filter(sigma, imshape, cx=0, cy=0):
    """Generate a 2D Gaussian blob centered at (cx, cy) in an image of shape
    `imshape`.

    Parameters:
        sigma (float): Standard deviation of the Gaussian.
        imshape (tuple): Shape of the image as (height, width).
        cx (float): x-coordinate of the Gaussian center (column index).
        cy (float): y-coordinate of the Gaussian center (row index).

    Returns:
        numpy.ndarray: A 2D Gaussian blob.
    """
    h, w = imshape
    y_grid, x_grid = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    distance_squared = (x_grid - cx) ** 2 + (y_grid - cy) ** 2
    gaussian = np.exp(-distance_squared / (2 * sigma**2))
    return gaussian


def annulus_filter(r_0, sd, imshape, cx=0, cy=0):
    """Generate a circular annulus-shaped filter centered at (cx, cy) in a 2D
    image.

    Parameters:
        r_0 (float): Desired radius of the annulus (in pixels).
        sd (float): Standard deviation controlling the ring width.
        imshape (tuple): Shape of the image as (height, width).
        cx (float): x-coordinate (column) of the annulus center.
        cy (float): y-coordinate (row) of the annulus center.

    Returns:
        ndarray: 2D annulus filter.
    """
    h, w = imshape
    y, x = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    annulus = np.exp(-((r - r_0) ** 2) / (2 * sd**2))
    return annulus


# def data_theta(imshape, shift_x=None, shift_y=None):
#     '''given the shape of your image, this function outputs DATA_THETA'''
#     if shift_x == None:
#         shift_x = 0
#     if shift_y == None:
#         shift_y = 0
#     h, w = imshape
#     cx, cy = w // 2, h // 2
#     x_grid, y_grid = np.meshgrid(np.arange(w), np.arange(h))
#     # handle slight offcentering of DPs
#     x_grid_shifted = x_grid - shift_x
#     y_grid_shifted = y_grid - shift_y
#     DATA_THETA = np.arctan2(y_grid_shifted - cy, x_grid_shifted - cx)
#     return DATA_THETA
