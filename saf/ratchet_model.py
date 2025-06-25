import numpy as np

from saf.filters import n_fold_filter
from saf.regression_functions import regress_offset_ccw, regress_offset_cw


def determine_rotation_direction(
    dps, n_folds, k, imshape=None, cx=0, cy=0, offset_step=0.5
):
    """
    The function calculates the optimal rotation offsets for a series of
    diffraction patterns and determines whether the rotation is counterclockwise
    (`ccw`) or clockwise (`cw`) based on the slope of a linear fit to the offsets.
    Parameters:
        dps (numpy.ndarray): Diffraction patterns, either as a single 2D image
            or a stack of 3D images.
        n_folds (int): Number of rotational symmetry folds.
        k (float): Scaling factor for the n-fold filter.
        imshape (tuple, optional): Shape of the diffraction pattern images.
            If None, it is inferred from `dps`. Defaults to None.
        cx (int, optional): X-coordinate of the center for the n-fold filter.
            Defaults to 0.
        cy (int, optional): Y-coordinate of the center for the n-fold filter.
            Defaults to 0.
        offset_step (float, optional): Step size for the offset angle in degrees.
            Defaults to 0.5.
    Returns:
        str: "ccw" if the rotation direction is counterclockwise,
             "cw" if the rotation direction is clockwise.
    Raises:
        ValueError: If the shape of `dps` is not supported.
    Notes:
        - The function uses at most the first 15 diffraction patterns from `dps`.
        - The rotation direction is determined by fitting a linear model to
          the optimal offsets and analyzing the slope.
    """
    # If dps is a single image, convert to list
    if imshape is None:
        if dps.ndim == 2:
            imshape = dps.shape  # single image
            dps = [dps]  # make 2D array iterable
        elif dps.ndim == 3:
            imshape = dps.shape[1:]  # stack of images
        else:
            raise ValueError(f"Unsupported dps shape: {dps.shape}")
    # Use at most the first 15 diffraction patterns
    dps = dps[:15]
    optimal_offsets = []
    prev_offset = 0

    for index, dp in enumerate(dps):
        loss_list = []
        offset_list_deg = []
        # Set offset angle range
        if index == 0:
            offset_angles_deg = np.arange(
                -360 / (3 * n_folds), 360 / (3 * n_folds), offset_step
            )
        else:
            offset_angles_deg = np.arange(
                prev_offset - 360 / (3 * n_folds),
                prev_offset + 360 / (3 * n_folds),
                offset_step,
            )
        for offset in offset_angles_deg:
            offset_rad = np.deg2rad(offset)
            filt = n_fold_filter(
                k, offset_rad, n_folds=n_folds, cx=cx, cy=cy, imshape=imshape
            )
            loss = -(dp * filt).sum()
            loss_list.append(loss)
            offset_list_deg.append(offset)

        min_loss_idx = np.argmin(loss_list)
        best_offset = offset_list_deg[min_loss_idx]
        optimal_offsets.append(best_offset)
        prev_offset = best_offset

    frames = np.arange(len(optimal_offsets))
    offsets = np.array(optimal_offsets)
    slope, _ = np.polyfit(frames, offsets, 1)
    return "ccw" if slope > 0 else "cw"


def calculate_offsets(
    dps, n_folds, k, tolerance_forward, tolerance_reverse, imshape=None
):
    """
    Calculates offset values using the ratchet model.
    This function determines the rotation direction of the input data and
    computes offset values based on the specified tolerances and parameters.
    Args:
        dps (numpy.ndarray): Input data points, either a single 2D image
            (shape: [height, width]) or a stack of 2D images (shape: [num_images, height, width]).
        n_folds (int): Number of folds or symmetry in the data.
        k (float): Scaling factor for the n-fold filter.
        imshape (tuple, optional): Shape of the diffraction pattern images.
        tolerance_forward (float): Forward tolerance value for offset regression.
        tolerance_reverse (float): Reverse tolerance value for offset regression.
        imshape (tuple, optional): Shape of the input images. If None, it will be
            inferred from the shape of `dps`. Defaualt is None.
    Returns:
        tuple: A tuple containing two lists of offsets:
            - offsets1 (list): Offset values for the first direction.
            - offsets2 (list): Offset values for the second direction.
    Raises:
        ValueError: If the shape of `dps` is not supported (neither 2D nor 3D).
    """
    if imshape is None:
        if dps.ndim == 2:
            imshape = dps.shape  # single image
            dps = [dps]  # make 2D array iterable
        elif dps.ndim == 3:
            imshape = dps.shape[1:]  # stack of images
        else:
            raise ValueError(f"Unsupported dps shape: {dps.shape}")

    direction = determine_rotation_direction(
        dps=dps, n_folds=n_folds, k=k, imshape=imshape
    )

    if direction == "cw":
        offsets1, offsets2, overlap_score1, overlap_score2 = regress_offset_cw(
            dps=dps,
            n_folds=n_folds,
            k=k,
            imshape=imshape,
            tolerance_forward=tolerance_forward,
            tolerance_reverse=tolerance_reverse,
        )
    else:
        offsets1, offsets2, overlap_score1, overlap_score2 = (
            regress_offset_ccw(
                dps=dps,
                n_folds=n_folds,
                k=k,
                imshape=imshape,
                tolerance_forward=tolerance_forward,
                tolerance_reverse=tolerance_reverse,
            )
        )

    return offsets1, offsets2, overlap_score1, overlap_score2
