import numpy as np

from saf.filters import n_fold_filter
from saf.ratchet_model_helpers import (
    compute_loss_near_offset1,
    get_approx_offset_values,
)


def regress_offset_ccw(
    dps,
    tolerance_forward,
    tolerance_reverse,
    n_folds,
    k,
    imshape=None,
    cx=None,
    cy=None,
    offset_step=0.5,
    mask_threshold=0.01,
):
    """Perform a regression to determine optimal angular offsets for a set of
    diffraction patterns, primarily searching in the counter-clockwise (CCW)
    direction while allowing limited backward (clockwise, CW) search.

    This function computes two angular offsets (offset1 and offset2) for each
    diffraction pattern in the input. The first offset is determined by minimizing
    a loss function based on a filtered version of the pattern. The second offset
    is computed after masking out the signal corresponding to the first offset,
    allowing for the detection of additional domains.

    Parameters:
        dps (numpy.ndarray): A 2D array (single diffraction pattern) or a 3D array
            (stack of diffraction patterns) representing the input data.
        tolerance_forward (float): The forward search range (in degrees) for offsets
            relative to the previously determined offset.
        tolerance_reverse (float): The backward search range (in degrees) for offsets
            relative to the previously determined offset.
        n_folds (int): The number of folds (symmetry) to apply in the filtering process.
        k (float): A scaling factor used in the filtering process.
        imshape (tuple, optional): The shape of the diffraction patterns. If None,
            it is inferred from the input data.
        cx (int, optional): The x-coordinate of the center of the diffraction pattern.
            Defaults to middle of dps.
        cy (int, optional): The y-coordinate of the center of the diffraction pattern.
            Defaults to middle of dps.
        offset_step (float, optional): The step size (in degrees) for the angular search.
            Defaults to 0.5.
        mask_threshold (float, optional): The threshold for masking out the first signal
            during the computation of the second offset. Defaults to 0.01.

    Returns:
        tuple: A tuple containing two lists:
            - optimal_offsets1 (list of float): The optimal values for the first offset
              (in degrees) for each diffraction pattern.
            - optimal_offsets2 (list of float): The optimal values for the second offset
              (in degrees) for each diffraction pattern.

    Raises:
        ValueError: If the input data shape is not supported (neither 2D nor 3D).

    Notes:
        - The function uses a loss function based on the dot product of the diffraction
          pattern and a filter generated using the specified number of folds and offsets.
        - The second offset is computed only if multiple domains are detected based on
          the second derivative of the loss function near the first offset.
    """
    if imshape is None:
        if dps.ndim == 2:
            imshape = dps.shape  # single image
            dps = [dps]  # make 2D array iterable
        elif dps.ndim == 3:
            imshape = dps.shape[1:]  # stack of images
        else:
            raise ValueError(f"Unsupported data shape: {dps.shape}")

    if cx is None:
        cx = imshape[1] // 2
    if cy is None:
        cy = imshape[0] // 2

    optimal_offsets1 = []
    optimal_offsets2 = []
    prev_offset1 = 0
    prev_offset2 = 0

    for index, dp in enumerate(dps):
        loss_list1 = []
        offset_list_deg1 = []

        # Find initial offset
        if index == 0:
            offset1_angles_deg = np.arange(
                -360 / (2 * n_folds), 360 / (2 * n_folds), offset_step
            )
        else:
            # define new angle range based off previously determined offset
            offset1_angles_deg = np.arange(
                prev_offset1 - tolerance_reverse,
                prev_offset1 + tolerance_forward,
                offset_step,
            )

        for offset1 in offset1_angles_deg:
            offset_rad = np.deg2rad(offset1)
            filt = n_fold_filter(
                k=k,
                offset=offset_rad,
                n_folds=n_folds,
                cx=cx,
                cy=cy,
                imshape=imshape,
            )
            loss = -(dp * filt).sum()
            loss_list1.append(loss)
            offset_list_deg1.append(offset1)

        # assign offset1
        min_loss_idx = loss_list1.index(min(loss_list1))
        best_offset1 = offset_list_deg1[min_loss_idx]

        # Use second derivative to determine number of domains
        offset_range, _, _, second_derivative = compute_loss_near_offset1(
            dp, np.deg2rad(best_offset1), n_folds, k
        )
        approx_offset_values = get_approx_offset_values(
            offset_range, second_derivative
        )
        number_of_domains = len(approx_offset_values)

        # mask out first signal
        filt1 = n_fold_filter(
            k=k,
            offset=np.deg2rad(best_offset1),
            n_folds=n_folds,
            cx=cx,
            cy=cy,
            imshape=imshape,
        )
        filt1 = np.where(filt1 > mask_threshold, 0, 1)
        dp_filtered = dp * filt1

        if number_of_domains == 1:
            loss_list2 = []
            offset_list_deg2 = []
            offset2_angles_deg = np.arange(
                prev_offset2 - 360 / (2 * n_folds),
                prev_offset2 + 360 / (2 * n_folds),
                offset_step,
            )
        else:
            loss_list2 = []
            offset_list_deg2 = []
            offset2_angles_deg = np.arange(
                prev_offset2 - tolerance_reverse,
                prev_offset2 + tolerance_forward,
                offset_step,
            )

        # Compute loss for offset2
        for offset2 in offset2_angles_deg:
            offset_rad = np.deg2rad(offset2)
            filt2 = n_fold_filter(
                k, offset_rad, n_folds=n_folds, cx=cx, cy=cy, imshape=imshape
            )
            loss = (
                -(dp_filtered * filt2).sum()
                if number_of_domains > 1
                else -(dp * filt2).sum()
            )
            loss_list2.append(loss)
            offset_list_deg2.append(offset2)

        # assign offset2
        min_loss_idx = loss_list2.index(min(loss_list2))
        best_offset2 = offset_list_deg2[min_loss_idx]

        # Store results and update previous offsets
        optimal_offsets1.append(best_offset1)
        optimal_offsets2.append(best_offset2)
        prev_offset1 = best_offset1
        prev_offset2 = best_offset2

    return optimal_offsets1, optimal_offsets2


def regress_offset_cw(
    dps,
    tolerance_forward,
    tolerance_reverse,
    n_folds,
    k,
    imshape=None,
    cx=None,
    cy=None,
    offset_step=0.5,
    mask_threshold=0.01,
):
    """
        Perform regression to determine optimal offsets for a given dataset of
    diffraction patterns (dps) using n-fold symmetry filtering.
    This function iteratively computes the optimal offsets for each diffraction
    pattern in the dataset by minimizing a loss function. It supports both single
    and multiple domain scenarios and applies masking to isolate signals.
    Parameters:
        dps (numpy.ndarray): A 2D array (single image) or 3D array (stack of images)
            representing the diffraction patterns.
        tolerance_forward (float): The forward tolerance for restricting the search
            range of offsets.
        tolerance_reverse (float): The reverse tolerance for restricting the search
            range of offsets.
        n_folds (int): The number of folds for the symmetry filter.
        k (float): A parameter for the n-fold filter.
        imshape (tuple, optional): The shape of the diffraction pattern images. If
            None, it is inferred from the input `dps`.
        cx (int, optional): The x-coordinate of the center of the image. Default is 0.
        cy (int, optional): The y-coordinate of the center of the image. Default is 0.
        offset_step (float, optional): The step size for the offset angle in degrees.
            Default is 0.5.
        mask_threshold (float, optional): The threshold for masking the first signal.
            Default is 0.01.
    Returns:
        tuple: A tuple containing two lists:
            - optimal_offsets1 (list): The optimal offsets for the first domain.
            - optimal_offsets2 (list): The optimal offsets for the second domain.
    Raises:
        ValueError: If the shape of `dps` is not supported (neither 2D nor 3D).
    Notes:
        - The function uses an n-fold symmetry filter to compute the loss function.
        - The search range for offsets is dynamically adjusted based on the previous
          offset and the specified tolerances.
        - For single-domain scenarios, the second offset is computed over a wider
          range, while for multiple-domain scenarios, it is restricted based on
          tolerances.
    """
    if imshape is None:
        if dps.ndim == 2:
            imshape = dps.shape  # single image
            dps = [dps]  # make 2D array iterable
        elif dps.ndim == 3:
            imshape = dps.shape[1:]  # stack of images
        else:
            raise ValueError(f"Unsupported dps shape: {dps.shape}")

    if cx is None:
        cx = imshape[1] // 2
    if cy is None:
        cy = imshape[0] // 2

    optimal_offsets1 = []
    optimal_offsets2 = []
    prev_offset1 = 0
    prev_offset2 = 0
    for index, dp in enumerate(dps):
        loss_list1 = []
        offset_list_deg1 = []
        if index == 0:
            offset1_angles_deg = np.arange(
                -360 / (2 * n_folds), 360 / (2 * n_folds), offset_step
            )
        else:
            # Restrict search to mostly CW direction
            offset1_angles_deg = np.arange(
                prev_offset1 - tolerance_forward,
                prev_offset1 + tolerance_reverse,
                offset_step,
            )

        # Compute loss function for offset1
        for offset1 in offset1_angles_deg:
            offset_rad = np.deg2rad(offset1)
            filt = n_fold_filter(
                k=k,
                offset=offset_rad,
                n_folds=n_folds,
                cx=cx,
                cy=cy,
                imshape=imshape,
            )
            loss = -(dp * filt).sum()
            loss_list1.append(loss)
            offset_list_deg1.append(offset1)

        # find offset1
        min_loss_idx = loss_list1.index(min(loss_list1))
        best_offset1 = offset_list_deg1[min_loss_idx]

        # Determine number of domains present
        offset_range, _, _, second_derivative = compute_loss_near_offset1(
            dp, np.deg2rad(best_offset1), n_folds=n_folds, k=k
        )
        number_of_domains = len(
            get_approx_offset_values(offset_range, second_derivative)
        )

        # mask out first signal
        filt1 = n_fold_filter(
            k=k,
            offset=np.deg2rad(best_offset1),
            n_folds=n_folds,
            cx=cx,
            cy=cy,
            imshape=imshape,
        )
        filt1 = np.where(filt1 > mask_threshold, 0, 1)
        dp_filtered = dp * filt1

        if number_of_domains == 1:
            loss_list2 = []
            offset_list_deg2 = []
            offset2_angles_deg = np.arange(
                prev_offset2 - 360 / (2 * n_folds),
                prev_offset2 + 360 / (2 * n_folds),
                offset_step,
            )
        else:
            loss_list2 = []
            offset_list_deg2 = []
            offset2_angles_deg = np.arange(
                prev_offset2 - tolerance_forward,
                prev_offset2 + tolerance_reverse,
                offset_step,
            )

        # Compute loss function for offset2
        for offset2 in offset2_angles_deg:
            offset_rad = np.deg2rad(offset2)
            filt2 = n_fold_filter(
                k=k,
                offset=offset_rad,
                n_folds=n_folds,
                cx=cx,
                cy=cy,
                imshape=imshape,
            )
            loss = (
                -(dp_filtered * filt2).sum()
                if number_of_domains > 1
                else -(dp * filt2).sum()
            )
            loss_list2.append(loss)
            offset_list_deg2.append(offset2)

        min_loss_idx = loss_list2.index(min(loss_list2))
        best_offset2 = offset_list_deg2[min_loss_idx]

        # Store results
        optimal_offsets1.append(best_offset1)
        optimal_offsets2.append(best_offset2)
        prev_offset1 = best_offset1
        prev_offset2 = best_offset2

    return optimal_offsets1, optimal_offsets2
