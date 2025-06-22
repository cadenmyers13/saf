import numpy as np
from scipy.signal import argrelextrema

from saf.filters import n_fold_filter


def _get_correct_period_indices(overlap_score, n_folds):
    """Get the correct period indices based on the maximum value in the loss
    values.

    This function is used to find the correct period indices based on the
    maximum value in the loss values. If only one max index is found, it
    adds adjacent indices at -360/`n_folds` or +360/`n_folds`. The function
    returns the sorted indices of the selected period.
    Args:
        overlap_score (list): List of loss values.
        n_folds (int): Number of folds in the periodicity.
    Returns:
        list: Sorted list of selected period indices.
    """
    OFFSET_ADJUSTMENT = int(360 / n_folds)
    indices = [
        i for i, loss in enumerate(overlap_score) if loss == max(overlap_score)
    ]
    if len(indices) > 1:
        return sorted(indices[:2])
    index = indices[0]
    adjacent_indices = [
        i
        for i in [index + OFFSET_ADJUSTMENT, index - OFFSET_ADJUSTMENT]
        if 0 <= i < len(overlap_score)
    ]
    return sorted([index] + adjacent_indices)


def compute_loss_near_offset1(intensity, prev_offset, n_folds, k):
    """
    Compute the loss, its first derivative, and second derivative for a range of offsets
    near a given previous offset, based on the intensity and periodicity of the image.
    Args:
        intensity (numpy.ndarray): The intensity matrix of the image.
        prev_offset (float): The previous offset value in radians.
        n_folds (int): The number of folds or periodic repetitions in the image.
        k (float): A parameter used in the `n_fold_filter` function.
    Returns:
        tuple: A tuple containing:
            - offset_values (numpy.ndarray): Array of offset values within the selected range.
            - overlap_score (numpy.ndarray): Array of computed loss values for each offset.
            - first_derivative (numpy.ndarray): Array of first derivative values of the loss.
            - second_derivative (numpy.ndarray): Array of second derivative values of the loss.
    """
    # Define the periodic range and step size
    imshape = intensity.shape
    angle_period = np.deg2rad(360 / n_folds)
    step_size_radians = np.deg2rad(1)
    offset_values = np.arange(
        prev_offset - angle_period,
        prev_offset + angle_period,
        step_size_radians,
    )
    overlap_score = []
    for offset in offset_values:
        evaluate_image_theta = n_fold_filter(
            k=k,
            offset=offset,
            n_folds=n_folds,
            imshape=imshape,
            cx=imshape[1] // 2,
            cy=imshape[0] // 2,
        )
        loss = -(intensity * evaluate_image_theta).sum()
        overlap_score.append(loss.item())
    first_derivative = np.gradient(overlap_score)
    second_derivative = np.gradient(first_derivative)
    selected_indices = _get_correct_period_indices(overlap_score, n_folds)

    offset_values = np.array(
        offset_values[selected_indices[0] : selected_indices[1] + 1]
    )
    overlap_score = np.array(
        overlap_score[selected_indices[0] : selected_indices[1] + 1]
    )
    first_derivative = np.array(
        first_derivative[selected_indices[0] : selected_indices[1] + 1]
    )
    second_derivative = np.array(
        second_derivative[selected_indices[0] : selected_indices[1] + 1]
    )

    return offset_values, overlap_score, first_derivative, second_derivative


def get_approx_offset_values(offset_values, second_derivative):
    """Identify approximate offset positions by finding local maxima and
    inflection points in the second derivative.

    Args:
        offset_values (array-like): A list or array of offset values.
        second_derivative (array-like): The second derivative values corresponding
            to the offset values.
    Returns:
        list: A list containing up to two approximate offset values corresponding
        to the highest local maxima in the second derivative.
    """

    """"""
    local_maxima_indices = argrelextrema(second_derivative, np.greater)[0]
    # Remove indices close to the edges
    filtered_indices = local_maxima_indices[
        (local_maxima_indices >= 5)
        & (local_maxima_indices <= len(second_derivative) - 6)
    ]
    indices_info = [
        (index, offset_values[index], second_derivative[index])
        for index in filtered_indices
    ]
    # Sort by second derivative values (descending) and return top two offsets
    sorted_maxima = sorted(indices_info, key=lambda x: x[2], reverse=True)
    approx_offset_values = [offset for _, offset, _ in sorted_maxima][:2]
    return approx_offset_values
