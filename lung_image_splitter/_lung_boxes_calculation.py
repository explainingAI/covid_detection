"""
This module contains the logic to get the box coordinates of each lung from a lung mask
"""

import numpy as np
import cv2


def get_lung_boxes_coordinates(mask: np.ndarray) -> dict[str, np.ndarray]:
    """
    Get the upper left and the lower right coordinates of each lung from a binary mask

    Returns:
        A dict with the box coordinates of each lung
    """
    mask[mask < 127] = 0
    mask[mask >= 127] = 1
    l_contours, r_contours = _get_contours(mask)

    upper_height = min(min(l_contours[:, 0, 1]), min(r_contours[:, 0, 1]))
    lower_height = max(max(l_contours[:, 0, 1]), max(r_contours[:, 0, 1]))

    left_leftmost = min(l_contours[:, 0, 0])
    left_rightmost = max(l_contours[:, 0, 0])

    right_leftmost = min(r_contours[:, 0, 0])
    right_rightmost = max(r_contours[:, 0, 0])

    left_leftmost, left_rightmost, right_leftmost, right_rightmost = \
        _resize_to_biggest_width(left_leftmost,
                                 left_rightmost,
                                 right_leftmost,
                                 right_rightmost)

    return {
        "left_lung": np.array([[left_leftmost, upper_height],
                               [left_rightmost, lower_height]]),
        "right_lung": np.array([[right_leftmost, upper_height],
                                [right_rightmost, lower_height]])
    }


def _get_contours(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    if len(contours) > 2:
        raise ValueError("More than two lungs")
    if len(contours) == 1:
        raise ValueError("Only one lung detected")

    # Order the lungs putting first the left one and then the right one
    l_contour, r_contour = sorted(contours, key=lambda x: min(x[:, 0, 0]))

    return l_contour, r_contour


def _resize_to_biggest_width(left_leftmost: int,
                             left_rightmost: int,
                             right_leftmost: int,
                             right_rightmost: int) -> (int, int, int, int):
    """
    Resizes the pair (right or left) with the smallest distance between values to
    have the same distance of the other pair.

    Args:
        left_leftmost:  left leftmost pixel
        left_rightmost: left rightmost pixel
        right_leftmost: right leftmost pixel
        right_rightmost: right rightmost pixel

    Returns:
        The pixels resized to the biggest width
    """
    if ((left_width := left_rightmost - left_leftmost) >
            (right_width := right_rightmost - right_leftmost)):
        right_leftmost, right_rightmost = _resize_widths(right_leftmost,
                                                         right_rightmost,
                                                         left_width - right_width)
    else:
        left_leftmost, left_rightmost = _resize_widths(left_leftmost,
                                                       left_rightmost,
                                                       right_width - left_width)

    return left_leftmost, left_rightmost, right_leftmost, right_rightmost


def _resize_widths(left_width: int, right_width: int, points_to_resize: int) -> (
        int, int):
    """
    Increments right_width and decrements left_width points_to_resize/2.

    Notes:
        right_width width will be incremented in one more unit if points_to_resize is
        even
    Args:
        left_width: value of the left point width
        right_width: value of the right point width
        points_to_resize: pixels to resize

    Returns:
        resized values
    """
    right_width_resized = right_width + points_to_resize / 2 + points_to_resize % 2
    left_width_resized = left_width - points_to_resize / 2

    return int(left_width_resized), int(right_width_resized)
