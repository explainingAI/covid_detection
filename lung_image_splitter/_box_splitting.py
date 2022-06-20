"""
This module contains the logic to split a lung box into multiple smallest boxes each one
of the same size
"""

import numpy as np


def split_box(box: np.ndarray, n: int) -> np.ndarray:
    """
    Splits a lung box into multiple smallest boxes each one
    of the same size
    Args:
        box: lung box
        n: number of slices to perform

    Returns:
        Sliced boxes
    """

    lower = box[1][1]
    upper = box[0][1]
    rightmost = box[1][0]
    leftmost = box[0][0]

    height_jump = round((lower - upper) / n)
    lung_rectangles_points = []

    for i in range(n):
        y_1 = upper + height_jump * i
        y_2 = upper + height_jump * (i + 1)

        lung_rectangles_points.append([[leftmost, y_1],
                                       [rightmost, y_2]])
    return np.array(lung_rectangles_points)
