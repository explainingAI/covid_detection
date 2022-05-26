import cv2 as cv
from pathlib import Path
from typing import Union
from numpy.typing._array_like import ndarray

WHITE_BOUNDARY = 128  # Black and white values from 0 to 256


def _get_upper_white_pixel(mask: ndarray) -> tuple[int, int]:
    for i, row in enumerate(mask):
        for j, pixel in enumerate(row):
            if pixel >= WHITE_BOUNDARY:
                return i, j


def _get_lowest_white_pixel(mask: ndarray) -> tuple[int, int]:
    for i, row in enumerate(mask[::-1]):
        for j, pixel in enumerate(row[::-1]):
            if pixel >= WHITE_BOUNDARY:
                return len(mask) - i, len(mask[0]) - j


def _get_leftmost_white_pixel(mask: ndarray) -> tuple[int, int]:
    for j in range(len(mask[0])):
        for i in range(len(mask)):
            if mask[i][j] >= WHITE_BOUNDARY:
                return i, j


def _get_rightmost_white_pixel(mask: ndarray) -> tuple[int, int]:
    for j in reversed(range(len(mask[0]))):
        for i in range(len(mask)):
            if mask[i][j] >= WHITE_BOUNDARY:
                return i, j


def _get_first_white_pixel_from_column(mask: ndarray, column: int) -> tuple[int, int]:
    """
    Get the first white pixel of a given column

    Args:
        mask: numpy array representing the mask image
        column: the column to scan

    Returns:
        The first white pixel coordinates on the given column or None if there aren't
            white pixel on the column
    """
    for i, row in enumerate(mask):
        if row[column] >= WHITE_BOUNDARY:
            return i, column
    return None


def _get_left_lung_rightmost_pixel(mask: ndarray) -> tuple[int, int]:
    pixel = list(_get_leftmost_white_pixel(mask))
    while True:
        if mask[pixel[0]][pixel[1] + 1] >= WHITE_BOUNDARY:  # Check the right pixel
            pixel[1] += 1
        else:  # If there are a white pixel in the next column and row, scan all the rows of the next column
            if (aux_pixel := _get_first_white_pixel_from_column(mask, pixel[1] + 1)) is not None:
                pixel = list(aux_pixel)
            else:
                break

    return tuple(pixel)


def _get_right_lung_leftmost_pixel(mask: ndarray) -> tuple[int, int]:
    pixel = list(_get_rightmost_white_pixel(mask))
    while True:
        if mask[pixel[0]][pixel[1] - 1] >= WHITE_BOUNDARY:  # Check the right pixel
            pixel[1] -= 1
        else:  # If there are a white pixel in the next column and row, scan all the rows of the next column
            if (aux_pixel := _get_first_white_pixel_from_column(mask, pixel[1] - 1)) is not None:
                pixel = list(aux_pixel)
            else:
                break
    return tuple(pixel)


def _resize_widths(left_width: int, right_width: int, points_to_resize: int) -> (int, int):
    """

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


def _resize_to_biggest_width(left_leftmost: int, left_rightmost: int,
                             right_leftmost: int, right_rightmost: int) -> (int, int, int, int):
    """
    The function equals left and right with values to be equals to the biggest ones. To do this,
         adds the extra size to the
    Args:
        left_leftmost:  left leftmost pixel
        left_rightmost: left rightmost pixel
        right_leftmost: right leftmost pixel
        right_rightmost: right rightmost pixel

    Returns:
        The pixels resized to the biggest width
    """
    if (left_width := left_rightmost - left_leftmost) > (right_width := right_rightmost - right_leftmost):
        right_leftmost, right_rightmost = _resize_widths(right_leftmost, right_rightmost, left_width - right_width)
    else:
        left_leftmost, left_rightmost = _resize_widths(left_leftmost, left_rightmost, right_width - left_width)

    return left_leftmost, left_rightmost, right_leftmost, right_rightmost


def get_rectangles_coordinates(mask_path: Union[str, Path], height_cuts: int = 4) -> \
        list[tuple[tuple[int, int], tuple[int, int]]]:
    """
    Gets the coordinates of the n rectangles of a lungs image mask (being n the height_cuts)
    Args:
        mask_path: Path where the mask image is stores
        height_cuts: Number of cuts in height that the function will perform to the mask.

    Returns:
        A  coordinates of the upper left and the downer right pixels of each rectangle extracted from the mask.

    Notes:
        The order of the output is: first upper left, first upper right, second upper left, second upper right,
            etc.

    """
    if type(mask_path) == str:
        mask_path = Path(mask_path)

    mask = cv.imread(str(mask_path.resolve()), cv.IMREAD_GRAYSCALE)

    upper = _get_upper_white_pixel(mask)[0]
    lowest = _get_lowest_white_pixel(mask)[0]

    left_rightmost = _get_left_lung_rightmost_pixel(mask)[1]
    left_leftmost = _get_leftmost_white_pixel(mask)[1]

    right_leftmost = _get_right_lung_leftmost_pixel(mask)[1]
    right_rightmost = _get_rightmost_white_pixel(mask)[1]

    left_leftmost, left_rightmost, right_leftmost, right_rightmost = _resize_to_biggest_width(
        left_leftmost, left_rightmost, right_leftmost, right_rightmost
    )
    # Rectangles
    height_jump = round((lowest - upper) / height_cuts)
    lung_rectangles_points = []
    for i in range(height_cuts):
        y_1 = upper + height_jump * i
        y_2 = upper + height_jump * (i + 1)

        # Left lung
        lung_rectangles_points.append(((left_leftmost, y_1),
                                       (left_rightmost, y_2)))

        # right lung
        lung_rectangles_points.append(((right_leftmost, y_1),
                                       (right_rightmost, y_2)))

    return lung_rectangles_points
