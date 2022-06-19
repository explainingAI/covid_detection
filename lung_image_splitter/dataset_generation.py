"""
Script to split a lungs image with a given black and white mask

Args:
    1: mask folder
    2: output filepath
    3: (Optional) number of cuts. default value = 4
"""
import pathlib
import numpy as np
import cv2
from sys import argv
import pandas as pd

from _lung_boxes_calculation import get_lung_boxes_coordinates
from _box_splitting import split_box


def calculate_boxes(mask: np.ndarray,
                    cuts: int = 4) -> list[int]:
    final_boxes = []
    lung_boxes = get_lung_boxes_coordinates(mask)
    for lung_box in lung_boxes.values():
        boxes_split = split_box(lung_box, cuts)
        for spitted_box in boxes_split:
            final_boxes.append(spitted_box)
    return final_boxes


if __name__ == "__main__":
    mask_folder = pathlib.Path(argv[1]).resolve()
    output_file = pathlib.Path(argv[2]).resolve()

    if not mask_folder.exists() or not mask_folder.is_dir():
        raise ValueError("Mask folder isn't valid")
    try:
        cuts = argv[3]
    except IndexError:
        cuts = 4

    data = []
    for i, mask_path in enumerate(sorted(mask_folder.glob("*"))):
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        data.append(calculate_boxes(mask, cuts))

        print(f"image{i} processed")

    columns = []

    for i in range(cuts):
        columns.append(f"LeftBox{i}")

    for i in range(cuts, cuts * 2):
        columns.append(f"RightBox{i}")

    pd.DataFrame(columns=columns, data=data).to_csv(output_file)
