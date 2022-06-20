"""
Script to split a lungs image with a given black and white mask

Args:
    1: mask folder
    2: image folder
    3: output folder
    4: (Optional) number of cuts. default value = 4
"""
import pathlib
import numpy as np
import cv2
from sys import argv

from _lung_boxes_calculation import get_lung_boxes_coordinates
from _box_splitting import split_box


def generate_images(mask: np.ndarray,
                    image: np.ndarray,
                    output_file_path: str,
                    cuts: int):
    boxes = get_lung_boxes_coordinates(mask)

    for side, box in boxes.items():
        slices = split_box(box, cuts)

        for i, slice in enumerate(slices):
            roi = image[slice[0][1]:slice[1][1],
                  slice[0][0]:slice[1][0]]
            cv2.imwrite(f"{output_file_path}_{side}_part_{i}.png", roi)


if __name__ == "__main__":
    mask_folder = pathlib.Path(argv[1]).resolve()
    image_folder = pathlib.Path(argv[2]).resolve()
    output_folder = pathlib.Path(argv[3]).resolve()

    if not image_folder.exists() or not image_folder.is_dir():
        raise ValueError("Image folder is not valid")

    if not mask_folder.exists() or not mask_folder.is_dir():
        raise ValueError("Mask folder is not valid")

    output_folder.mkdir(parents=True, exist_ok=True)

    try:
        cuts = argv[4]
    except IndexError:
        cuts = 4

    for i, (mask_path, image_path) in enumerate(zip(
            sorted((mask_folder.glob("*"))),
            sorted(image_folder.glob("*")))):
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

        generate_images(mask, image, f"{output_folder}/image_{i}", cuts)

        print(f"image{i} processed")
