"""
Script to split a lungs image with a given black and white mask

Args:
    1: mask folder
    2: image folder
    3: output folder
    4: (Optional) number of cuts. default value = 4
"""

from _coordinates_calculation import get_rectangles_coordinates
import cv2
import glob
from sys import argv


def main(mask_folder: str, image_folder: str, output_folder: str, cuts: int):
    for i, (mask_path, image_path) in enumerate(
            zip(
                sorted(glob.glob(mask_folder + "/*")),
                sorted(glob.glob(image_folder + "/*")))):
        coordinates = get_rectangles_coordinates(mask_path, cuts)
        for j in range(len(coordinates)):
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            roi = image[coordinates[j][0][1]:coordinates[j][1][1], coordinates[j][0][0]:coordinates[j][1][0]]
            cv2.imwrite(f"{output_folder}/image_{i}_part_{j}.png", roi)
        print(f"image{i} processed")


if __name__ == "__main__":
    mask_folder = argv[1]
    image_folder = argv[2]
    output_folder = argv[3]

    try:
        cuts = argv[4]
    except IndexError:
        cuts = 4

    main(mask_folder, image_folder, output_folder, cuts)
