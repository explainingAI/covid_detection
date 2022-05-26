"""
Script to split a lungs image with a given black and white mask

Args:
    1: mask folder
    2: output folder
    3:(Optional) number of cuts. default value = 4
"""

from _coordinates_calculation import get_rectangles_coordinates
import glob
from sys import argv
import pandas as pd
from pathlib import Path
from functools import reduce


def main(mask_folder: str, output_file: str, cuts: int):
    columns = ["Mask"]

    for i in range(cuts * 2):
        columns.append(f"R{i}_Upp")
        columns.append(f"R{i}_Bottom")

    coordinates = list()
    for i, mask_path in enumerate(sorted(glob.glob(mask_folder + "/*"))):
        coordinates.append(
            (Path(mask_path).name, ) +
            reduce(lambda x, y: x + y, get_rectangles_coordinates(mask_path, cuts)))

        print(f"image{i} processed")

    print("Storing coordinates on dataframe...")

    pd.DataFrame(columns=columns, data=coordinates).to_csv(output_file, index= False)


if __name__ == "__main__":
    mask_folder = argv[1]
    output_folder = argv[2]

    try:
        cuts = argv[3]
    except IndexError:
        cuts = 4

    main(mask_folder, output_folder, cuts)
