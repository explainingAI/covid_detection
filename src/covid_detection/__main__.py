import argparse
import copy
import json
import sys

import cv2
import numpy as np

from covid_detection.uiblungs.cropper.cropper_imp import BoxCropperImp
from covid_detection.uiblungs.separator.separator_imp import LungSeparatorImp
from covid_detection.uiblungs.slicer.slicer_imp import BoxSlicerImp
from covid_detection.uiblungs.splitter import Splitter
from covid_detection.vfeatures.calculator.color.color_imp import \
    ColorFeatureCalculatorImp
from covid_detection.vfeatures.calculator.mask.mask_imp import \
    MaskFeatureCalculatorImp
from covid_detection.vfeatures.calculator.texture.texture_imp import \
    TextureFeatureCalculatorImp
from covid_detection.vfeatures.vfeatures_calculator import VFeaturesCalculator


class Main:

    def __init__(self, spliter: Splitter):
        self.splitter = spliter

        parser = argparse.ArgumentParser()
        parser.add_argument('command', help="""
        Subcommand to run. Avaliable subcommands:
            - split image in two (one image for each lung)
            - slice slice split images in n slices
            - calculate_features calculate the v-features of the image
        """)
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print('Unrecognized command')
            parser.print_help()
            exit(1)

        getattr(self, args.command)()

    def split(self):
        parser = argparse.ArgumentParser(
            description='split image in two (one image for each lung)')

        parser.add_argument('image_path', help="Path of lungs image")
        parser.add_argument('mask_path', help="Path of the mask image")
        parser.add_argument('output_path', help="Path of the output folder")

        parser.add_argument('--output_mode', help="""
        Supported modes:
            csv: saves a csv with the coordinates of the split boxes
            jpg: saves the jpg images
        """,
                            default='jpg')

        args = parser.parse_args(sys.argv[2:])
        image = self._read_image(args.image_path)
        mask = self._convert_to_binary(self._read_image(args.mask_path))
        mask[mask < 128] = 0
        mask[mask >= 128] = 1
        output_path = args.output_path
        mode = args.output_mode

        match mode:
            case 'jpg':
                self.splitter.save_split_lungs(image, mask, output_path)
            case 'csv':
                self.splitter.save_split_coordinates_json(mask, output_path)
            case _:
                print("Invalid output_mode")
                parser.print_help()
                exit()

    def slice(self):
        parser = argparse.ArgumentParser(
            description='slice split images in n slices')

        parser.add_argument('image_path', help="Path of lungs image")
        parser.add_argument('mask_path', help="Path of the mask image")
        parser.add_argument('output_path', help="Path of the output folder")
        parser.add_argument('--n_slices', help="Number of slices for each"
                                               " lung", default=4)

        parser.add_argument('--output_mode', help="""
                Supported modes:
                    csv: saves a csv with the coordinates of the split boxes
                    jpg: saves the jpg images
                """,
                            default='jpg')

        args = parser.parse_args(sys.argv[2:])
        image = self._read_image(args.image_path)
        mask = self._convert_to_binary(self._read_image(args.mask_path))
        mask[mask < 128] = 0
        mask[mask >= 128] = 1
        n_slices = int(args.n_slices)
        output_path = args.output_path
        mode = args.output_mode
        match mode:
            case 'jpg':
                self.splitter.save_sliced_lungs(image,
                                                mask,
                                                n_slices,
                                                output_path)
            case _:
                print("Invalid output_mode")
                parser.print_help()
                exit()

    def calculate_features(self):
        parser = argparse.ArgumentParser(
            description='calculate_features calculate '
                        'the v-features of the image')

        parser.add_argument('image_path', help="Path of lungs image")
        parser.add_argument('mask_path', help="Path of the mask image")
        parser.add_argument('output_path', help="Path of the output folder")
        parser.add_argument('--n_colors', help="Number of colors of an image")
        parser.add_argument('--distances', help="Distances to calculate features",
                            nargs='*')
        parser.add_argument('--angles', help="Distances to calculate features",
                            nargs='*')
        parser.add_argument('--n_slices', help="Number of slices performed "
                                               "for the lung image", default=4)

        args = parser.parse_args(sys.argv[2:])

        image = self._read_image(args.image_path)
        mask = self._convert_to_binary(self._read_image(args.mask_path))
        output_path = args.output_path
        n_colors = int(args.n_colors)
        distances = [*map(float, args.distances)]
        angles = [*map(float, args.angles)]
        n_slices = int(args.n_slices)

        calculator = VFeaturesCalculator(ColorFeatureCalculatorImp(),
                                         MaskFeatureCalculatorImp(),
                                         TextureFeatureCalculatorImp())

        values = calculator.calculate(image,
                                      mask,
                                      n_colors,
                                      distances,
                                      angles,
                                      n_slices)

        with open(output_path, 'w+') as f:
            json.dump(self._to_serializable_dict(values), f)

    @staticmethod
    def _read_image(path: str):
        return cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    @staticmethod
    def _convert_to_binary(mask: np.ndarray) -> np.ndarray:
        mask_ = copy.deepcopy(mask)
        mask_[mask_ < 128] = 0
        mask_[mask_ >= 128] = 1
        return mask_

    def _to_serializable_dict(self, my_dict: dict) -> dict:
        result = dict()
        for key, item in my_dict.items():
            if type(item) == np.ndarray:
                item = item.tolist()
            elif type(item) == dict:
                item = self._to_serializable_dict(item)
            elif type(item) == tuple or type(item) == list:
                item = list(item)
                for i, val in enumerate(item):
                    if type(val) == np.ndarray:
                        item[i] = val.tolist()
                    elif type(val) == dict:
                        item[i] = self._to_serializable_dict(val)

            result[key] = item

        return result


if __name__ == "__main__":
    Main(Splitter(
        separator=LungSeparatorImp(),
        cropper=BoxCropperImp(),
        slicer=BoxSlicerImp()
    ))
