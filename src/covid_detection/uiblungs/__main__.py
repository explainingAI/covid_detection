from splitter import Splitter
from _slicer.slicer_imp import BoxSlicerImp
from _separator.separator_imp import LungSeparatorImp
from _cropper.cropper_imp import BoxCropperImp
import cv2
import sys
import argparse


class Main:

    def __init__(self, spliter: Splitter):
        self.splitter = spliter

        parser = argparse.ArgumentParser()
        parser.add_argument('command', help="""
        Subcommand to run. Avaliable subcommands:
            - split image in two (one image for each lung)
            - slice slice split images in n slices
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
        mask = self._read_image(args.mask_path)
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
        mask = self._read_image(args.mask_path)
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

    @staticmethod
    def _read_image(path: str):
        return cv2.imread(path, cv2.IMREAD_GRAYSCALE)


if __name__ == "__main__":
    Main(Splitter(
        separator=LungSeparatorImp(),
        cropper=BoxCropperImp(),
        slicer=BoxSlicerImp()
    ))
