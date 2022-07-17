import numpy as np
import pathlib
import cv2
import json

from uiblungs.box.box import ImageBox
from uiblungs.separator.separator import LungSeparator
from uiblungs.slicer.slicer import BoxSlicer
from uiblungs.cropper.cropper import BoxCropper

from uiblungs.separator.separator_imp import LungSeparatorImp
from uiblungs.slicer.slicer_imp import BoxSlicerImp
from uiblungs.cropper.cropper_imp import BoxCropperImp


class Splitter:
    def __init__(self,
                 separator: LungSeparator = LungSeparatorImp(),
                 slicer: BoxSlicer = BoxSlicerImp(),
                 cropper: BoxCropper = BoxCropperImp()):
        self._separator = separator
        self._slicer = slicer
        self._cropper = cropper

    def get_split_image_boxes(self,
                              mask: np.ndarray) -> dict[str, ImageBox]:
        return self._separator.get_boxes(mask)

    def split(self,
              image: np.ndarray,
              mask: np.ndarray) -> dict[str, np.ndarray]:
        boxes = self.get_split_image_boxes(mask)

        return {'left': self._cropper.crop(image, boxes['left']),
                'right': self._cropper.crop(image, boxes['right'])}

    def save_split_lungs(self,
                         image: np.ndarray,
                         mask: np.ndarray,
                         path: pathlib.Path | str):
        split = self.split(image, mask)

        if type(path) is not pathlib.Path:
            path = pathlib.Path(path)

        path.mkdir(parents=True, exist_ok=True)
        left_path = str(path.joinpath('left.jpg'))
        right_path = str(path.joinpath('right.jpg'))

        cv2.imwrite(left_path, split['left'])
        cv2.imwrite(right_path, split['right'])

    def save_split_coordinates_json(self,
                                    mask: np.ndarray,
                                    path: pathlib.Path | str):
        split = self.get_split_image_boxes(mask)

        data = {'right': split['right'].to_dict(),
                'left': split['left'].to_dict()}

        if type(path) is not pathlib.Path:
            pathlib.Path(path)

        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w+') as f:
            json.dump(data, f)

    def get_slices_boxes(self,
                         mask: np.ndarray,
                         n: int) -> dict[str, dict[int, ImageBox]]:
        split = self.get_split_image_boxes(mask)

        return {'left': self._slicer.slice(split['left'], n),
                'right': self._slicer.slice(split['right'], n)}

    def slice_lungs(self,
                    image: np.ndarray,
                    mask: np.ndarray,
                    n: int = 4) -> dict[str, dict[int, np.ndarray]]:
        slices_boxes = self.get_slices_boxes(mask, n)

        left_slices = dict()
        right_slices = dict()

        for i, box in slices_boxes['left'].items():
            left_slices[i] = self._cropper.crop(image, box)

        for i, box in slices_boxes['right'].items():
            right_slices[i] = self._cropper.crop(image, box)

        return {'left': left_slices,
                'right': right_slices}

    def save_sliced_lungs(self,
                          image: np.ndarray,
                          mask: np.ndarray,
                          n: int,
                          path: pathlib.Path | str):
        if type(path) is not pathlib.Path:
            path = pathlib.Path(path)

        path.mkdir(parents=True, exist_ok=True)

        for position, slices in self.slice_lungs(image, mask, n).items():
            pos_path = path.joinpath(position)
            pos_path.mkdir(exist_ok=True, parents=True)

            for i, lung_slice in slices.items():
                cv2.imwrite(str(pos_path.joinpath(f"{i}_slice.jpg")), lung_slice)

    def save_slice_coordinates_json(self,
                                    mask: np.ndarray,
                                    n: int,
                                    path: pathlib.Path | str):
        if type(path) is not pathlib.Path:
            path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        sliced = self.get_slices_boxes(mask, n)

        data = {
            'left': {f"slice_{i}": slice_.to_dict()
                     for i, slice_ in sliced['left'].items()},
            'right': {f"slice_{i}": slice_.to_dict()
                      for i, slice_ in sliced['right'].items()}
        }

        with open(path, 'w+') as f:
            json.dump(data, f)
