import numpy as np

from covid_detection.uiblungs.cropper.cropper import BoxCropper
from covid_detection.uiblungs.box.box import ImageBox


class BoxCropperImp(BoxCropper):

    def crop(self, image: np.ndarray, box: ImageBox) -> np.ndarray:
        return image[box.upper:box.lower, box.leftmost: box.rightmost]
