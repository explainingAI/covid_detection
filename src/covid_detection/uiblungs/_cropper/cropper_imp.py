import numpy as np

from cropper.cropper import BoxCropper
from box.box import ImageBox


class BoxCropperImp(BoxCropper):

    def crop(self, image: np.ndarray, box: ImageBox) -> np.ndarray:
        return image[box.upper:box.lower, box.leftmost: box.rightmost]
