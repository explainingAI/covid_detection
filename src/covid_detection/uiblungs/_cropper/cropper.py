import abc

import numpy as np

from box.box import ImageBox


class BoxCropper(abc.ABC):

    def crop(self, image: np.ndarray, box: ImageBox) -> np.ndarray:
        ...
