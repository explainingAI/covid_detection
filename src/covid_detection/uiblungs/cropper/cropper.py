import abc

import numpy as np

from covid_detection.uiblungs.box.box import ImageBox


class BoxCropper(abc.ABC):

    def crop(self, image: np.ndarray, box: ImageBox) -> np.ndarray:
        ...
