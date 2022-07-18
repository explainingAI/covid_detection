import abc

import numpy as np


class TextureVFeaturesCalculator(abc.ABC):

    @abc.abstractmethod
    def calculate_features(self,
                           image: np.ndarray,
                           distances: list,
                           angles: list) -> dict:
        ...
