import abc

import numpy as np


class ColorVFeaturesCalculator(abc.ABC):

    @abc.abstractmethod
    def calculate_features(self,
                           image: np.ndarray,
                           mask: np.ndarray,
                           n_colors: int) -> dict:
        ...
