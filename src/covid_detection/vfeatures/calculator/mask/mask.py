import abc

import numpy as np


class MaskVFeaturesCalculator(abc.ABC):

    @abc.abstractmethod
    def calculate_features(self,
                           mask: np.ndarray) -> dict:
        ...
