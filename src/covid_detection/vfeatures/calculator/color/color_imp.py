import numpy as np
from uib_vfeatures.color import Color

from covid_detection.vfeatures.calculator.color.color import ColorVFeaturesCalculator


class ColorFeatureCalculatorImp(ColorVFeaturesCalculator):

    def calculate_features(self,
                           image: np.ndarray,
                           mask: np.ndarray,
                           n_colors: int) -> dict:
        features = dict()

        features['mean_sdv'] = Color.mean_sdv(image)
        features['mean_sdv_rgb'] = Color.mean_sdv_rgb(image)
        features['color_bins'] = Color.color_bins(image,
                                                  mask,
                                                  n_colors)
        return features
