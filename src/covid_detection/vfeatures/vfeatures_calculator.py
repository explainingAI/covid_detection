import numpy as np

from covid_detection.uiblungs.splitter import Splitter
from covid_detection.vfeatures.calculator.color.color import ColorVFeaturesCalculator
from covid_detection.vfeatures.calculator.mask.mask import MaskVFeaturesCalculator
from covid_detection.vfeatures.calculator.texture.texture import TextureVFeaturesCalculator


class VFeaturesCalculator:
    def __init__(self,
                 color_calculator: ColorVFeaturesCalculator,
                 mask_calculator: MaskVFeaturesCalculator,
                 texture_calculator: TextureVFeaturesCalculator):
        self.color_calculator = color_calculator
        self.mask_calculator = mask_calculator
        self.texture_calculator = texture_calculator

    def calculate(self,
                  image: np.ndarray,
                  mask: np.ndarray,
                  n_colors: int,
                  distances: list,
                  angles: list,
                  n_slices: int = 4):
        features = dict()
        features['left_lung'] = dict()
        features['right_lung'] = dict()
        features['left_lung_sliced'] = dict()
        features['right_lung_sliced'] = dict()

        splitter = Splitter()

        images = splitter.split(image, mask)
        masks = splitter.split(mask, mask)
        images_slices = splitter.slice_lungs(image, mask, n_slices)
        masks_slices = splitter.slice_lungs(mask, mask, n_slices)
        l_image_slices = images_slices['left']
        r_image_slices = images_slices['right']
        l_mask_slices = masks_slices['left']
        r_mask_slices = masks_slices['right']

        features['left_lung'] = self._calculate(images['left'],
                                                masks['left'],
                                                n_colors,
                                                distances,
                                                angles)

        features['right_lung'] = self._calculate(images['right'],
                                                 masks['right'],
                                                 n_colors,
                                                 distances,
                                                 angles)

        for i in range(n_slices):
            idx = f'slice_{i}'
            features['left_lung_sliced'][idx] = self._calculate(l_image_slices[i],
                                                                l_mask_slices[i],
                                                                n_colors,
                                                                distances,
                                                                angles)
            features['right_lung_sliced'][idx] = self._calculate(r_image_slices[i],
                                                                 r_mask_slices[i],
                                                                 n_colors,
                                                                 distances,
                                                                 angles)

        return features

    def _calculate(self,
                   image: np.ndarray,
                   mask: np.ndarray,
                   n_colors: int,
                   distances: list,
                   angles: list) -> dict[str, dict]:
        calculations = dict()

        calculations['color'] = self.color_calculator \
            .calculate_features(image, mask, n_colors)

        calculations['mask'] = self.mask_calculator.calculate_features(mask)
        calculations['texture'] = self.texture_calculator \
            .calculate_features(image, distances, angles)

        return calculations
