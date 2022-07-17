import numpy as np
from uib_vfeatures.texture import Texture

from covid_detection.vfeatures.calculator.texture.texture import TextureVFeaturesCalculator


class TextureFeatureCalculatorImp(TextureVFeaturesCalculator):

    def calculate_features(self,
                           image: np.ndarray,
                           distances: list,
                           angles: list) -> dict:
        features = dict()

        features['contrast'] = Texture.texture_features(image,
                                                        distances,
                                                        angles,
                                                        ['contrast'])
        features['dissimilarity'] = Texture.texture_features(image,
                                                             distances,
                                                             angles,
                                                             ['dissimilarity'])
        features['homogeneity'] = Texture.texture_features(image,
                                                           distances,
                                                           angles,
                                                           ['homogeneity'])
        features['ASM'] = Texture.texture_features(image,
                                                   distances,
                                                   angles,
                                                   ['ASM'])
        features['energy'] = Texture.texture_features(image,
                                                      distances,
                                                      angles,
                                                      ['energy'])
        features['correlation'] = Texture.texture_features(image,
                                                           distances,
                                                           angles,
                                                           ['correlation'])
        features['skew'] = Texture.skew(image)
        features['kurtosis'] = Texture.kurtosis(image)

        return features
