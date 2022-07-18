import numpy as np
from uib_vfeatures.masks import Masks

from covid_detection.vfeatures.calculator.mask.mask import MaskVFeaturesCalculator


class MaskFeatureCalculatorImp(MaskVFeaturesCalculator):

    def calculate_features(self,
                           mask: np.ndarray) -> dict:
        features = dict()

        features['solidity'] = Masks.solidity(mask)
        features['convex_hull_perimeter'] = Masks.convex_hull_perimeter(mask)
        features['convex_hull_area'] = Masks.convex_hull_area(mask)
        features['bounding_box_area'] = Masks.bounding_box_area(mask)
        features['rectangularity'] = Masks.rectangularity(mask)
        features['min_r'] = Masks.min_r(mask)
        features['max_r'] = Masks.max_r(mask)
        features['feret'] = Masks.feret(mask)
        features['breadth'] = Masks.breadth(mask)
        features['circularity'] = Masks.circularity(mask)
        features['roundness'] = Masks.roundness(mask)
        features['feret_angle'] = Masks.feret_angle(mask)
        features['eccentricity'] = Masks.eccentricity(mask)
        features['center'] = Masks.center(mask)
        features['sphericity'] = Masks.sphericity(mask)
        features['aspect_ratio'] = Masks.aspect_ratio(mask)
        features['area_equivalent_diameter'] = Masks.area_equivalent_diameter(mask)
        features['perimeter_equivalent_diameter'] = Masks.perimeter_equivalent_diameter(mask)
        features['equivalent_ellipse_area'] = Masks.equivalent_ellipse_area(mask)
        features['compactness'] = Masks.compactness(mask)
        features['area'] = Masks.area(mask)
        features['convexity'] = Masks.convexity(mask)
        features['shape'] = Masks.shape(mask)
        features['perimeter'] = Masks.perimeter(mask)
        features['extract_contour'] = Masks.extract_contour(mask)

        return features
