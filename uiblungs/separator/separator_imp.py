import numpy as np
from copy import deepcopy
import cv2

from uiblungs.separator.separator import LungSeparator
from uiblungs.box.box import ImageBox


class LungSeparatorImp(LungSeparator):

    def get_boxes(self, mask: np.ndarray) -> dict[str, ImageBox]:
        contours = self._get_contours(mask)
        left_raw_box = self._get_raw_boxes(contours['left'])
        right_raw_box = self._get_raw_boxes(contours['right'])
        left_normalized, right_normalized = self._normalize_boxes(
            left_raw_box, right_raw_box)

        return {'left': left_normalized, 'right': right_normalized}

    @staticmethod
    def _get_contours(image: np.ndarray) -> dict[str, np.ndarray]:
        contours, _ = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        if len(contours) > 2:
            raise ValueError("More than two lungs")
        if len(contours) == 1:
            raise ValueError("Only one lung detected")

        # Order the lungs putting first the left one and then the right one
        l_contour, r_contour = sorted(contours, key=lambda x: min(x[:, 0, 0]))

        return {"left": l_contour, "right": r_contour}

    @staticmethod
    def _get_raw_boxes(contour: np.ndarray) -> ImageBox:
        upper_value = min(min(contour[:, 0, 1]), min(contour[:, 0, 1]))
        lower_value = max(max(contour[:, 0, 1]), max(contour[:, 0, 1]))

        leftmost_value = min(contour[:, 0, 0])
        rightmost_value = max(contour[:, 0, 0])

        return ImageBox(upper=upper_value,
                        lower=lower_value,
                        rightmost=rightmost_value,
                        leftmost=leftmost_value)

    def _normalize_boxes(self,
                         box1: ImageBox,
                         box2: ImageBox) -> tuple[ImageBox, ImageBox]:

        box1, box2 = self._normalize_widths(box1, box2)
        box1, box2 = self._normalize_height(box1, box2)

        return box1, box2

    def _normalize_widths(self,
                          box1: ImageBox,
                          box2: ImageBox) -> tuple[ImageBox, ImageBox]:
        width1 = box1.rightmost - box1.leftmost
        width2 = box2.rightmost - box2.leftmost

        if width1 > width2:
            resized1 = deepcopy(box1)
            resized2 = self._resize_width(box2, width1 - width2)

        elif width1 < width2:
            resized1 = self._resize_width(box1, width2 - width1)
            resized2 = deepcopy(box2)

        else:
            resized1 = deepcopy(box1)
            resized2 = deepcopy(box2)

        return resized1, resized2

    def _normalize_height(self,
                          box1: ImageBox,
                          box2: ImageBox) -> tuple[ImageBox, ImageBox]:
        height1 = box1.lower - box1.upper
        height2 = box2.lower - box2.upper

        if height1 > height2:
            resized1 = deepcopy(box1)
            resized2 = self._resize_height(box2, height1 - height2)

        elif height1 < height2:
            resized1 = self._resize_height(box1, height2 - height1)
            resized2 = deepcopy(box2)

        else:
            resized1 = deepcopy(box1)
            resized2 = deepcopy(box2)

        return resized1, resized2

    @staticmethod
    def _resize_width(box: ImageBox, units: int) -> ImageBox:

        new_r = int(box.rightmost + units / 2 + units % 2)
        new_l = int(box.leftmost - units / 2 + units % 2)

        return ImageBox(upper=box.upper,
                        lower=box.lower,
                        rightmost=new_r,
                        leftmost=new_l)

    @staticmethod
    def _resize_height(box: ImageBox, units: int) -> ImageBox:

        new_u = int(box.upper - units / 2 + units % 2)
        new_l = int(box.lower + units / 2 + units % 2)

        return ImageBox(upper=new_u,
                        lower=new_l,
                        rightmost=box.rightmost,
                        leftmost=box.leftmost)
