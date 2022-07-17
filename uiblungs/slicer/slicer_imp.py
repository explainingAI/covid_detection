import numpy as np

from uiblungs.box.box import ImageBox
from uiblungs.slicer.slicer import BoxSlicer


class BoxSlicerImp(BoxSlicer):

    def slice(self, box: ImageBox, n: int) -> dict[int, ImageBox]:
        if n < 2:
            raise ValueError('n have to be >= 2')

        slice_height = round((box.lower - box.upper) / n)

        boxes = dict()
        for i in range(n):
            upper = box.upper + slice_height * i
            lower = box.upper + slice_height * (i + 1)

            upper = upper + 1 if i != 0 else upper

            boxes[i] = ImageBox(upper=upper,
                                lower=lower,
                                rightmost=box.rightmost,
                                leftmost=box.leftmost)

        return boxes
