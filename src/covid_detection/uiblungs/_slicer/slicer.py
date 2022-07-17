import abc

from box.box import ImageBox


class BoxSlicer(abc.ABC):

    @abc.abstractmethod
    def slice(self, box: ImageBox, n: int) -> dict[int, ImageBox]:
        ...
