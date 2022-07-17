from dataclasses import dataclass


@dataclass
class ImageBox:
    upper: int
    lower: int

    rightmost: int
    leftmost: int

    def __post_init__(self):
        self.upper = int(self.upper)
        self.lower = int(self.lower)
        self.rightmost = int(self.rightmost)
        self.leftmost = int(self.leftmost)

    def get_upper_left_corner(self) -> list:
        return [self.upper, self.rightmost]

    def get_lower_right_corner(self) -> list:
        return [self.leftmost, self.lower]

    def to_dict(self):
        return {
            'upper_left': {
                'x': self.leftmost,
                'y': self.upper
            },
            'lower_right': {
                'x': self.rightmost,
                'y': self.lower
            }
        }
