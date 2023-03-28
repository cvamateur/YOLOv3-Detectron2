from typing import Tuple, Union, Optional
import numpy as np
import cv2
import math
import random

from detectron2.data.transforms import (
    Augmentation,
    Transform,
    TransformList,
    ResizeShortestEdge,
    NoOpTransform,
    RandomCrop,
    RandomFlip,
)

from .transform import (
    ResizeTransform,
    PadTransform,
    ShiftPixelTransform,
    PerspectiveTransform,
    JitterCropTransform,
    JitterColorTransform,
    HistEqualizeTransform,
    ToGrayTransform,
)

__all__ = [
    "ResizeShortestEdge",
    "NoOpTransform",
    "RandomCrop",
    "RandomFlip",
    "LetterBox",
    "YOLOJitterShift",
    "YOLOJitterCrop",
    "YOLOJitterResize",
    "YOLOJitterColor",
    "RandomPerspective",
    "RandomHistEqualize",
    "RandomToGray",
]


def _as_tuple_of_n(x, n=3):
    if isinstance(x, (list, tuple, np.ndarray)):
        assert len(x) == n, f"{len(x)=} != {n}"
        return x
    x = (x,) * n
    return x


class LetterBox(Augmentation):
    """
    Resize and pad image while meeting stride-multiple constraints.

    This class does not introduce randomization, it resizes original
    image on the longest edge in order to keep aspect ratio unchanged,
    then pad the resized image to final output size.

    If rect is True, then pad as small as possible and return a
    rectangular image. This mode is often used in inference.

    Args:
        dsize (int | Tuple[int, int]): Final image size width and height.
        stride (int): Divisor of final image size.
        rect (bool): If True, return minimum rectangular image.
        stretch (bool): If True, no padding is applied.
        allow_scale_up (bool): If True, allow resize small image to large one.
        pad_value (int or Tuple[int, ...]): padding color to image.
    """

    def __init__(self,
                 dsize: Union[int, Tuple[int, int]],
                 stride: int = 32,
                 rect: bool = False,
                 stretch: bool = False,
                 allow_scale_up: bool = True,
                 pad_value: Union[int, Tuple[int, ...]] = 114):
        super().__init__()
        self._init(locals())

    def get_transform(self, img: np.ndarray) -> Transform:
        h, w = img.shape[:2]
        dsize = _as_tuple_of_n(self.dsize, 2)
        pad_value = _as_tuple_of_n(self.pad_value, img.ndim)

        # scale ratio on longest edge
        r = self._compute_scale_ratio(w, h, dsize)

        # padding to dsize
        paddings, unpad_wh = self._compute_padding(w, h, r, dsize)

        letterbox = []
        if (w, h) != unpad_wh:
            letterbox.append(ResizeTransform(h, w, unpad_wh[1], unpad_wh[0], cv2.INTER_LINEAR))
        if any(x != 0 for x in paddings):
            letterbox.append(PadTransform(*paddings, *unpad_wh, pad_value))
        return TransformList(letterbox)

    def _compute_scale_ratio(self, w, h, dsize):
        ratio = min(dsize[0] / w, dsize[1] / h)
        if not self.allow_scale_up:
            ratio = min(ratio, 1.0)
        return ratio

    def _compute_padding(self, w, h, r, dsize):
        unpad_w, unpad_h = int(round(w * r)), int(round(h * r))
        dw, dh = dsize[0] - unpad_w, dsize[1] - unpad_h
        if self.rect:
            dw %= self.stride
            dh %= self.stride
        elif self.stretch:
            dw = dh = 0.0
            unpad_w, unpad_h = dsize
        top, bottom = int(round(dh / 2.0 - 0.1)), int(round(dh / 2.0 + 0.1))
        left, right = int(round(dw / 2.0 - 0.1)), int(round(dw / 2.0 + 0.1))
        return (left, top, right, bottom), (unpad_w, unpad_h)


class YOLOJitterShift(Augmentation):
    """
    Randomly shift image left/right and top/down by max_shift pixels,
    the pixels shifted out will be padded by `value`.

    Args:
        p (float): Probability to do shifting.
        max_shift (int): Maximum number of pixels to shift.
        value (int, Tuple[int, ...]): Pad value for shifted out pixels.
    """

    def __init__(self,
                 max_shift: int = 32,
                 border_value: Union[int, Tuple[int, ...]] = 114):
        self._init(locals())

    def get_transform(self, img: np.ndarray) -> Transform:
        max_shift = self.max_shift
        shift_x, shift_y = np.random.randint(-max_shift, max_shift, 2)
        border_value = _as_tuple_of_n(self.border_value, img.ndim)
        return ShiftPixelTransform(int(shift_x), shift_y, border_value)


class YOLOJitterCrop(Augmentation):
    """
    JitterCrop augmentation in YOLOv4.

    Args:
        jitter_ratio (float): ratio of crop.
    """

    def __init__(self,
                 jitter_ratio: float = 0.0,
                 border_value: Union[int, Tuple[int, ...]] = 114):
        self._init(locals())

    def get_transform(self, img: np.ndarray) -> Transform:
        h, w = img.shape[:2]
        r = self.jitter_ratio
        dw, dh = int(r * w), int(r * h)
        left, right = np.random.randint(-dw, dw, 2)
        top, bottom = np.random.randint(-dh, dh, 2)
        dst_w = w - left - right
        dst_h = h - top - bottom
        border_value = _as_tuple_of_n(self.border_value, img.ndim)
        return JitterCropTransform(left, top, dst_w, dst_h, border_value)


class YOLOJitterResize(Augmentation):
    """
    Resize in YOLOF.

    Args:
        dsize (int | Tuple[int, int]): Target size or (width, height).
        jitter_ratio (float | Tuple[float, float]): None or 0.2 or (0.8, 1.2)
    """

    def __init__(self,
                 dsize: Union[int, Tuple[int, int]],
                 jitter_ratio: Union[float, Tuple[float, float]] = None,
                 interp: int = cv2.INTER_LINEAR):
        if isinstance(dsize, int):
            dsize = (dsize, dsize)
        if isinstance(jitter_ratio, (int, float)):
            assert abs(jitter_ratio) < 1.0, "jitter_ratio in [0, 1)"
            jitter_ratio = (1.0 - jitter_ratio, 1.0 + jitter_ratio)
        self._init(locals())

    def get_transform(self, img: np.ndarray) -> Transform:
        h, w = img.shape[:2]
        r = self.jitter_ratio
        s = self.dsize
        if r is not None:
            r = np.random.uniform(r[0], r[1])
            s = (int(s[0] * r), int(s[1] * r))
        return ResizeTransform(h, w, s[1], s[0], self.interp)


class YOLOJitterColor(Augmentation):
    """
    Randomly do Color Jittering on image.

    Args:
        jitter_h (float): Image HSV-Hue augmentation (fraction)
        jitter_s (float): Image HSV-Saturation augmentation (fraction)
        jitter_v (float): Image HSV-Value augmentation (fraction)
    """

    def __init__(self,
                 jitter_h: float = 0.015,
                 jitter_s: float = 0.7,
                 jitter_v: float = 0.4):
        super().__init__()
        self._init(locals())

    def get_transform(self, img: np.ndarray) -> Transform:
        hsv_ratio = self.jitter_h, self.jitter_s, self.jitter_v
        if any(x != 0.0 for x in hsv_ratio):
            hsv_gain = self._rand_range(-1.0, 1.0, 3) * hsv_ratio + 1.0
            return JitterColorTransform(hsv_gain[0], hsv_gain[1], hsv_gain[2])
        else:
            return NoOpTransform()


class RandomPerspective(Augmentation):
    """
    This class do random perspective transform on image, and
    returns a copy of it. If perspective is 0.0, then affine
    transform is used.

    Args:
        translate (float): Translation on x and y axis (+/- fraction).
        scale (float): Scale of the affine transform (+/- gain).
        degrees (float): Angle in degrees to rotate counter-clockwise (+/- deg).
        sheer (float): Angle in degrees to sheer on x and y axis (+/- deg).
        perspective (float): Perspective transform, (+/- fraction), range 0-0.001.
        border_value (int or Tuple[int, int, int]): Border value of transformation.
    """

    def __init__(self,
                 translate: float = 0.1,
                 scale: float = 0.25,
                 degrees: float = 0.,
                 sheer: float = 0.,
                 perspective: float = 0.0,  # in range [0, 0.0001]
                 border_value: Union[int, Tuple[int, ...]] = 114):
        self._init(locals())

    def get_transform(self, img: np.ndarray) -> Transform:
        h, w = img.shape[:2]
        translate = self.translate
        scale = self.scale
        degrees = self.degrees
        sheer = self.sheer
        perspective = self.perspective

        # Perspective
        P = np.eye(3)
        P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
        P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

        # rotation, scale
        R = np.eye(3, dtype=np.float32)
        a = random.uniform(-degrees, degrees)
        s = random.uniform(1 - scale, 1 + scale)
        R[:2] = cv2.getRotationMatrix2D((w // 2, h // 2), angle=a, scale=s)

        # sheer
        S = np.eye(3, dtype=np.float32)
        S[0, 1] = math.tan(random.uniform(-sheer, sheer) * math.pi / 180.)  # x sheer (deg)
        S[1, 0] = math.tan(random.uniform(-sheer, sheer) * math.pi / 180.)  # y sheer (deg)

        # translate
        T = np.eye(3, dtype=np.float32)
        T[0, 2] = w * random.uniform(-translate, translate)
        T[1, 2] = h * random.uniform(-translate, translate)

        # combined all matrix
        M = T @ S @ R @ P  # the order MATTERS!
        affine_only = (perspective == 0.0)
        border_value = _as_tuple_of_n(self.border_value, img.ndim)
        return PerspectiveTransform(w, h, M, affine_only, border_value)


class RandomHistEqualize(Augmentation):
    """
    Args:
        p (float): Probability.
        p_clahe (bool): Probability to do CLAHE.
        rgb (bool): Whether input image is a RGB image.
    """

    def __init__(self,
                 p: float = 0.0,
                 p_clahe: float = 1.0,
                 rgb: bool = False):
        super().__init__()
        self._init(locals())

    def get_transform(self, img: np.ndarray) -> Transform:
        if self._rand_range() >= self.p:
            return NoOpTransform()
        clahe = self._rand_range() < self.p_clahe
        return HistEqualizeTransform(clahe, self.rgb)


class RandomToGray(Augmentation):

    def __init__(self,
                 p: float = 0.0,
                 rgb: bool = False):
        super().__init__()
        self._init(locals())

    def get_transform(self, img: np.ndarray) -> Transform:
        if self._rand_range() >= self.p:
            return NoOpTransform()
        return ToGrayTransform(self.rgb)
