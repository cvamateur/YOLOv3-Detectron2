import functools
import math

import cv2
import numpy as np
import torch

from typing import Optional, Union, Tuple, List, Dict, Any
from detectron2.data.transforms import (
    Transform,
    TransformList,
    ColorTransform,
    CropTransform,
)

__all__ = [
    "PadTransform",
    "ResizeTransform",
    "PerspectiveTransform",
    "ShiftPixelTransform",
    "JitterCropTransform",
    "JitterColorTransform",
    "HistEqualizeTransform",
    "ToGrayTransform",
]

_DEFAULT_PAD_COLOR = (114, 114, 114)


def _clip_boxes(boxes, w, h):
    boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0, w - 1)
    boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0, h - 1)
    return boxes


class PadTransform(Transform):
    """
    Same as :class: T.PadTransform, but uses opencv.

    Args:
        left, top: number of padded pixels on the left and top.
        right, bottom: number of padded pixels on the right and bottom.
        orig_w, orig_h: optional, original width and height.
            Needed to make this transform invertible.
        pad_value: the padding color to the image.
        seg_pad_value: the padding value to the segmentation mask.
    """

    def __init__(self,
                 x0: int,
                 y0: int,
                 x1: int,
                 y1: int,
                 orig_w: Optional[int] = None,
                 orig_h: Optional[int] = None,
                 pad_value: Tuple[int, ...] = _DEFAULT_PAD_COLOR,
                 seg_pad_value: int = 0):
        super().__init__()
        self._set_attributes(locals())

    def apply_image(self, img: np.ndarray, pad_value=None) -> np.ndarray:
        value = pad_value if pad_value is not None else self.pad_value
        return cv2.copyMakeBorder(
            img, self.y0, self.y1, self.x0, self.x1, cv2.BORDER_CONSTANT, value=value)

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        return self.apply_image(segmentation, self.seg_pad_value)

    def apply_coords(self, coords: np.ndarray):
        coords[:, 0] += self.x0
        coords[:, 1] += self.y0
        return coords

    def inverse(self) -> "Transform":
        assert (
                self.orig_w is not None and self.orig_h is not None
        ), f"orig_w and orig_h are required for PadTransform to be invertible!"
        new_w = self.orig_w + self.x0 + self.x1
        new_h = self.orig_h + self.y0 + self.y1
        return CropTransform(
            self.x0, self.y0, self.orig_w, self.orig_h, new_w, new_h)


class ResizeTransform(Transform):
    """
    Same as :class: T.ResizeTransform but use opencv.

    Args:
        h, w (int): original image size.
        new_h, new_w (int): new image size.
        interp: OpenCV interpolation methods.
    """

    def __init__(self,
                 h: int,
                 w: int,
                 new_h: int,
                 new_w: int,
                 interp: int = cv2.INTER_LINEAR):
        super().__init__()
        self._set_attributes(locals())

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        assert img.shape[:2] == (self.h, self.w), f"{img.shape[:2]} != {(self.h, self.w)}"
        return cv2.resize(img, (self.new_w, self.new_h), interpolation=self.interp)

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        coords[:, 0] *= (self.new_w * 1.0 / self.w)
        coords[:, 1] *= (self.new_h * 1.0 / self.h)
        return coords

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        assert segmentation.shape[:2] == (self.h, self.w)
        return cv2.resize(
            segmentation, (self.new_w, self.new_h), interpolation=cv2.INTER_NEAREST)

    def inverse(self) -> "Transform":
        return ResizeTransform(
            self.new_h, self.new_w, self.h, self.w, self.interp)


class PerspectiveTransform(Transform):
    """
    This class do affine transform or perspective transform on
    an image, returns a copy of it.

    Args:
        w, h (int, int): Output image width and height.
        M (ndarray): Transform matrix.
        affine (bool): Controls transform type.
        border_value (Tuple[int, ...]): border value of each channel.
    """

    def __init__(self,
                 w: int,
                 h: int,
                 M: np.ndarray,
                 affine: bool,
                 border_value: Tuple[int, ...] = _DEFAULT_PAD_COLOR):
        super().__init__()
        self._set_attributes(locals())

    def apply_image(self, img: np.ndarray):
        if self.affine:
            return cv2.warpAffine(img, self.M[:2], (self.w, self.h), borderValue=self.border_value)
        else:
            return cv2.warpPerspective(img, self.M, (self.w, self.h), borderValue=self.border_value)

    def apply_coords(self, coords: np.ndarray):
        self._clip_coords(coords)

        coords_ = np.ones([len(coords), 3], dtype=coords.dtype)
        coords_[:, :2] = coords
        coords_ = coords_ @ self.M.T
        coords_ = coords_[:, :2] if self.affine else coords_[:, :2] / coords_[:, 2:3]
        return coords_

    def _clip_coords(self, coords):
        np.clip(coords[:, 0], 0, self.w - 1, out=coords[:, 0])
        np.clip(coords[:, 1], 0, self.h - 1, out=coords[:, 1])

class ShiftPixelTransform(Transform):
    """
    Shift pixels on x and y axis.
    """

    def __init__(self,
                 shift_x: int,
                 shift_y: int,
                 border_value: Tuple[int, ...] = _DEFAULT_PAD_COLOR):
        super().__init__()
        self._set_attributes(locals())

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2] if img.ndim <= 3 else img.shape[-3:-1]

        sx, sy = self.shift_x, self.shift_y
        if sx < 0:
            x0a, x1a = -sx, w  # old image
            x0b, x1b = 0, w + sx  # new image
        else:
            x0a, x1a = 0, w - sx
            x0b, x1b = sx, w
        if sy < 0:
            y0a, y1a = -sy, h
            y0b, y1b = 0, h + sy
        else:
            y0a, y1a = 0, h - sy
            y0b, y1b = sy, h

        new_img = np.full_like(img, self.border_value)
        if img.ndim <= 3:
            new_img[y0b: y1b, x0b: x1b, ...] = img[y0a: y1a, x0a: x1a]
        else:
            new_img[..., y0b: y1b, x0b: x1b, :] = img[..., y0a: y1a, x0a: x1a, :]
        return new_img

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        coords[:, 0] += self.shift_x
        coords[:, 1] += self.shift_y
        return coords


class JitterCropTransform(Transform):
    """
    The JitterCropTransform in YOLOv4.

    Args:
        x0, y0 (int): Number of offset pixels in left, top.
        w, h (int): Final cropped image width and height.
        value (Tuple[int, ...]): Pixel value of regions outside
            of the cropped image.
    """

    def __init__(self,
                 x0: int,
                 y0: int,
                 w: int,
                 h: int,
                 border_value: Tuple[int, ...] = _DEFAULT_PAD_COLOR):
        super().__init__()
        self._set_attributes(locals())

    def apply_image(self, img: np.ndarray):
        src_h, src_w = img.shape[:2]
        x0, y0, w, h = self.x0, self.y0, self.w, self.h

        # itersection
        ix0, iy0 = max(x0, 0), max(y0, 0)
        ix1, iy1 = min(x0 + w, src_w), min(y0 + h, src_h)

        # destination
        ox0, oy0 = max(-x0, 0), max(-y0, 0)
        ox1, oy1 = ox0 + ix1 - ix0, oy0 + iy1 - iy0

        # copy intersection to destination as same position
        new_img = np.full([h, w, img.ndim], self.border_value, dtype=img.dtype)
        new_img[oy0: oy1, ox0: ox1] = img[iy0: iy1, ix0: ix1]
        return new_img

    def apply_coords(self, coords: np.ndarray):
        coords -= (self.x0, self.y0)
        return coords

    def apply_box(self, box: np.ndarray) -> np.ndarray:
        box = super().apply_box(box)
        box = _clip_boxes(box, self.w, self.h)
        return box


class JitterColorTransform(ColorTransform):

    def __init__(self, hue: float, saturation: float, brightness: float):
        jitter = functools.partial(self._hsv_jitter, hue, saturation, brightness)
        super().__init__(jitter)

    @classmethod
    def _hsv_jitter(cls, h_gain, s_gain, v_gain, img):
        hsv_gain = np.array([h_gain, s_gain, v_gain], dtype=np.float32)
        lut = np.arange(0, 256, dtype=np.float32).reshape(-1, 1) * hsv_gain
        lut[:, 0] %= 180
        lut[:, 1:] = np.clip(lut[:, 1:], 0, 255)
        lut = lut.astype(img.dtype).T

        h, s, v = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
        cv2.merge([cv2.LUT(h, lut[0]), cv2.LUT(s, lut[1]), cv2.LUT(v, lut[2])], dst=img)
        cv2.cvtColor(img, cv2.COLOR_HSV2BGR, dst=img)
        return img


class HistEqualizeTransform(ColorTransform):

    def __init__(self, clahe: bool = False, rgb: bool = False):
        hist_eq = functools.partial(self._hist_eq, clahe, rgb)
        super().__init__(hist_eq)

    @classmethod
    def _hist_eq(cls, clahe, rgb, img):
        code = cv2.COLOR_RGB2HSV if rgb else cv2.COLOR_BGR2HSV
        img = cv2.cvtColor(img, code, dst=img)
        if clahe:
            clahe = cv2.createCLAHE(2.0, (8, 8))
            img[:, :, 2] = clahe.apply(img[:, :, 2])
        else:
            img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])

        code = cv2.COLOR_HSV2RGB if rgb else cv2.COLOR_HSV2BGR
        img = cv2.cvtColor(img, code, dst=img)
        return img


class ToGrayTransform(ColorTransform):

    def __init__(self, rgb: bool = False):
        to_gray = functools.partial(self._to_gray, rgb)
        super().__init__(to_gray)

    @classmethod
    def _to_gray(cls, rgb, img):
        code = cv2.COLOR_RGB2GRAY if rgb else cv2.COLOR_BGR2GRAY
        gray = cv2.cvtColor(img, code)
        np.stack([gray, gray, gray], axis=2, out=img)
        return img
