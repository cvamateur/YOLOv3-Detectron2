from typing import Optional, Union

import torch
from torch import Tensor

from detectron2.modeling.box_regression import _DEFAULT_SCALE_CLAMP


class YOLOBox2BoxTransform(object):
    """
    The box-to-box transform defined in YOLO. The transformation is parameterized
    by 4 deltas (tx, ty, tw, th). The original YOLO transformation is given by
    following formulas:
        x_p = x_a + tx
        y_p = y_a + ty
        w_p = w_a * exp(tw)
        h_p = h_a * exp(th)
    where,
        tx, ty: values in range (-0.5, 0.5)
        tw, th: values in range (-inf, inf)
        w_*, y_*, w_*, h_*: all values are in feature map scale.
    """

    def __init__(self, scale_clamp: float = _DEFAULT_SCALE_CLAMP):
        self.scale_clamp = scale_clamp

    @torch.no_grad()
    def get_deltas(self, src_boxes: Tensor, target_boxes: Tensor, strides: Optional[Tensor] = None) -> Tensor:
        """
        Get box regression transformation (tx, ty, tw, th) that can transform src_boxes
        into target_boxes.

        If both src_boxes and target_boxes are in original image scale, then stride is
        needed to rescale them to feature map scale, thus the calculated 4 deltas are
        in range (0, 1).

        Args:
            src_boxes: a float tensor of arbitrary shape [C0, C1, ..., 4], where the last dimension
                is (x0, y0, x1, y1).
            target_boxes: same type as src_boxes.
            strides: If provided, it must have shape [C0, C1, ...] which is broadcastable with boxes.
        """
        if strides is not None:
            src_boxes = src_boxes / strides[:, None]
            target_boxes = target_boxes / strides[:, None]

        src_wh = src_boxes[..., 2:4] - src_boxes[..., 0:2]
        src_xy = src_boxes[..., 0:2] + 0.5 * src_wh
        tgt_wh = target_boxes[..., 2:4] - target_boxes[..., 0:2]
        tgt_xy = target_boxes[..., 0:2] + 0.5 * tgt_wh

        deltas = torch.empty_like(target_boxes[..., 0:4])
        deltas[..., 0:2] = tgt_xy - src_xy
        deltas[..., 2:4] = torch.log(tgt_wh / (src_wh))
        return deltas

    def apply_deltas(self, deltas: Tensor, boxes: Tensor, stride: Union[float, Tensor] = 1.0):
        """
        Apply transformation `deltas` (dx, dy, dw, dh) to `boxes`.

        Args:
            deltas: transformations of shape (N, k*4), where k >= 1.
                deltas[i] represents k potentially different class-specific
                box transformations for the single box boxes[i].
            boxes (Tensor): boxes to transform, of shape (N, 4)
        """
        # ensure fp32 for decoding precision
        deltas = deltas.float()
        boxes = boxes.to(deltas.dtype)
        if isinstance(stride, Tensor):
            assert stride.dim() == 1
            stride = stride[:, None]

        widths = boxes[..., 2] - boxes[..., 0]
        heights = boxes[..., 3] - boxes[..., 1]
        ctr_x = boxes[..., 0] + 0.5 * widths
        ctr_y = boxes[..., 1] + 0.5 * heights

        dx = deltas[:, 0::4] * stride
        dy = deltas[:, 1::4] * stride
        dw = deltas[:, 2::4].clamp(max=self.scale_clamp)
        dh = deltas[:, 3::4].clamp(max=self.scale_clamp)

        proposals = torch.empty_like(deltas)
        proposals[..., 0::4] = ctr_x[..., None] + dx
        proposals[..., 1::4] = ctr_y[..., None] + dy
        proposals[..., 2::4] = widths[..., None] * dw.exp()
        proposals[..., 3::4] = heights[..., None] * dh.exp()

        proposals[..., 0::4] -= 0.5 * proposals[..., 2::4]
        proposals[..., 1::4] -= 0.5 * proposals[..., 3::4]
        proposals[..., 2::4] += proposals[..., 0::4]
        proposals[..., 3::4] += proposals[..., 1::4]
        return proposals
