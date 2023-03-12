# Copyright (c) Facebook, Inc. and its affiliates.
from typing import List
import math

import numpy as np
import torch

from detectron2.layers import ShapeSpec
from detectron2.modeling.anchor_generator import (
    ANCHOR_GENERATOR_REGISTRY,
    DefaultAnchorGenerator,
    BufferList,
)


class YOLOAnchorGenerator(DefaultAnchorGenerator):

    @classmethod
    def from_config(cls, cfg, input_shape: List[ShapeSpec]):
        anchors = cfg.MODEL.ANCHOR_GENERATOR.ANCHORS
        assert all(len(x) == len(anchors[0]) for x in anchors), \
            "number anchors of all layers must be equal"

        num_layers = len(anchors)
        anchor_sizes, aspect_ratios = [], []
        for i in range(num_layers):
            anchors_i = np.array(anchors[i]).reshape(-1, 2)
            anchor_sizes_i = [math.sqrt(anc[0] * anc[1]) for anc in anchors_i]
            aspect_ratios_i = [float(anc[1]) / anc[0] for anc in anchors_i]
            anchor_sizes.append(anchor_sizes_i)
            aspect_ratios.append(aspect_ratios_i)

        return {
            "sizes": anchor_sizes,
            "aspect_ratios": aspect_ratios,
            "strides": [x.stride for x in input_shape],
        }

    def generate_cell_anchors(self, sizes, aspect_ratios):
        """
        Generate a tensor storing canonical anchor boxes, which are all anchor
        boxes of different sizes and aspect_ratios centered at (0, 0).
        We can later build the set of anchors for a full feature map by
        shifting and tiling these tensors (see `meth:_grid_anchors`).

        Args:
            sizes (tuple[float]):
            aspect_ratios (tuple[float]]):

        Returns:
            Tensor of shape (len(sizes) * len(aspect_ratios), 4) storing anchor boxes
                in XYXY format.
        """
        anchors = []
        for size, aspect_ratio in zip(sizes, aspect_ratios):
            area = size ** 2.0
            w = math.sqrt(area / aspect_ratio)
            h = aspect_ratio * w
            x0, y0, x1, y1 = -w / 2.0, -h / 2.0, w / 2.0, h / 2.0
            anchors.append([x0, y0, x1, y1])
        return torch.tensor(anchors, dtype=torch.float32)
