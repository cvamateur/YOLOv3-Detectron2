from typing import List, Dict
from itertools import zip_longest

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from fvcore.nn import weight_init
from detectron2.modeling import Backbone, BACKBONE_REGISTRY
from detectron2.modeling.backbone import build_resnet_backbone
from detectron2.layers import CNNBlockBase, Conv2d, ShapeSpec, get_norm

from .darknet import build_darknet53_backbone, build_tiny_darknet_backbone


class YOLOV3_FPN(Backbone):

    def __init__(self,
                 bottom_up: Backbone,
                 in_features: List[str],
                 lateral_channels: List[int],
                 num_laterals_blocks: List[int],
                 fpn_outconv_channels: List[int],
                 out_channels,
                 norm="BN"):
        super().__init__()
        assert isinstance(bottom_up, Backbone), Backbone
        assert in_features, in_features
        self.bottom_up = bottom_up
        input_shapes: Dict[str, ShapeSpec] = bottom_up.output_shape()
        strides = [input_shapes[f].stride for f in in_features]
        in_channels_per_feature = [input_shapes[f].channels for f in in_features]
        self._out_feature_strides = {}

        _assert_strides_are_log2_contiguous(strides)

        lateral_convs = []
        upsample_convs = []
        output_convs = []
        for i in reversed(range(len(in_features))):  # top-down order
            stage = int(math.log2(strides[i]))
            self._out_feature_strides[f"p{stage}"] = strides[i]
            lateral_in_chs = in_channels_per_feature[i]
            lateral_out_chs = lateral_channels[i]
            if i < len(in_features) - 1:
                lateral_in_chs += upsample_convs[-1].out_channels
            if lateral_out_chs <= 0:
                lateral_out_chs = lateral_in_chs

            # Lateral conv
            lateral_conv = FPN_LateralBlock(
                in_channels=lateral_in_chs,
                out_channels=lateral_out_chs,
                h_channels=lateral_out_chs * 2,
                n_blocks=num_laterals_blocks[i],
                norm=norm,
            )
            lateral_convs.append(lateral_conv)
            self.add_module(f"fpn_lateral{stage}", lateral_conv)

            # Output conv
            output_conv = FPN_OutputBlock(
                in_channels=lateral_out_chs,
                out_channels=out_channels,
                h_channels=fpn_outconv_channels[i],
                norm=norm,
            )
            output_convs.append(output_conv)
            self.add_module(f"fpn_output{stage}", output_conv)

            # Upsample
            if i > 0:
                upsample_conv = FPN_Upsample(lateral_out_chs, lateral_out_chs // 2, norm=norm)
                upsample_convs.append(upsample_conv)
                self.add_module(f"fpn_upsample{stage}", upsample_conv)

        self.lateral_convs = lateral_convs
        self.upsample_convs = upsample_convs
        self.output_convs = output_convs
        self.in_features = tuple(in_features)
        self._out_features = list(self._out_feature_strides.keys())
        self._out_feature_channels = {k: out_channels for k in self._out_features}
        self._size_divisibility = strides[-1]

    def forward(self, x):
        bottom_up_features = self.bottom_up(x)
        results = []  # top-down order

        topdown_features = None
        for i, (lateral_conv, upsample_conv, output_conv) in enumerate(
                zip_longest(self.lateral_convs, self.upsample_convs, self.output_convs)
        ):
            features = self.in_features[-i - 1]
            features = bottom_up_features[features]
            if topdown_features is None:
                lateral_feature = lateral_conv(features)
                results.append(output_conv(lateral_feature))
            else:
                features = torch.cat([topdown_features, features], dim=1)
                lateral_feature = lateral_conv(features)
                results.append(output_conv(lateral_feature))
            if upsample_conv is not None:
                topdown_features = upsample_conv(lateral_feature)

        assert len(results) == len(self._out_features)
        return dict(zip(self._out_features, results))

    @property
    def size_divisibility(self) -> int:
        return self._size_divisibility

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }

    def darknet_modules(self):
        for name, conv in self.named_modules():
            if isinstance(conv, Conv2d):
                yield name, conv


class FPN_OutputBlock(CNNBlockBase):
    def __init__(self, in_channels, out_channels, h_channels=0, *, norm="BN"):
        super().__init__(in_channels, out_channels, stride=1)
        h_channels = h_channels if h_channels else in_channels * 2
        self.conv0 = Conv2d(
            in_channels,
            h_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            norm=get_norm(norm, h_channels)
        )
        self.conv1 = Conv2d(
            h_channels,
            out_channels,
            kernel_size=1,
            stride=1,
        )

        weight_init.c2_xavier_fill(self.conv0)
        weight_init.c2_xavier_fill(self.conv1)

    def forward(self, x):
        x = self.conv0(x)
        x = F.leaky_relu_(x, 0.1)
        x = self.conv1(x)
        return x


class FPN_LateralBlock(CNNBlockBase):
    """
    LateralBlock is just 5 consecutive conv-bn-leaky blocks.

    n_blocks = 0:  YOLOV3-Tiny (p4 lateral, that is, a pass-through layer)
    n_blocks = 1:  YOLOV3-Tiny (p5 lateral conv)
    n_blocks = 5:  YOLOV3
    """
    _allow_num_blocks = {0, 1, 5}

    def __init__(self, in_channels, out_channels, h_channels=0, n_blocks=5, *, norm="BN"):
        assert n_blocks in self._allow_num_blocks
        super().__init__(in_channels, out_channels, stride=1)
        self.blocks = nn.ModuleList()

        if n_blocks == 0:  # used in yolov3-tiny
            self.in_channels = out_channels
        else:
            self.blocks.append(
                Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=1,
                    bias=False,
                    norm=get_norm(norm, out_channels),
                )
            )

        h_channels = h_channels if h_channels else out_channels * 2
        for i in range(1, n_blocks):
            self.blocks.append(
                Conv2d(
                    out_channels,
                    h_channels,
                    kernel_size=1 if i % 2 == 0 else 3,
                    stride=1,
                    padding=0 if i % 2 == 0 else 1,
                    bias=False,
                    norm=get_norm(norm, h_channels)
                )
            )
            h_channels, out_channels = out_channels, h_channels

        for conv in self.blocks:
            weight_init.c2_xavier_fill(conv)

    def forward(self, x):
        for conv in self.blocks:
            x = conv(x)
            x = F.leaky_relu_(x, 0.1)
        return x


class FPN_Upsample(CNNBlockBase):

    def __init__(self, in_channels, out_channels, *, norm="BN"):
        super().__init__(in_channels, out_channels, stride=1)
        self.conv = Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            bias=False,
            norm=get_norm(norm, out_channels),
        )
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        weight_init.c2_xavier_fill(self.conv)

    def forward(self, x):
        x = self.conv(x)
        x = F.leaky_relu_(x, 0.1)
        x = self.upsample(x)
        return x


def _assert_strides_are_log2_contiguous(strides):
    """
    Assert that each stride is 2x times its preceding stride, i.e. "contiguous in log2".
    """
    for i, stride in enumerate(strides[1:], 1):
        assert stride == 2 * strides[i - 1], "Strides {} {} are not log2 contiguous".format(
            stride, strides[i - 1]
        )


@BACKBONE_REGISTRY.register()
def build_yolov3_backbone(cfg, input_shape: ShapeSpec):
    bottom_up = build_darknet53_backbone(cfg, input_shape)

    fpn_norm = cfg.MODEL.YOLOV3_FPN.NORM
    fpn_in_features = cfg.MODEL.YOLOV3_FPN.IN_FEATURES
    fpn_lateral_channels = cfg.MODEL.YOLOV3_FPN.LATERAL_OUT_CHANNELS
    fpn_lateral_num_blocks = cfg.MODEL.YOLOV3_FPN.LATERAL_NUM_BLOCKS
    fpn_outconv_channels = cfg.MODEL.YOLOV3_FPN.OUTCONV_CHANNELS
    yolo_out_channels = _calc_yolo_out_channels(cfg)

    backbone = YOLOV3_FPN(
        bottom_up,
        fpn_in_features,
        fpn_lateral_channels,
        fpn_lateral_num_blocks,
        fpn_outconv_channels,
        yolo_out_channels,
        norm=fpn_norm)

    return backbone


@BACKBONE_REGISTRY.register()
def build_yolov3_tiny_backbone(cfg, input_shape: ShapeSpec):
    bottom_up = build_tiny_darknet_backbone(cfg, input_shape)

    fpn_norm = cfg.MODEL.YOLOV3_FPN.NORM
    fpn_in_features = cfg.MODEL.YOLOV3_FPN.IN_FEATURES
    fpn_lateral_channels = cfg.MODEL.YOLOV3_FPN.LATERAL_OUT_CHANNELS
    fpn_lateral_num_blocks = cfg.MODEL.YOLOV3_FPN.LATERAL_NUM_BLOCKS
    fpn_outconv_channels = cfg.MODEL.YOLOV3_FPN.OUTCONV_CHANNELS
    yolo_out_channels = _calc_yolo_out_channels(cfg)

    backbone = YOLOV3_FPN(
        bottom_up,
        fpn_in_features,
        fpn_lateral_channels,
        fpn_lateral_num_blocks,
        fpn_outconv_channels,
        yolo_out_channels,
        norm=fpn_norm)

    return backbone


def _calc_yolo_out_channels(cfg):
    """
    YOLO3 encodes out_channels as A*(5+C), where A is the number of anchors per
    each level, 5 is dimensions of each predicted box (tx, ty, tw, th, c), C
    is the number of classes.
    """
    A = len(cfg.MODEL.ANCHOR_GENERATOR.ANCHORS[0]) // 2
    C = cfg.MODEL.YOLOV3.NUM_CLASSES
    out_channels = A * (5 + C)
    return out_channels
