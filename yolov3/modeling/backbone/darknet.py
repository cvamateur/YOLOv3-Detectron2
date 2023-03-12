import functools
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from fvcore.nn import weight_init
from detectron2.config import configurable
from detectron2.modeling.backbone import Backbone, BACKBONE_REGISTRY
from detectron2.layers import CNNBlockBase, Conv2d, ShapeSpec, get_norm

DarkNetBlockBase = CNNBlockBase


class DarkNetResidualBlock(DarkNetBlockBase):
    """
    The basic residual block for DarkNet 53, defined in :paper:`YOLOv3`,
    with 1x1 conv layer followed by a 3x3 conv layer

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        norm (str or callable): normalization for all conv layers.
            See :func:`layers.get_norm` for supported format.
    """
    degradation = 2

    def __init__(self, in_channels, out_channels, *, downsample=False, norm="BN"):
        super().__init__(in_channels, out_channels, stride=2 if downsample else 1)
        if downsample:
            self.downsample = Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
                norm=get_norm(norm, out_channels),
            )
        else:
            self.downsample = None

        hidden_channels = out_channels // self.degradation
        self.conv1 = Conv2d(
            out_channels,
            hidden_channels,
            kernel_size=1,
            stride=1,
            bias=False,
            norm=get_norm(norm, hidden_channels),
        )

        self.conv2 = Conv2d(
            hidden_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            norm=get_norm(norm, out_channels),
        )

        weight_init.c2_msra_fill(self.conv1)
        weight_init.c2_msra_fill(self.conv2)

    def forward(self, x):
        if self.downsample is not None:
            x = self.downsample(x)
            x = F.leaky_relu_(x, 0.1)

        identity = x
        x = self.conv1(x)
        x = F.leaky_relu_(x, 0.1)
        x = self.conv2(x)
        x = F.leaky_relu_(x, 0.1)
        x = x + identity
        return x


class BasicStem(CNNBlockBase):

    def __init__(self, in_channels, out_channels, *, norm="BN"):
        super().__init__(in_channels, out_channels, stride=2)
        hidden_channels = out_channels // 2
        self.conv = Conv2d(
            in_channels,
            hidden_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            norm=get_norm(norm, hidden_channels),
        )
        self.res1 = DarkNetResidualBlock(hidden_channels, out_channels, downsample=True, norm=norm)

    def forward(self, x):
        x = self.conv(x)
        x = F.leaky_relu_(x, 0.1)
        x = self.res1(x)
        return x


class DarkNet(Backbone):
    """
    Implement of :paper:`YOLO_V3`.

    Args:
        stem (nn.Module): a stem module
        stages (list[list[CNNBlockBase]]): several (typically 4) stages,
            each contains multiple :class:`CNNBlockBase`.
        num_classes (None or int): if None, will not perform classification.
            Otherwise, will create a linear layer.
        out_features (list[str]): name of the layers whose outputs should
            be returned in forward. Can be anything in "stem", "linear", or "res2" ...
            If None, will return the output of the last layer.
        freeze_at (int): The number of stages at the beginning to freeze.
            see :meth:`freeze` for detailed explanation.
    """
    DEPTH: int = 53

    def __init__(self, stem, stages, num_classes=None, out_features=None, freeze_at=0):
        super().__init__()
        self.stem = stem
        self.num_classes = num_classes

        current_stride = stem.stride
        current_channels = stem.out_channels
        name = "stem"
        self._out_feature_strides = {name: current_stride}
        self._out_feature_channels = {name: current_channels}
        self.stage_names, self.stages = [], []
        if out_features is not None:
            num_stages = max(
                {"res2": 1, "res3": 2, "res4": 3, "res5": 4}[f] for f in out_features
            )
            stages = stages[:num_stages]
        for i, blocks in enumerate(stages, start=2):
            assert len(blocks) > 0, len(blocks)
            for block in blocks:
                assert isinstance(block, CNNBlockBase), block
            name = f"res{i}"
            stage = nn.Sequential(*blocks)
            self.add_module(name, stage)
            self.stage_names.append(name)
            self.stages.append(stage)

            self._out_feature_strides[name] = current_stride * np.prod([k.stride for k in blocks])
            self._out_feature_channels[name] = blocks[-1].out_channels
            current_stride = self._out_feature_strides[name]
            current_channels = self._out_feature_channels[name]

        self.stage_names = tuple(self.stage_names)
        if num_classes is not None:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.linear = Conv2d(
                current_channels,
                num_classes,
                kernel_size=1,
                stride=1,
            )
            nn.init.normal_(self.linear.weight, std=0.01)
            nn.init.constant_(self.linear.bias, 0)
            name = "linear"

        if out_features is None:
            out_features = [name]
        self._out_features = out_features
        assert len(self._out_features)
        self.freeze(freeze_at)

    @classmethod
    def make_stage(cls, block_class, num_blocks, *, in_channels, out_channels, norm):
        """
        Create a list of blocks that follows the same pattern:
            Downsample + block_class() * num_blocks
        """
        blocks = []
        for i in range(num_blocks):
            downsample = (i == 0)
            blocks.append(block_class(in_channels, out_channels, downsample=downsample, norm=norm))
        return blocks

    def freeze(self, freeze_at=0):
        if freeze_at >= 1:
            self.stem.freeze()
        for stage_i, stage in enumerate(self.stages, start=2):
            if freeze_at >= stage_i:
                for block in stage.children():
                    block.freeze()

    def output_shape(self) -> Dict[str, ShapeSpec]:
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (N,C,H,W). H, W must be a multiple of ``self.size_divisibility``.

        Returns:
            dict[str->Tensor]: names and the corresponding features
        """
        assert x.dim() == 4
        outputs = {}
        x = self.stem(x)
        if "stem" in self._out_features:
            outputs["stem"] = x
        for name, stage in zip(self.stage_names, self.stages):
            x = stage(x)
            if name in self._out_features:
                outputs[name] = x
        if self.num_classes is not None:
            x = self.avg_pool(x)
            x = torch.flatten(x, 1)
            logits = self.linear(x)
            if "linear" in self._out_features:
                outputs["linear"] = logits
        return outputs

    def darknet_modules(self):
        for name, conv in self.named_modules():
            if isinstance(conv, Conv2d):
                yield name, conv


@BACKBONE_REGISTRY.register()
def build_darknet53_backbone(cfg, input_shape: ShapeSpec):
    assert cfg.MODEL.DARKNET.DEPTH == DarkNet.DEPTH
    norm = cfg.MODEL.DARKNET.NORM
    out_channels = cfg.MODEL.DARKNET.STEM_OUT_CHANNELS
    stem = BasicStem(
        in_channels=input_shape.channels,
        out_channels=out_channels,
        norm=norm,
    )

    freeze_at = cfg.MODEL.BACKBONE.FREEZE_AT
    out_features = cfg.MODEL.DARKNET.OUT_FEATURES
    num_blocks_per_stage = [2, 8, 8, 4]

    stages = []
    for i, res_i in enumerate(range(2, 6)):
        in_channels = out_channels
        out_channels *= 2
        stage_kwargs = {
            "block_class": DarkNetResidualBlock,
            "num_blocks": num_blocks_per_stage[i],
            "in_channels": in_channels,
            "out_channels": out_channels,
            "norm": norm,
        }
        blocks = DarkNet.make_stage(**stage_kwargs)
        stages.append(blocks)

    return DarkNet(stem, stages, out_features=out_features, freeze_at=freeze_at)


class TinyBasicBlock(CNNBlockBase):
    """
    Basic conv-pool block in Yolov3-tiny.
    """

    def __init__(self, in_channels, out_channels, downsample=True, *, norm="BN"):
        super().__init__(in_channels, out_channels, stride=2 if downsample else 1)
        self.pad = not downsample
        self.pool = nn.MaxPool2d(2, self.stride)
        self.conv = Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            norm=get_norm(norm, out_channels)
        )
        weight_init.c2_xavier_fill(self.conv)

    def forward(self, x):
        if self.pad:
            x = F.pad(x, [0, 1, 0, 1], "constant", 0)
        x = self.pool(x)
        x = self.conv(x)
        x = F.leaky_relu_(x, 0.1)
        return x


class YOLO_TinyStem(CNNBlockBase):

    def __init__(self, in_channels, out_channels, *, norm="BN"):
        super().__init__(in_channels, out_channels, 2)
        hidden_dim = out_channels // 2
        self.conv = Conv2d(
            in_channels,
            hidden_dim,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            norm=get_norm(norm, hidden_dim),
        )
        weight_init.c2_xavier_fill(self.conv)

        self.res1 = TinyBasicBlock(hidden_dim, out_channels, norm=norm)

    def forward(self, x):
        x = self.conv(x)
        x = F.leaky_relu_(x, 0.1)
        x = self.res1(x)
        return x


@BACKBONE_REGISTRY.register()
def build_tiny_darknet_backbone(cfg, input_shape: ShapeSpec):
    norm = cfg.MODEL.DARKNET.NORM
    out_channels = 32

    stem = YOLO_TinyStem(
        in_channels=input_shape.channels,
        out_channels=out_channels,
        norm=norm,
    )

    freeze_at = cfg.MODEL.BACKBONE.FREEZE_AT
    out_features = cfg.MODEL.DARKNET.OUT_FEATURES[-2:]
    num_blocks_per_stage = [1, 1, 1, 2]

    stages = []
    for i, num_blocks in enumerate(num_blocks_per_stage):
        blocks = []
        for n in range(num_blocks):
            in_channels = out_channels
            out_channels *= 2
            downsample = (n == 0)
            blocks.append(TinyBasicBlock(in_channels, out_channels, downsample, norm=norm))
        stages.append(blocks)

    return DarkNet(stem, stages, out_features=out_features, freeze_at=freeze_at)
