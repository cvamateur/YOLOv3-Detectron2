import io
import os
import torch
import logging
import numpy as np
import collections

from detectron2.layers import Conv2d


def _load_single_tensor_to_dict(d, fp, name, shape):
    count = int(np.prod(shape))
    w = np.fromfile(fp, dtype=np.float32, count=count)
    w = torch.from_numpy(w).view(shape)
    d[name] = w
    return count


def load_darknet_weights_to_dict(f, module_list, module_prefix="backbone"):
    """
    Load weights from official `darknet53.weights`.

    Official weights file is a binary file where weights are stored in serial order.
    The weights are stored as floats. Extreme care must be taken while loading weight
    file.Some key points to note:

    The weights belong to only two types of layers, either a batch norm layer or a
    convolutional layer.
    The weights are stored in exactly the same order as in configuration file.
    If the convolution layer contains batch normalization, there we will be no bias
    value for convolution.Only weights value will be there for such convolution layers.
    If the convolution layer does not contain batch normalization there will be both
    bias and weight value.

    The first 5 values are header information:
        1. Major version number
        2. Minor Version Number
        3. Subversion number
        4,5. Images seen by the network (during training)
    """
    assert isinstance(f, io.IOBase)
    module_list = list(module_list)
    logger = logging.getLogger(__name__)

    header = np.fromfile(f, dtype=np.int32, count=5)
    logger.info(f"load official weights (v{header[0]}.{header[1]}.{header[2]})")

    state_dict = collections.OrderedDict()
    for field, m in module_list:
        if not isinstance(m, Conv2d):
            continue
        if module_prefix:
            field = f"{module_prefix}.{field}"

        bn = m.norm
        if bn is not None:
            _load_single_tensor_to_dict(state_dict, f, f"{field}.norm.bias", bn.bias.shape)
            _load_single_tensor_to_dict(state_dict, f, f"{field}.norm.weight", bn.weight.shape)
            _load_single_tensor_to_dict(state_dict, f, f"{field}.norm.running_mean", bn.running_mean.shape)
            _load_single_tensor_to_dict(state_dict, f, f"{field}.norm.running_var", bn.running_var.shape)
            _load_single_tensor_to_dict(state_dict, f, f"{field}.weight", m.weight.shape)
        else:
            # the conv layer has no bn, then it is a yolo-output-conv.
            # the original yolo-output-conv is trained on coco dataset,
            # so its channels must be 255(=3x(5+80)). If these channels
            # are not 255, then the parameters should be ignored
            bias_shape = [255]
            weight_shape = [255, *m.weight.shape[1:]]
            _load_single_tensor_to_dict(state_dict, f, f"{field}.bias", bias_shape)
            _load_single_tensor_to_dict(state_dict, f, f"{field}.weight", weight_shape)
            if m.bias.shape[0] != 255:
                state_dict.pop(f"{field}.bias")
                state_dict.pop(f"{field}.weight")

    return state_dict
