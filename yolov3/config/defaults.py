from detectron2.config.defaults import _C
from detectron2.config import CfgNode as CN

_C.MODEL.META_ARCHITECTURE = "YOLOV3"
_C.MODEL.PIXEL_MEAN = [0.0] * 3
_C.MODEL.PIXEL_STD = [255.0] * 3
_C.INPUT.FORMAT = "RGB"

# -----------------------------------------------------------------------------
# Solver
# -----------------------------------------------------------------------------
_C.SOLVER.WEIGHT_DECAY_NORM = 0.

# ---------------------------------------------------------------------------- #
# Backbone options
# ---------------------------------------------------------------------------- #
_C.MODEL.BACKBONE.NAME = "build_yolov3_backbone"  # tiny: "build_yolov3_tiny_backbone"
_C.MODEL.BACKBONE.FREEZE_AT = 0

# ---------------------------------------------------------------------------- #
# DarkNet
# ---------------------------------------------------------------------------- #
_C.MODEL.DARKNET = CN()
_C.MODEL.DARKNET.DEPTH = 53
_C.MODEL.DARKNET.NORM = "BN"
_C.MODEL.DARKNET.STEM_OUT_CHANNELS = 64
_C.MODEL.DARKNET.OUT_FEATURES = ["res3", "res4", "res5"]

# ---------------------------------------------------------------------------- #
# YOLOV3 FPN
# ---------------------------------------------------------------------------- #
_C.MODEL.YOLOV3_FPN = CN()
_C.MODEL.YOLOV3_FPN.NORM = "BN"
_C.MODEL.YOLOV3_FPN.IN_FEATURES = ["res3", "res4", "res5"]  # tiny: ["res4", "res5"]
_C.MODEL.YOLOV3_FPN.LATERAL_OUT_CHANNELS = [128, 256, 512]  # tiny: [384, 256]
_C.MODEL.YOLOV3_FPN.LATERAL_NUM_BLOCKS = [5] * 3  # tiny: [0, 1]
_C.MODEL.YOLOV3_FPN.OUTCONV_CHANNELS = [0] * 3  # 0: out = 2 * lateral_out

# ---------------------------------------------------------------------------- #
# YOLOV3
# ---------------------------------------------------------------------------- #
_C.MODEL.YOLOV3 = CN()
_C.MODEL.YOLOV3.HEAD_IN_FEATURES = ["p3", "p4", "p5"]  # tiny: ["p4", "p5"]
_C.MODEL.YOLOV3.NUM_CLASSES = 80
_C.MODEL.YOLOV3.TEST_CONF_THRESH = 0.05
_C.MODEL.YOLOV3.TEST_TOPK_CANDIDATES = 1000
_C.MODEL.YOLOV3.TEST_NMS_THRESH = 0.5
_C.MODEL.YOLOV3.TEST_DETECTIONS_PER_IMAGE = 100

# ---------------------------------------------------------------------------- #
# YOLOV3 Loss
# ---------------------------------------------------------------------------- #
_C.MODEL.YOLOV3.LOSS = CN()
_C.MODEL.YOLOV3.LOSS.BBOX_REG_LOSS_TYPE = "smooth_l1_loss"
_C.MODEL.YOLOV3.LOSS.REG_OBJ_TO_IOU = True
_C.MODEL.YOLOV3.LOSS.WEIGHT_OBJ = 1.0
_C.MODEL.YOLOV3.LOSS.WEIGHT_NOOBJ = 100.0
_C.MODEL.YOLOV3.LOSS.WEIGHT_CLS = 1.0
_C.MODEL.YOLOV3.LOSS.WEIGHT_BOX_REG = 2.0

# ---------------------------------------------------------------------------- #
# YOLOV3 Matcher
# ---------------------------------------------------------------------------- #
_C.MODEL.MATCHER = CN()
_C.MODEL.MATCHER.NAME = "MaxIoUMatcher"
_C.MODEL.MATCHER.IGNORE_THRESHOLD = 0.5
_C.MODEL.MATCHER.NEGATIVE_THRESHOLD = 0.3
_C.MODEL.MATCHER.ALLOW_LOW_QUALITY_MATCH = True

# ---------------------------------------------------------------------------- #
# Anchor Generator
# ---------------------------------------------------------------------------- #
_C.MODEL.ANCHOR_GENERATOR.NAME = "YOLOAnchorGenerator"
_C.MODEL.ANCHOR_GENERATOR.ANCHORS = [
    [10, 13, 16, 30, 33, 23],
    [30, 61, 62, 45, 59, 119],
    [116, 90, 156, 198, 373, 326],
]

# ---------------------------------------------------------------------------- #
# Test
# ---------------------------------------------------------------------------- #
_C.TEST.DETECTIONS_PER_IMAGE = 100
