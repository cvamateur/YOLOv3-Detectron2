from .augmentation_impl import (
    ResizeShortestEdge,
    RandomFlip,
    LetterBox,
    YOLOJitterColor,
    RandomPerspective,
    RandomToGray,
    RandomHistEqualize,
)


def build_yolo_augmentation(cfg, is_train=False):
    """
    If is_train:
        ResizeShortestEdge
        YOLOJitterColor
        RandomPerspective
        RandomToGray
        RandomHistEqualize
        RandomFlip
    else:
        LetterBox(rect=True)
    """
    border_value = cfg.INPUT.BORDER_VALUE

    # testing phase
    if is_train:
        mosaic_enabled = cfg.INPUT.MOSAIC.ENABLED
        if mosaic_enabled:
            min_size = max_size = cfg.INPUT.MOSAIC.PATCH_SIZE
            sample_style = "choice"
        else:
            min_size = cfg.INPUT.MIN_SIZE_TRAIN
            max_size = cfg.INPUT.MAX_SIZE_TRAIN
            sample_style = cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING

        augmentation = [ResizeShortestEdge(min_size, max_size, sample_style)]
        if cfg.INPUT.JITTER_COLOR.ENABLED:
            augmentation.append(
                YOLOJitterColor(
                    jitter_h=cfg.INPUT.JITTER_COLOR.HSV_H,
                    jitter_s=cfg.INPUT.JITTER_COLOR.HSV_S,
                    jitter_v=cfg.INPUT.JITTER_COLOR.HSV_V,
                )
            )
        if cfg.INPUT.RANDOM_PERSPECTIVE.ENABLED:
            augmentation.append(
                RandomPerspective(
                    translate=cfg.INPUT.RANDOM_PERSPECTIVE.TRANSLATE,
                    scale=cfg.INPUT.RANDOM_PERSPECTIVE.SCALE,
                    degrees=cfg.INPUT.RANDOM_PERSPECTIVE.DEGREES,
                    sheer=cfg.INPUT.RANDOM_PERSPECTIVE.SHEER,
                    perspective=cfg.INPUT.RANDOM_PERSPECTIVE.PERSPECTIVE,
                    border_value=border_value,
                )
            )

        # other augmentations
        is_rgb = (cfg.INPUT.FORMAT == "RGB")
        augmentation.extend([
            RandomToGray(0.01, rgb=is_rgb),
            RandomHistEqualize(0.01, rgb=is_rgb),
        ])
        augmentation.append(
            RandomFlip(
                horizontal=cfg.INPUT.RANDOM_FLIP == "horizontal",
                vertical=cfg.INPUT.RANDOM_FLIP == "vertical",
            )
        )
    else:
        max_size = cfg.INPUT.MAX_SIZE_TEST
        stride = cfg.MODEL.YOLOV3.MAX_STRIDE
        kwargs = {"rect": True,
                  "stretch": False,
                  "allow_scale_up": False,
                  "pad_value": border_value}
        augmentation = [LetterBox(max_size, stride, **kwargs)]

    return augmentation
