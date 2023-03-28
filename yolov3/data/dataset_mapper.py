import copy
import logging
import random

import torch
import cv2
import numpy as np

from collections import deque
from typing import List, Dict, Any, Union, Optional, Tuple

from detectron2.config import configurable
from detectron2.data import transforms as T
from detectron2.data import detection_utils as utils

from .detection_utils import build_yolo_augmentation


class YOLODatasetMapper(object):
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by the model.

    This is the default callable to be used to map your dataset dict into training data.
    You may need to follow it to implement your own one for customized logic,
    such as a different way to read or transform images.
    See :doc:`/tutorials/data_loading` for details.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies cropping/geometric transforms to the image and annotations
    3. Prepare data and annotations to Tensor and :class:`Instances`


    NOTE: this interface is experimental.

    Args:
        is_train: whether it's used in training or inference
        augmentations: a list of augmentations or deterministic transforms to apply
        image_format: an image format supported by :func:`detection_utils.read_image`.
        use_mosaic: load 4 images and blend them together.
        use_instance_mask: whether to process instance segmentation annotations, if available
        use_keypoint: whether to process keypoint annotations if available
        instance_mask_format: one of "polygon" or "bitmask". Process instance segmentation
            masks into this format.
    """

    @configurable
    def __init__(self,
                 is_train: bool,
                 *,
                 augmentations: List[Union[T.Augmentation, T.Transform]],
                 image_format: str,
                 use_instance_mask: bool = False,
                 use_keypoint: bool = False,
                 instance_mask_format: str = "polygon",
                 keypoint_hflip_indices: Optional[np.ndarray] = None,
                 recompute_boxes: bool = False,
                 mosaic_enabled: bool = False,
                 mosaic_patch_size: int = 640,
                 mosaic_pool_size: int = 1000,
                 mosaic_border: Union[int, Tuple[int, ...]] = 0):

        # fmt: off
        self.is_train = is_train
        self.augmentations = T.AugmentationList(augmentations)
        self.image_format = image_format
        self.use_instance_mask = use_instance_mask
        self.use_keypoint = use_keypoint
        self.instance_mask_format = instance_mask_format
        self.keypoint_hflip_indices = keypoint_hflip_indices
        self.recompute_boxes = recompute_boxes

        self.mosaic_enabled = mosaic_enabled
        self.mosaic_patch_size = mosaic_patch_size
        self.mosaic_pool = deque(maxlen=mosaic_pool_size) if mosaic_enabled else None
        self.mosaic_border = mosaic_border

        # fmt: on
        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(f"[DatasetMapper] Augmentations used in {mode}: {augmentations}")

    @classmethod
    def from_config(self, cfg, is_train: bool = True):
        if not cfg.INPUT.USE_YOLO_AUG:
            augs = utils.build_augmentation(cfg, is_train)
        else:
            augs = build_yolo_augmentation(cfg, is_train)
        if cfg.INPUT.CROP.ENABLED and is_train:
            augs.insert(0, T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE))
            recompute_boxes = cfg.MODEL.MASK_ON
        else:
            recompute_boxes = False

        ret = {
            "is_train": is_train,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "use_instance_mask": cfg.MODEL.MASK_ON,
            "use_keypoint": cfg.MODEL.KEYPOINT_ON,
            "instance_mask_format": cfg.INPUT.MASK_FORMAT,
            "recompute_boxes": recompute_boxes,
        }

        if cfg.MODEL.KEYPOINT_ON:
            ret["keypoint_hflip_indices"] = utils.create_keypoint_hflip_indices(cfg.DATASETS.TRAIN)
        if cfg.INPUT.MOSAIC.ENABLED:
            ret["mosaic_enabled"] = True
            ret["mosaic_patch_size"] = cfg.INPUT.MOSAIC.PATCH_SIZE
            ret["mosaic_pool_size"] = cfg.INPUT.MOSAIC.POOL_SIZE
            ret["mosaic_border"] = cfg.INPUT.BORDER_VALUE
        return ret

    def __call__(self, dataset_dict: Dict[str, Any]):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.
        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        do_mosaic = 0
        if self.mosaic_enabled and self.is_train:
            do_mosaic = 1
            self.mosaic_pool.append(copy.deepcopy(dataset_dict))

        if do_mosaic:
            image, annos, image_shape = self._load_mosaic(dataset_dict)
        else:
            image, annos, image_shape = self._load_image(dataset_dict)

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        if annos:
            instances = utils.annotations_to_instances(
                annos, image_shape, mask_format=self.instance_mask_format)

            # After transforms such as cropping are applied, the bounding box may no longer
            # tightly bound the object. As an example, imagine a triangle object
            # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format). The tight
            # bounding box of the cropped triangle should be [(1,0),(2,1)], which is not equal to
            # the intersection of original bounding box and the cropping box.
            if self.recompute_boxes:
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            dataset_dict["instances"] = utils.filter_empty_instances(instances)

        return dataset_dict

    def _load_image(self, dataset_dict):
        """
        Load image and its annotations from dataset_dict.
        """
        dataset_dict = copy.deepcopy(dataset_dict)
        image = utils.read_image(dataset_dict["file_name"], self.image_format)
        utils.check_image_size(dataset_dict, image)

        aug_input = T.AugInput(image)
        transforms = self.augmentations(aug_input)
        image = aug_input.image
        image_shape = image.shape[:2]

        if not self.is_train:
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            return image, None, image_shape

        for anno in dataset_dict.get("annotations", []):
            if not self.use_instance_mask:
                anno.pop("segmentation", None)
            if not self.use_keypoint:
                anno.pop("keypoints", None)

        annos = [
            utils.transform_instance_annotations(
                anno, transforms, image_shape,
                keypoint_hflip_indices=self.keypoint_hflip_indices)
            for anno in dataset_dict.pop("annotations", [])
            if anno.get("iscrowd", 0) == 0
        ]
        return image, annos, image_shape

    def _load_mosaic(self, dataset_dict):
        """
        Load 4 images and their annotations, where one image from given
        dataset_dict, other three images are sampled from mosaic_pool.
        """
        mosaic_samples = [dataset_dict] + random.choices(self.mosaic_pool, k=3)
        random.shuffle(mosaic_samples)

        s = 2 * self.mosaic_patch_size
        cx, cy = np.random.randint(s // 4, 3 * s // 4, size=[2])  # mosaic center
        image, annos = None, []
        for i, dataset_dict in enumerate(mosaic_samples):
            image_i, annos_i, (h, w) = self._load_image(dataset_dict)
            if i == 0:  # top-left
                image = np.full([s, s, image_i.shape[2]], self.mosaic_border, dtype=np.uint8)
                x1a, y1a, x2a, y2a = max(0, cx - w), max(0, cy - h), cx, cy  # xmin, ymin, xmax, ymax (dst image)
                x1b, y1b, x2b, y2b = max(0, w - cx), max(0, h - cy), w, h  # xmin, ymin, xmax, ymax (src image)
            elif i == 1:  # top-right
                x1a, y1a, x2a, y2a = cx, max(0, cy - h), min(s, cx + w), cy
                x1b, y1b, x2b, y2b = 0, max(0, h - cy), min(w, s - cx), h
            elif i == 2:  # bottom-left
                x1a, y1a, x2a, y2a = max(0, cx - w), cy, cx, min(s, cy + h)
                x1b, y1b, x2b, y2b = max(0, w - cx), 0, w, min(h, s - cy)
            else:  # bottom-right
                x1a, y1a, x2a, y2a = cx, cy, min(s, cx + w), min(s, cy + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, s - cx), min(h, s - cy)

            image[y1a: y2a, x1a: x2a] = image_i[y1b: y2b, x1b: x2b]

            offset_x = x1a - x1b
            offset_y = y1a - y1b
            for anno in annos_i:
                bbox = anno["bbox"]
                bbox[0::2] += offset_x
                bbox[1::2] += offset_y
                np.clip(bbox, 0, s, out=bbox)
            annos.extend(annos_i)

        return image, annos, image.shape[:2]
