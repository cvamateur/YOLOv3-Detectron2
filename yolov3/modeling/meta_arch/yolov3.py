from typing import List, Dict, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from detectron2.config import configurable
from detectron2.utils.events import get_event_storage
from detectron2.modeling.meta_arch import DenseDetector
from detectron2.structures import Boxes, Instances, ImageList
from detectron2.layers.nms import batched_nms
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.backbone import build_backbone

from ..matcher import MaxIoUMatcher
from ..anchor_generator import YOLOAnchorGenerator
from ..box_regression import YOLOBox2BoxTransform


@META_ARCH_REGISTRY.register()
class YOLOV3(DenseDetector):
    @configurable
    def __init__(self,
                 backbone,
                 head,
                 head_in_features,
                 anchor_generator,
                 box2box_transform,
                 anchor_matcher,
                 num_classes,
                 pixel_mean,
                 pixel_std,
                 loss_weights,
                 test_conf_thresh=0.25,
                 test_nms_thresh=0.5,
                 test_topk_candidates=1000,
                 test_detections_per_image=100):
        super().__init__(backbone, head, head_in_features, pixel_mean=pixel_mean, pixel_std=pixel_std)
        self.num_classes = num_classes

        self.anchor_generator = anchor_generator
        self.box2box_transform = box2box_transform
        self.anchor_matcher = anchor_matcher

        self.loss_weights = loss_weights

        self.test_conf_thresh = test_conf_thresh
        self.test_topk_candidates = test_topk_candidates
        self.test_nms_thresh = test_nms_thresh
        self.test_detections_per_image = test_detections_per_image

    @property
    def device(self):
        return self.pixel_mean.device

    @classmethod
    def from_config(cls, cfg):
        num_anchors = len(cfg.MODEL.ANCHOR_GENERATOR.ANCHORS[0]) // 2
        num_classes = cfg.MODEL.YOLOV3.NUM_CLASSES
        head_in_features = cfg.MODEL.YOLOV3.HEAD_IN_FEATURES

        backbone = build_backbone(cfg)
        backbone_shape = backbone.output_shape()
        features_shape = [backbone_shape[f] for f in head_in_features]
        return {
            "backbone": backbone,
            "head": YOLOV3Head(num_anchors, num_classes),
            "head_in_features": head_in_features,
            "anchor_generator": YOLOAnchorGenerator(cfg, features_shape),
            "box2box_transform": YOLOBox2BoxTransform(),
            "anchor_matcher": MaxIoUMatcher(cfg),
            "num_classes": num_classes,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "loss_weights": {
                "loss_obj": cfg.MODEL.YOLOV3.LOSS.WEIGHT_OBJ,
                "loss_noobj": cfg.MODEL.YOLOV3.LOSS.WEIGHT_NOOBJ,
                "loss_cls": cfg.MODEL.YOLOV3.LOSS.WEIGHT_CLS,
                "loss_box_reg": cfg.MODEL.YOLOV3.LOSS.WEIGHT_BOX_REG,
            },
            "test_conf_thresh": cfg.MODEL.YOLOV3.TEST_CONF_THRESH,
            "test_topk_candidates": cfg.MODEL.YOLOV3.TEST_TOPK_CANDIDATES,
            "test_nms_thresh": cfg.MODEL.YOLOV3.TEST_NMS_THRESH,
            "test_detections_per_image": cfg.MODEL.YOLOV3.TEST_DETECTIONS_PER_IMAGE,
        }

    def forward_training(self,
                         images: ImageList,
                         features: List[Tensor],
                         predictions: List[List[Tensor]],
                         gt_instances: List[Instances]):
        del images

        predictions = self._transpose_dense_predictions(predictions, [self.num_classes, 4, 1])
        anchors = self.anchor_generator(features)
        gt_labels, gt_deltas = self.label_anchors(anchors, gt_instances)
        losses = self.compute_losses(predictions, gt_labels, gt_deltas)
        return {k: v * self.loss_weights.get(k, 1.0) for k, v in losses.items()}

    def compute_losses(self, predictions, gt_labels, gt_boxes):
        pred_logits, pred_deltas, pred_confs = predictions

        # get pos/neg anchor masks
        gt_labels = torch.stack(gt_labels)
        pos_mask = (gt_labels >= 0) & (gt_labels != self.num_classes)
        neg_mask = (gt_labels == self.num_classes)
        get_event_storage().put_scalar("num_pos_anchors", pos_mask.sum().item())

        return {
            "loss_obj": self._compute_conf_loss(pred_confs, pos_mask, 1),
            "loss_noobj": self._compute_conf_loss(pred_confs, neg_mask, 0),
            "loss_cls": self._compute_cls_loss(pred_logits, gt_labels, pos_mask),
            "loss_box_reg": self._compute_box_reg_loss(pred_deltas, gt_boxes, pos_mask),
        }

    def _compute_cls_loss(self, pred_logits, gt_labels, pos_mask):
        if not torch.any(pos_mask).item():
            return pos_mask.new_tensor(0.0)

        pred_logits = torch.cat(pred_logits, dim=1)[pos_mask]
        gt_labels = F.one_hot(gt_labels[pos_mask], num_classes=self.num_classes + 1)
        gt_labels = gt_labels[:, :-1].to(dtype=torch.float32)
        loss_cls = F.binary_cross_entropy_with_logits(pred_logits, gt_labels)
        return loss_cls

    def _compute_conf_loss(self, pred_confs, mask, target_value: int):
        if not torch.any(mask).item():
            return mask.new_tensor(0.0)

        pred_confs = torch.cat(pred_confs, dim=1)[mask]
        gt_confs = torch.full_like(pred_confs, target_value)
        loss_conf = F.binary_cross_entropy(pred_confs, gt_confs)
        return loss_conf

    def _compute_box_reg_loss(self, pred_deltas, gt_deltas, pos_mask):
        if not torch.any(pos_mask).item():
            return pos_mask.new_tensor(0.0)

        pred_deltas = torch.cat(pred_deltas, dim=1)[pos_mask]
        gt_deltas = torch.stack(gt_deltas)[pos_mask]
        loss_box_reg = F.smooth_l1_loss(pred_deltas, gt_deltas)
        return loss_box_reg

    @torch.no_grad()
    def label_anchors(self, anchors, gt_instances):
        """
        Args:
            anchors (list[Boxes]): A list of #feature level Boxes.
                The Boxes contains anchors of this image on the specific feature level.
            gt_instances (list[Instances]): a list of N `Instances`s. The i-th
                `Instances` contains the ground-truth per-instance annotations
                for the i-th input image.

        Returns:
            list[Tensor]: List of #img tensors. i-th element is a vector of labels whose length is
            the total number of anchors across all feature maps (sum(Hi * Wi * A)).
            Label values are in {-1, 0, ..., K}, with -1 means ignore, and K means background.

            list[Tensor]: i-th element is a Rx4 tensor, where R is the total number of anchors
            across feature maps. The values are the matched gt boxes for each anchor.
            Values are undefined for those anchors not labeled as foreground.
        """
        anchor_strides = self.get_anchor_strides(anchors)
        anchors = Boxes.cat(anchors)

        gt_labels = []
        gt_deltas = []
        for gt_per_image in gt_instances:
            # anchors and gt_boxes are in original image scale
            matched_idxs, anchor_labels = self.anchor_matcher(
                gt_per_image.gt_boxes, anchors)

            gt_labels_i = gt_per_image.gt_classes[matched_idxs]
            gt_labels_i[anchor_labels == 0] = self.num_classes
            gt_labels_i[anchor_labels == -1] = -1
            gt_labels.append(gt_labels_i)

            gt_boxes_i = gt_per_image.gt_boxes.tensor[matched_idxs]
            gt_deltas_i = self.box2box_transform.get_deltas(
                anchors.tensor, gt_boxes_i, anchor_strides)
            gt_deltas.append(gt_deltas_i)

        return gt_labels, gt_deltas

    def forward_inference(self,
                          images: ImageList,
                          features: List[Tensor],
                          predictions: List[List[Tensor]]):
        pred_logits, pred_deltas, pred_confs = self._transpose_dense_predictions(
            predictions, [self.num_classes, 4, 1])
        anchors = self.anchor_generator(features)
        strides = self.anchor_generator.strides

        results = []
        for i, img_size in enumerate(images.image_sizes):
            scores_per_img = [x[i].sigmoid_() for x in pred_logits]
            deltas_per_img = [x[i] for x in pred_deltas]
            confs_per_img = [x[i] for x in pred_confs]
            results_per_img = self.inference_single_image(
                anchors, strides, deltas_per_img, confs_per_img, scores_per_img, img_size)
            results.append(results_per_img)
        return results

    def inference_single_image(self,
                               anchors: List[Boxes],
                               strides: List[Union[float, Tensor]],
                               box_deltas: List[Tensor],
                               box_confs: List[Tensor],
                               cls_scores: List[Tensor],
                               image_size: Tuple[int, int]):
        """
        Single-image inference. Return bounding-box detection results by thresholding
        on scores and applying non-maximum suppression (NMS).

        Arguments:
            anchors (list[Boxes]): list of #feature levels. Each entry contains
                a Boxes object, which contains all the anchors in that feature level.
            strides (List[float, Tensor]: list of #anchors strides for each group of
                anchors or each anchor.
            box_deltas (list[Tensor]): list of #feature levels. Each entry contains
                tensor of size (HxWxA, 4)
            box_confs (list[Tensor]): same as `box_deltas`, except last dim is 1.
            cls_scores (List[Tensor]): same as `box_confs`, except last dim is C.
            image_size (tuple(H, W)): a tuple of the image height and width.

        Returns:
            Same as `inference`, but for only one image.
        """
        pred = self._decode_multi_level_predictions(
            anchors,
            strides,
            box_deltas,
            box_confs,
            cls_scores,
            image_size,
        )
        keep = batched_nms(  # per-class NMS
            pred.pred_boxes.tensor, pred.scores, pred.pred_classes, self.test_nms_thresh
        )
        return pred[keep[: self.test_detections_per_image]]

    def _decode_multi_level_predictions(self,
                                        anchors: List[Boxes],
                                        strides: List[Union[float, Tensor]],
                                        box_deltas: List[Tensor],
                                        box_confs: List[Tensor],
                                        cls_scores: List[Tensor],
                                        image_size: Tuple[int, int]):
        predictions = [
            self._decode_per_level_predictions(
                anchors_i,
                stride_i,
                deltas_i,
                confs_i,
                scores_i,
                image_size,
            )
            for anchors_i, stride_i, deltas_i, confs_i, scores_i, in zip(
                anchors, strides, box_deltas, box_confs, cls_scores)
        ]
        return predictions[0].cat(predictions)

    def _decode_per_level_predictions(self,
                                      anchors: Boxes,
                                      stride: Union[float, Tensor],
                                      box_deltas: Tensor,
                                      box_confs: Tensor,
                                      cls_scores: Tensor,
                                      image_size: Tuple[int, int]) -> Instances:
        """
        Decode boxes and classification predictions of one feature level, by
        the following steps:
        1. filter the predictions based on confidence threshold and top K scores.
        2. transform the box regression outputs
        3. return the predicted scores, classes and boxes

        Args:
            anchors: Boxes, anchor for this feature level
            box_deltas: HxWxA,4
            box_confs: HxWxA,1
            cls_scores: HxWxA, C
        Returns:
            Instances: with field "scores", "pred_boxes", "pred_classes".
        """
        # 1. Keep boxes with confidence score higher than threshold
        conf_mask = (box_confs >= self.test_conf_thresh)
        box_confs = box_confs[conf_mask]
        topk_idxs = torch.nonzero(conf_mask)[:, 0]

        # 2. Keep top k scoring boxes only
        num_topk = min(self.test_topk_candidates, topk_idxs.size(0))
        box_confs, idxs = box_confs.topk(num_topk)
        topk_idxs = topk_idxs[idxs]

        if isinstance(stride, Tensor): stride = stride[topk_idxs]
        pred_boxes = self.box2box_transform.apply_deltas(
            box_deltas[topk_idxs], anchors.tensor[topk_idxs], stride,
        )

        # 3. P(cls) = P(obj) * P(cls | obj)
        cls_scores = cls_scores[topk_idxs]
        cls_scores.mul_(box_confs[:, None])
        pred_scores, pred_classes = cls_scores.max(dim=1)

        # 4. Filter out low confidence boxes
        conf_mask = (pred_scores >= self.test_conf_thresh)
        pred_scores = pred_scores[conf_mask]
        pred_classes = pred_classes[conf_mask]
        pred_boxes = pred_boxes[conf_mask]

        return Instances(image_size, pred_boxes=Boxes(pred_boxes), scores=pred_scores, pred_classes=pred_classes)

    def get_anchor_strides(self, anchors: List[Boxes], concat=True):
        all_strides = self.anchor_generator.strides
        anchor_strides = []
        for anchors_per_level, stride_per_level in zip(anchors, all_strides):
            num_anchors_this_level = len(anchors_per_level)
            stride_per_level = torch.tensor(stride_per_level, dtype=torch.float32, device=self.device)
            stride_per_level = stride_per_level.expand(num_anchors_this_level)
            anchor_strides.append(stride_per_level)
        if concat:
            anchor_strides = torch.cat(anchor_strides)
        return anchor_strides


class YOLOV3Head(nn.Module):
    """
    The head used in YOLO-V3 for object classification and box regression.

    It contains no parameters, just simple decode the input to multi-class logits
    and box deltas.
    """

    def __init__(self, num_anchors, num_classes):
        super().__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes

    def forward(self, features: List[Tensor]):
        """
        Arguments:
            features (list[Tensor]): FPN feature map tensors in high to low resolution.
                Each tensor in the list correspond to different feature levels, and is
                shape of [B, A(5+C), Hi, Wi], where B is batch_size, A is the number of
                anchors, C is the number of classes. The channel dimension (dim 1) is
                encoded as [(x0,y0,w0,h0,conf0, classes, ...), (x1, y1, ...)]

        Returns:
            logits (list[Tensor]): #lvl tensors, each has shape (N, AxK, Hi, Wi).
                The tensor predicts the classification probability
                at each spatial position for each of the A anchors and K object
                classes.
            bbox_reg (list[Tensor]): #lvl tensors, each has shape (N, Ax5, Hi, Wi).
                The tensor predicts 4-vector (dx,dy,dw,dh) box
                regression values for every anchor. These values are the
                relative offset between the anchor and the ground truth box.
        """
        pred_logits = []
        pred_deltas = []
        pred_confs = []

        A, C = self.num_anchors, self.num_classes
        for feature_i in features:
            B, _, H, W = feature_i.shape
            feature_i = feature_i.view(B, A, -1, H, W)

            deltas_i = feature_i[:, :, 0:4].reshape(B, -1, H, W)
            confs_i = feature_i[:, :, 4:5].reshape(B, -1, H, W)
            logits_i = feature_i[:, :, -C:].reshape(B, -1, H, W)

            # squashes tx,ty into range (-0,5, 0.5)
            deltas_i[:, 0::4] = torch.sigmoid(deltas_i[:, 0::4]) - 0.5
            deltas_i[:, 1::4] = torch.sigmoid(deltas_i[:, 1::4]) - 0.5

            # squashed conf into range (0, 1)
            confs_i = torch.sigmoid(confs_i)

            pred_logits.append(logits_i)
            pred_deltas.append(deltas_i)
            pred_confs.append(confs_i)

        return pred_logits, pred_deltas, pred_confs
