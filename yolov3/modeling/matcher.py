from typing import Union, List
import torch

from detectron2.config import configurable
from detectron2.structures import pairwise_iou, Boxes


class MaxIoUMatcher(object):
    """
    Assign each ground-truth box to a best overlapped predicted "element" (e.g. anchor).
    Each predicted element will have at most one match; each ground-truth will have
    exactly one match.

    The matching is determined by the MxN match_quality_matrix, that characterizes
    how well each (ground-truth, prediction)-pair match each other.

    The matcher returns (a) a vector of length N containing the index of the
    ground-truth element m in [0, M) that matches to prediction n in [0, N).
    (b) a vector of length N containing the labels for each prediction.
    """

    @configurable
    def __init__(self, labels: List[int], ignore_thresh: float, allow_low_quality_matches: bool = True):
        assert all(l in [-1, 0, 1] for l in labels)
        self.ignore_thresh = ignore_thresh
        self.labels = [-1, 0, 1]
        self.allow_low_quality_matches = allow_low_quality_matches

    @classmethod
    def from_config(cls, cfg):
        return {
            "labels": [-1, 0, 1],
            "ignore_thresh": cfg.MODEL.MATCHER.IGNORE_THRESHOLD,
            "allow_low_quality_matches": True,
        }

    def __call__(self, gt_boxes, anchors):
        """
        Args:
            gt_boxes (Tensor[Boxes])
            anchors (Tensor[Boxes])

        Returns:
            matches (Tensor[int64]): a vector of length N, where matches[i] is a matched
                ground-truth index in [0, M)
            match_labels (Tensor[int8]): a vector of length N, where pred_labels[i] indicates
                whether a prediction is a true or false positive or ignored
        """
        iou_matrix = pairwise_iou(gt_boxes, anchors)
        matched_vals, matches = iou_matrix.max(dim=0)

        # In YOLO-V3, each gt only matches ONE anchor
        # 1. Find the best matched anchor for each gt, we can achieve this
        # by maxing over the predictions (dim 1)
        best_ious = iou_matrix.max(dim=1, keepdim=True)[0]
        best_ious_mask = (best_ious == iou_matrix)

        # 2. The matched anchor must have minimum distances with gt
        dist_matrix = _pairwise_l1_distances(gt_boxes, anchors)
        dist_matrix.masked_fill_(best_ious_mask.logical_not(), float("inf"))
        min_dists = dist_matrix.min(dim=1, keepdim=True)[0]
        min_dist_mask = min_dists == dist_matrix
        matched_gt_idx, best_matches = torch.nonzero(min_dist_mask, as_tuple=True)
        matches[best_matches] = matched_gt_idx

        # Assign labels to each anchor:
        #   1: Best match
        #  -1: Not the best but iou > ignore_thresh
        #   0: Others
        match_labels = matches.new_zeros(matches.size(), dtype=torch.int8)
        match_labels[matched_vals > self.ignore_thresh] = -1
        match_labels[best_matches] = 1

        # In case that max-iou < 0.3, then ignore that gt
        if not self.allow_low_quality_matches:
            match_labels[matched_vals < 0.3] = 0
        return matches, match_labels


def _pairwise_l1_distances(boxes1: Boxes, boxes2: Boxes):
    """
    :param boxes1: [N, 4]
    :param centers: [M, 2]
    :param stride: (float)
    :return: [N, M]
    """
    boxes1_ctrs = boxes1.get_centers()
    boxes2_ctrs = boxes2.get_centers()
    l1_dist = torch.abs(boxes1_ctrs.view(-1, 1, 2) - boxes2_ctrs).sum(dim=-1)
    return l1_dist
