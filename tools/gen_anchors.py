import argparse
import random
import time
import torch
import numpy as np

from detectron2.data import get_detection_dataset_dicts
from detectron2.structures.boxes import BoxMode

from yolov3.config import get_cfg
from yolov3.kmeans import KMeans, iou_distances

SCALE = 32.0
cuda_available = torch.cuda.is_available()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", default="configs/anc_gen.yaml", help="yaml config file.")
    parser.add_argument("--n_clusters", type=int, default=9, help="Number of clusters.")
    parser.add_argument("--init", default="k-means++", help="Initialization method in kmeans.")
    parser.add_argument("--input_size", type=int, default=608, help="Width and height of input image.")
    parser.add_argument("--cuda", action="store_true", help="Use GPU.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility.")
    parser.add_argument("--n_attempts", type=int, default=1, help="#kmeans to try.")
    parser.add_argument("--verbose", action="store_true", help="Print details during kmeans.")
    return parser.parse_args()


def main(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if (args.cuda and cuda_available) else "cpu")
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)

    bboxes = _get_all_boxes(cfg.DATASETS.TRAIN, args.input_size, device)
    for i in range(args.n_attempts):
        anchors = _cluster_bboxes(bboxes, args.init, args.n_clusters, args.verbose)
        _print_from_small_to_large(anchors)


def _get_all_boxes(dataset_names, input_size=416, device=torch.device("cpu")):
    print("info: load datasets from disk")
    dataset_dicts = get_detection_dataset_dicts(
        dataset_names, filter_empty=False, check_consistency=False)
    all_boxes = []
    num_suppressed = 0
    for info in dataset_dicts:
        rw = info["width"] / input_size
        rh = info["height"] / input_size
        img_boxes = []
        for ann in info["annotations"]:
            if ann["bbox_mode"] == BoxMode.XYXY_ABS:
                xmin, ymin, xmax, ymax = ann["bbox"]
                width = (xmax - xmin)
                height = (ymax - ymin)
            else:  # BoxMode.XYWH_ABS:
                assert ann["bbox_mode"] == BoxMode.XYWH_ABS
                width, height = ann["bbox"][2:4]

            # suppress invalid bboxes
            if width <= 0.0 or height <= 0.0:
                num_suppressed += 1
            else:
                img_boxes.append([width / rw, height / rh])
        all_boxes.extend(img_boxes)
    assert len(all_boxes), "empty bboxes"
    print(f"info: load {len(all_boxes)} bboxes from dataset.")
    print(f"info: drop {num_suppressed} bboxes because that are too small.")
    all_boxes = torch.tensor(all_boxes, dtype=torch.float32, device=device)
    all_boxes /= SCALE
    return all_boxes


def _cluster_bboxes(bboxes, init, n_clusters, verbose):
    print("info: run kmeans on dataset")
    tik = time.time()
    kmeans = KMeans(n_clusters, dist_fn=iou_distances, init=init, verbose=verbose)
    kmeans.fit(bboxes)
    tok = time.time()
    print(f"info: kmeans time collapsed: {tok - tik:.1f} s")
    return kmeans.cluster_centers_


def _print_from_small_to_large(anchors):
    if isinstance(anchors, torch.Tensor):
        anchors = anchors.detach().cpu().numpy() * SCALE
    anchors = np.int32(anchors).tolist()
    anchors = sorted(anchors, key=lambda x: x[0] * x[1])
    print("info: generated anchors\n", anchors)


if __name__ == '__main__':
    args = get_args()
    print(args)
    main(args)
