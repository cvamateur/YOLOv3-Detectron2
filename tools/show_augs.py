from detectron2.data import MetadataCatalog, build_detection_train_loader
from detectron2.utils.visualizer import Visualizer
from detectron2.structures import Instances

from yolov3.config import get_cfg
from yolov3.data import YOLODatasetMapper

import cv2
import numpy as np


def inverse_transform(img):
    img = img.permute(1, 2, 0).numpy()
    img = np.uint8(img)
    return img


def main():
    cfg = get_cfg()
    cfg.merge_from_file("configs/PascalVOC-Detection/yolov3_stage1_aug.yaml")
    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.DATALOADER.NUM_WORKERS = 0

    is_train = True
    do_mosaic = cfg.INPUT.MOSAIC.ENABLED
    ds_name = cfg.DATASETS.TRAIN[0] if is_train else cfg.DATASETS.TEST[0]

    metadata = MetadataCatalog.get(ds_name)
    mapper = YOLODatasetMapper(cfg, is_train)
    ds_loader = build_detection_train_loader(cfg, mapper=mapper)

    winName = "Image"
    num_show = 35
    for i, data_dict in enumerate(ds_loader):
        if i >= num_show:
            cv2.destroyAllWindows()
            break
        if i == 0:
            cv2.namedWindow(winName)

        data_dict = data_dict[0]
        image = inverse_transform(data_dict["image"])
        print(f"{i} Image shape:", image.shape)
        if is_train:
            ins = data_dict["instances"]
            ins.pred_boxes = ins.gt_boxes
            ins.pred_classes = ins.gt_classes
            vis = Visualizer(image, metadata)
            image = vis.draw_instance_predictions(ins).get_image()

            if do_mosaic:
                image = cv2.resize(image, (960, 960), cv2.INTER_LINEAR)

        cv2.imshow(winName, image[..., ::-1])
        cv2.waitKey(0)


if __name__ == '__main__':
    main()