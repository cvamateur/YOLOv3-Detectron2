# YOLOv3-Detectron2

A simple YOLOv3/YOLOv3-Tiny implementation on facebook's **Detectron2** framework.

The project code uses official yolov3 [weights](https://pjreddie.com/darknet/yolo/) trained on COCO dataset to initiate model weights.



## Getting Started
- Install **Detectron2** following official [installation instructions](https://detectron2.readthedocs.io/en/latest/tutorials/install.html).
- Install YOLOv3-Detectron2:
  ```
  python setup.py develop
  ```
  
- Run inference on custom image:
  ```
  # use YOLOv3 official weights 
  python tools/detect.py -i ./imgs/messi.jpg -o ./output/
  ```
  
- Use COCO pretrain weights to train YOLOv3 model on Pascal VOC dataset:
  ```
  # train on Pascal-VOC dataset
  python tools/train_net.py --config-file configs/PascalVOC-Detection/yolov3_stage1.yaml 
  ```
  
- Generate anchors by k-means clustering:
  ```
  python tools/gen_anchors.py --config-file configs/gen_anc.yaml --n_clusters 9
  ```

### COCO Detection Baseline
Load darknet official weights and evaluate mAP on COCO:

|      Model      |   AP   |  AP50  |  AP75  |  APs   |  APm   |  APl   |
|:---------------:|:------:|:------:|:------:|:------:|:------:|:------:|
|   YOLOv3-608    | 32.734 | 56.950 | 34.052 | 24.372 | 37.572 | 40.170 |
| YOLOv3-Tiny-416 | 9.175  | 18.744 | 7.977  | 0.246  | 9.591  | 22.219 |


### Detection Results
<img src="./imgs/messi_det.jpg" alt="messi" style="zoom:80%;" />