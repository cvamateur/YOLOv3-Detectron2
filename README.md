# YOLOv3-Detectron2

A simple YOLOv3/YOLOv3-Tiny implementation on facebook's **Detectron2** framework. 

## Getting Started
- Install **Detectron2** following official [installation instructions](https://detectron2.readthedocs.io/en/latest/tutorials/install.html).
- Install YOLOv3-Detectron2
  ```
  python setup.py develop
  ```
  
- Run inference
  ```
  # use YOLOv3 official weights 
  python tools/detect.py -i ./imgs/messi.jpg -o ./output/
  ```
  
- Train
  ```
  # train on Pascal-VOC dataset
  python tools/train_net.py --config-file configs/PascalVOC-Detection/yolov3_stage1.yaml 
  ```
  
- Generating anchor boxes on dataset
  ```
  python tools/gen_anchors.py --config-file configs/gen_anc.yaml --n_clusters 9
  ```

### COCO Detection Baseline
Load darknet official weights and evaluate mAP on COCO:

|      Model      |   AP   |  AP50  |  AP75  |  APs   |  APm   |  APl   |
|:---------------:|:------:|:------:|:------:|:------:|:------:|:------:|
|   YOLOv3-608    | 32.734 | 56.950 | 34.052 | 24.372 | 37.572 | 40.170 |
| YOLOv3-Tiny-416 | 9.175  | 18.744 | 7.977  | 0.246  | 9.591  | 22.219 |
