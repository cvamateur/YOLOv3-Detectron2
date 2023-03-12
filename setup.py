from setuptools import find_packages, setup
import torch

torch_ver = [int(x) for x in torch.__version__.split(".")[:2]]
assert torch_ver >= [1, 6], "Requires PyTorch >= 1.6"

setup(
    name="yolov3",
    version="0.0.1",
    author="liuquan",
    url="https://github.com/cvamateur/YOLOv3-Detectron2",
    description="YOLOV3 implementation on Detectron2",
    packages=find_packages(exclude=("configs", "datasets")),
    python_requires=">=3.7"
)
