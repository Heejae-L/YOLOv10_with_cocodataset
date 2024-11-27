import torch
from torchvision.datasets import CocoDetection
from ultralytics import YOLO
from torchvision.transforms import transforms
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import cv2
import numpy as np

# COCO 데이터셋 경로 설정
coco_root = "/Users/LeeHeejae/projects/sw_term_project/coco_dataset"
coco_val_images = f"{coco_root}/val2017"
coco_val_annotations = f"{coco_root}/annotations/instances_val2017.json"

# 데이터셋 로더 정의
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((640, 640))  # YOLO 모델에 맞게 조정
])

coco_dataset = CocoDetection(
    root=coco_val_images,
    annFile=coco_val_annotations,
    transform=transform
)

# YOLOv10 모델 로드 (가정: pretrained 모델이 로드 가능)
model = YOLO('yolov10n.pt')  # YOLOv10 가중치 파일 경로
model.eval()

