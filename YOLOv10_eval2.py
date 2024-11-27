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

# YOLO 모델 로드
model = YOLO('yolov10n.pt')  # YOLOv10 가중치 파일 경로
model.eval()

# COCO 클래스 이름 로드
coco = COCO(coco_val_annotations)
class_names = {k: v['name'] for k, v in coco.cats.items()}

def visualize(image, detections, class_names):
    img = image.permute(1, 2, 0).numpy() * 255
    img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR)

    for x1, y1, x2, y2, conf, cls_id in detections:
        label = f"{class_names.get(int(cls_id), 'unknown')}: {conf:.2f}"  # 기본값 'unknown'
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(img, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()


# val2017 전체 평가
for i in range(len(coco_dataset)):
    # 이미지와 라벨 가져오기
    image, target = coco_dataset[i]

    # YOLO 모델로 추론
    results = model(image.unsqueeze(0))  # 배치를 위해 차원 추가

    # 디텍션 정보 추출
    detections = results[0].boxes.data.cpu().numpy()  # (x1, y1, x2, y2, confidence, class)

    # 결과 시각화
    print(f"Processing image {i+1}/{len(coco_dataset)}")
    visualize(image, detections, class_names)
