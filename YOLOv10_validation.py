from ultralytics import YOLO
import matplotlib.pyplot as plt
import os
import yaml

# # YOLO 모델 로드 및 validation
model = YOLO('yolov10n.pt')

model.val(data='/Users/LeeHeejae/projects/sw_term_project/coco.yaml', batch=256)

# 저장된 결과 파일 불러오기
exp_dir = '/Users/LeeHeejae/projects/sw_term_project/runs/val5/exp'  # validation 결과 폴더 (exp 번호 확인 후 수정 가능)
results_path = os.path.join(exp_dir, 'results.csv')

# 결과 확인 및 시각화
import pandas as pd
data = pd.read_csv(results_path)

# Precision-Recall Curve 시각화 (예시)
plt.figure(figsize=(10, 6))
plt.plot(data['precision'], data['recall'], label="Precision-Recall Curve")
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.grid()
plt.show()

# mAP 값 시각화
classes = data['class'].values  # 클래스 ID
map50 = data['mAP50'].values  # mAP@0.5
map5095 = data['mAP50-95'].values  # mAP@0.5:0.95

plt.figure(figsize=(10, 6))
plt.bar(classes, map50, alpha=0.7, label='mAP@0.5')
plt.bar(classes, map5095, alpha=0.7, label='mAP@0.5:0.95')
plt.xlabel('Class')
plt.ylabel('mAP')
plt.title('mAP by Class')
plt.legend()
plt.grid(axis='y')
plt.show()

from PIL import Image

# 저장된 예측 결과 이미지를 시각화
images_dir = os.path.join(exp_dir, 'images')
image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]

# 예제 이미지 확인
for image_file in image_files[:5]:  # 상위 5개 이미지만 확인
    image_path = os.path.join(images_dir, image_file)
    image = Image.open(image_path)
    plt.figure(figsize=(8, 6))
    plt.imshow(image)
    plt.axis('off')
    plt.title(f"Prediction: {image_file}")
    plt.show()

