from ultralytics import YOLO

# YOLO 모델 로드 (사전 학습된 가중치)
model = YOLO('yolov10n.pt')

# 학습 시작
model.train(data='coco.yaml', epochs=50, batch=16, imgsz=640)

# 검증 실행
results = model.val(data='coco.yaml', batch=16)

# 검증 결과 저장 위치 확인
print(results.save_dir)

# 추론 실행
results = model.predict(source='/coco_dataset/val2017', save=True, save_txt=True)

import matplotlib.pyplot as plt
import cv2
from glob import glob

# 추론 결과 이미지 불러오기
result_images = glob('runs/detect/exp/*.jpg')

# 이미지 출력
for img_path in result_images:
    img = cv2.imread(img_path)
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
