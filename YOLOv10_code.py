from ultralytics import YOLOv10
import cv2

# YOLO 모델 로드 및 validation
model = YOLOv10('yolov10n.pt')

model.val(data='coco.yaml', batch=256)

# Training
model.train(data='coco.yaml', epochs=50, batch=16, imgsz=640)

model = YOLOv10('yolov10n.pt')

model.predict()