from ultralytics import YOLO

model = YOLO('yolov10n.pt')

model.predict()