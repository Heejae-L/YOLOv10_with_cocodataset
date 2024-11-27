from ultralytics import YOLO

# YOLOv10 모델 로드
model = YOLO('yolov10n.pt')

# COCO 데이터셋 검증
results = model.val(
    data='/Users/LeeHeejae/projects/sw_term_project/coco.yaml',
    conf=0.25,  # Confidence threshold
    iou=0.45,   # IoU threshold
    save=True   # 검증 결과 저장
)

import pandas as pd

# result.csv 파일 경로
result_csv_path = "/Users/LeeHeejae/projects/sw_term_project/runs/detect/train/results.csv"  # YOLO validation 디렉토리에 생성된 CSV 파일 경로

# CSV 파일 읽기
try:
    results_df = pd.read_csv(result_csv_path)
    print("CSV 파일을 성공적으로 읽었습니다.")
except FileNotFoundError:
    print(f"파일을 찾을 수 없습니다: {result_csv_path}")
    results_df = None

# 데이터 확인
if results_df is not None:
    print("데이터 프레임의 첫 5행:")
    print(results_df.head())
    print("\n데이터 컬럼 정보:")
    print(results_df.columns)
