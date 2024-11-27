from ultralytics import YOLO
import cv2

# YOLO 모델 로드
model = YOLO('yolov10n.pt')

video_path = 'soccer.mp4'

# OpenCV를 사용하여 비디오 파일 열기
cap = cv2.VideoCapture(video_path)

# 비디오가 열리지 않았다면 경고
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# 비디오 프레임을 읽기
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO 모델을 사용하여 프레임에서 객체 감지
    results = model(frame)  # results는 리스트로 반환

    # 결과로부터 바운딩 박스와 레이블 추출
    for result in results:  # results 내부의 각 결과 처리
        boxes = result.boxes  # 바운딩 박스 정보 가져오기
        for box in boxes:
            # 좌표 및 클래스 정보 추출
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # 좌표는 xyxy 형식
            conf = box.conf[0]  # 신뢰도
            cls_id = int(box.cls[0])  # 클래스 ID

            # 바운딩 박스 및 레이블 그리기
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{model.names[cls_id]} {conf:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 결과 영상 표시
    cv2.imshow('YOLO Detection', frame)

    # 'q'를 누르면 비디오 재생 중지
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()
