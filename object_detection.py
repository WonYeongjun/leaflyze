from ultralytics import YOLO
import cv2

# YOLOv8 기본 모델 다운로드 (첫 실행 시 자동 다운로드됨)
model = YOLO("./runs\detect/train19/weights/best.pt")  # "n"은 nano 버전으로 가벼움

# 이미지 로드
image_path = "./image/cloth5.jpg"
# image_path = "./datasets.yolov8/train/images/fin_cal_img_20250207_132352_jpg.rf.e04b30e5d29396cf057a95d248e910c5.jpg"
image = cv2.imread(image_path)

# YOLO 실행
results = model(image_path)

# 결과 시각화
for r in results:
    for box in r.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        confidence = box.conf[0]
        label = f"{r.names[int(box.cls[0])]} {confidence:.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(
            image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
        )
# 이미지 크기 조정
resized_image = cv2.resize(image, (800, 600))

# 결과 시각화
cv2.imshow("YOLO Detection", resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
