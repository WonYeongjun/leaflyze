from ultralytics import YOLO

# YOLO 모델 불러오기 (사전 학습된 모델 사용)
model = YOLO("yolov8n.pt")

# 학습 실행
model.train(data="My First Project.v2i.yolov8/data.yaml", epochs=50, imgsz=640)

# 학습된 모델 파일은 자동으로 저장됨!
# 경로: runs/detect/train/weights/best.pt
