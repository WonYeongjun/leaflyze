from ultralytics import YOLO
import torch


def train_model():
    # YOLO 모델 불러오기 (사전 학습된 모델 사용)
    model = YOLO("yolov8n.pt")

    torch.cuda.set_device(0)  # GPU 0번을 사용할 경우
    model.train(data="datasets.yolov8/data.yaml", epochs=50, device="cuda")


if __name__ == "__main__":
    # 모델 학습 함수 호출
    train_model()
