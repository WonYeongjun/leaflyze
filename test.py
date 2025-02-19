import cv2
import torch

print(cv2.cuda.getCudaEnabledDeviceCount())
print("Torch CUDA available:", torch.cuda.is_available())  # CUDA 사용 가능 여부
print("Torch CUDA device count:", torch.cuda.device_count())  # 사용 가능한 GPU 개수
print(
    "Torch current device:", torch.cuda.current_device()
)  # 현재 사용 중인 디바이스 ID
print(
    "Torch device name:",
    torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU",
)  # GPU 이름
