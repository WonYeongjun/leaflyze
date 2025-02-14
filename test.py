import torch

print("PyTorch 버전:", torch.__version__)
print("CUDA 사용 가능 여부:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("사용 가능한 GPU 개수:", torch.cuda.device_count())
    print("현재 사용 중인 GPU:", torch.cuda.get_device_name(0))
else:
    print("GPU를 사용할 수 없습니다.")
