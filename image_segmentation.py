import torch
import numpy as np
import cv2
from segment_anything import sam_model_registry, SamPredictor
from PIL import Image
import time

start_time = time.time()
# 모델 로드 (vit_b: 가벼운 모델)
device = "cuda" if torch.cuda.is_available() else "cpu"
sam = sam_model_registry["vit_b"](checkpoint="C:/Users/UserK/Desktop/sam_vit_b.pth").to(
    device
)
predictor = SamPredictor(sam)
end_time = time.time()
# 이미지 로드
image_path = "./image/cloth4.jpg"
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# SAM에 이미지 입력
predictor.set_image(image)

# 특정 좌표 클릭하여 객체 선택 (x, y는 수동 입력 또는 자동 탐색 가능)
input_point = np.array([[image.shape[1] // 2, image.shape[0] // 2]])  # 중앙 선택
input_label = np.array([1])  # 1: 객체 선택

# 마스크 예측
masks, _, _ = predictor.predict(
    point_coords=input_point, point_labels=input_label, multimask_output=False
)

# 가장 확신이 높은 마스크 선택
mask = masks[0]

# 객체 크기에 맞춰 크롭하기 위해 바운딩 박스 계산
y_indices, x_indices = np.where(mask)  # 마스크에서 객체 영역 찾기
y_min, y_max = y_indices.min(), y_indices.max()
x_min, x_max = x_indices.min(), x_indices.max()

# 객체 크기에 맞춰 크롭
cropped_image = image[y_min:y_max, x_min:x_max]


elapsed_time = end_time - start_time
print(f"작업에 걸린 시간: {elapsed_time} 초")
# 저장 (배경 없는 이미지)
output_path = "cropped_template.png"
cv2.imwrite(output_path, cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR))
print(f"✅ 배경 제거 및 크롭 완료! {output_path} 저장됨")
