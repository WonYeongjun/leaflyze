import cv2
import numpy as np
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import torch
import time

# 시작 시간 기록
start_time = time.time()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h.pth").to(device)

mask_generator = SamAutomaticMaskGenerator(sam)

# 이미지 불러오기
image = cv2.imread("./image/pink/fin_cal_img_20250207_141129.jpg")

# SAM으로 마스크 생성

masks = mask_generator.generate(image)

# 가장 큰 마스크 찾기 (천일 가능성이 높음)
largest_mask = max(masks, key=lambda x: x["segmentation"].sum())

# 마스크 적용 (배경 제거)
mask = largest_mask["segmentation"].astype(np.uint8) * 255
result = cv2.bitwise_and(image, image, mask=mask)

# 결과 이미지 크기 조정
scale_percent = 50  # 이미지 크기를 50%로 줄이기
width = int(result.shape[1] * scale_percent / 100)
height = int(result.shape[0] * scale_percent / 100)
dim = (width, height)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"작업에 걸린 시간: {elapsed_time} 초")
# 이미지 크기 조정
resized_result = cv2.resize(result, dim, interpolation=cv2.INTER_AREA)
cv2.imshow("Segmented Fabric", resized_result)
cv2.waitKey(0)
cv2.destroyAllWindows()
