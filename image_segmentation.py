import torch
import numpy as np
import cv2
from segment_anything import sam_model_registry, SamPredictor
from PIL import Image
import time


def point_of_interest(image):
    # 모델 로드 (vit_b: 가벼운 모델)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry["vit_b"](
        checkpoint="C:/Users/UserK/Desktop/sam_vit_b.pth"
    ).to(device)
    predictor = SamPredictor(sam)

    # 이미지 로드

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # SAM에 이미지 입력
    predictor.set_image(image)

    # 특정 좌표 클릭하여 객체 선택 (x, y는 수동 입력 또는 자동 탐색 가능)
    input_point = np.array([[image.shape[1] // 2, image.shape[0] // 2]])  # 중앙 선택
    input_label = np.array([1])  # 1: 객체 선택

    # 마스크 예측
    masks, sinre, _ = predictor.predict(
        point_coords=input_point, point_labels=input_label, multimask_output=True
    )
    print(sinre)
    mask = masks[0]
    print(mask.shape)
    # 객체 크기에 맞춰 크롭하기 위해 바운딩 박스 계산
    y_indices, x_indices = np.where(mask)  # 마스크에서 객체 영역 찾기
    y_min, y_max = y_indices.min(), y_indices.max()
    x_min, x_max = x_indices.min(), x_indices.max()

    # 객체 크기에 맞춰 크롭
    cropped_image = image[y_min:y_max, x_min:x_max]
    return cropped_image


if __name__ == "__main__":
    image_path = "./image/pink/fin_cal_img_20250207_141229.jpg"
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cropped_image = point_of_interest(image)
    # 크기 조정 (예: 50% 축소)
    scale_percent = 50
    width = int(cropped_image.shape[1] * scale_percent / 100)
    height = int(cropped_image.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized_image = cv2.resize(cropped_image, dim, interpolation=cv2.INTER_AREA)

    cv2.imshow("resized", resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
