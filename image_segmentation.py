import torch
import numpy as np
import cv2
from segment_anything import sam_model_registry, SamPredictor
from PIL import Image
import time
import glob
import matplotlib.pyplot as plt
import os


def point_of_interest(image):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry["vit_b"](
        checkpoint="C:/Users/UserK/Desktop/sam_vit_b.pth"
    ).to(device)
    predictor = SamPredictor(sam)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image)
    input_point = np.array([[image.shape[1] // 2, image.shape[0] // 2]])  # 중앙 선택
    input_label = np.array([1])
    masks, sinre, _ = predictor.predict(
        point_coords=input_point, point_labels=input_label, multimask_output=True
    )

    mask = masks[np.argmax([np.sum(m) for m in masks])]
    kernel = np.ones((15, 15), np.uint8)
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    # 객체 크기에 맞춰 크롭하기 위해 바운딩 박스 계산
    y_indices, x_indices = np.where(mask)  # 마스크에서 객체 영역 찾기
    y_min, y_max = y_indices.min(), y_indices.max()
    x_min, x_max = x_indices.min(), x_indices.max()

    # 객체 크기에 맞춰 크롭
    cropped_image = image[y_min:y_max, x_min:x_max]
    return cropped_image


if __name__ == "__main__":
    color = "pink"
    image_files = glob.glob(f"./image/{color}/*.jpg")
    num_images = len(image_files)  # 총 이미지 개수
    fig, axes = plt.subplots(num_images, 2, figsize=(8, num_images * 4))
    for i, file in enumerate(image_files):
        image = cv2.imread(file)
        cropped_image = point_of_interest(image)  # 관심영역 추출
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # RGB 변환
        # 원본 이미지 (1열)
        axes[i, 0].imshow(image)
        axes[i, 0].axis("off")

        # 크롭된 이미지 (2열)
        axes[i, 1].imshow(cropped_image, cmap="gray")  # 그레이스케일이면 cmap 설정
        axes[i, 1].axis("off")

    plt.tight_layout()  # 레이아웃 조정
    plt.savefig(f"image_comparison_{color}.png", dpi=300)  # 이미지 저장
    plt.show()
