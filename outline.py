import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. 이미지 불러오기 (이미지가 없으면 임의의 회색조 이미지 생성)


def center_emphasize(img):
    rows, cols = img.shape

    X, Y = np.meshgrid(np.arange(cols), np.arange(rows))
    center_x, center_y = cols / 2, rows / 2
    # 각 픽셀의 중심으로부터의 거리 계산
    distance = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)
    # 최대 거리 (이미지의 모서리까지의 거리)
    max_distance = np.sqrt(center_x**2 + center_y**2)

    # 3. 가중치 마스크 생성: 중심에서는 1에 가까워지고, 외곽에서는 0에 가까워짐.
    # 여기서는 선형 보간 후 제곱을 취해 중심 강조 효과를 줍니다.
    mask = 1 - (distance / max_distance)
    mask = mask**2  # 중심 부근을 더욱 강조

    # 4. 원본 이미지에 마스크를 곱하여 픽셀값 조정
    adjusted = img * mask
    adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)
    return adjusted


if __name__ == "__main__":
    img = cv2.imread("./failure/SED.png", cv2.IMREAD_GRAYSCALE)
    adjusted = center_emphasize(img)
    # 5. 결과 출력
    plt.imshow(adjusted, cmap="gray")
    plt.show()
