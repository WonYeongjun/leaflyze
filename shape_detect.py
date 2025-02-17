import cv2
import numpy as np

# 이미지 로드
image = cv2.imread("./image/pink/fin_cal_img_20250207_141201.jpg")

# 이미지를 BGR에서 RGB로 변환
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 이미지를 2D 배열로 변환 (각 픽셀을 한 행으로)
pixels = image_rgb.reshape((-1, 3))
# K-means 군집화
k = 10  # 군집의 수 (원하는 군집 수로 설정)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
_, labels, centers = cv2.kmeans(
    np.float32(pixels), k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
)

# 군집화된 색상 (centers는 각 군집의 중심 색상)
centers = np.uint8(centers)

# 각 픽셀을 군집화된 색상으로 교체
segmented_image = centers[labels.flatten()]

# 이미지를 원래 모양으로 다시 리셰이프
segmented_image = segmented_image.reshape(image_rgb.shape)

# 결과 이미지 출력
import matplotlib.pyplot as plt

plt.imshow(segmented_image)
plt.axis("off")  # 축을 숨김
plt.show()
