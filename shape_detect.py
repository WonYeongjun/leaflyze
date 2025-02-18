import cv2
import numpy as np
from sklearn.cluster import KMeans
from simplication import morphlogy_diff
import matplotlib.pyplot as plt

# 이미지 로드
image_path = "./image/pink/fin_cal_img_20250207_141201.jpg"
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
img_gray, _ = morphlogy_diff(image)
plt.imshow(img_gray, cmap="gray")
plt.axis("off")  # 축 숨기기
plt.show()
# 이미지 크기
height, width = img_gray.shape
# 컨투어 검출
# 컨투어 검출
contours, _ = cv2.findContours(img_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# RANSAC을 이용한 선형 검출
segmented_image = np.zeros_like(img_gray)
for contour in contours:
    if len(contour) >= 2:  # 최소 두 개의 점이 필요
        points = contour.reshape(-1, 2)
        model_ransac = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)
        vx, vy, x, y = model_ransac.flatten()
        lefty = int((-x * vy / vx) + y)
        righty = int(((width - x) * vy / vx) + y)
        cv2.line(segmented_image, (width - 1, righty), (0, lefty), 255, 1)

# 결과 이미지
segmented_image_rgb = cv2.cvtColor(segmented_image, cv2.COLOR_GRAY2RGB)


plt.imshow(segmented_image_rgb)
plt.axis("off")  # 축 숨기기
plt.show()
