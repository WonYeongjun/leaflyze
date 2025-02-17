import cv2
import numpy as np
from sklearn.cluster import KMeans

# 이미지 로드
image_path = "./image/pink/fin_cal_img_20250207_141201.jpg"
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 이미지 크기
height, width, _ = image.shape

# 색상과 위치 정보를 결합
pixels = image_rgb.reshape(-1, 3)
positions = np.indices((height, width)).reshape(2, -1).T
positions = positions / 5000
data = np.hstack((pixels, positions))

# K-means 클러스터링
k = 4  # 군집의 수
kmeans = KMeans(n_clusters=k, random_state=0).fit(data)
labels = kmeans.labels_

# 클러스터링 결과를 이미지로 변환
segmented_image = labels.reshape(height, width)

# 클러스터링 결과 시각화
unique_labels = np.unique(labels)
segmented_image_rgb = np.zeros_like(image_rgb)

for label in unique_labels:
    mask = segmented_image == label
    color = np.random.randint(0, 255, size=3)
    segmented_image_rgb[mask] = color

# 결과 이미지 저장 및 출력
import matplotlib.pyplot as plt

plt.imshow(segmented_image_rgb)
plt.axis("off")  # 축 숨기기
plt.show()
