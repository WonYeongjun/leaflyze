import cv2
import numpy as np
from sklearn.cluster import KMeans
from simplification import morphology_diff
import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor

# 이미지 로드
image_path = "./image/pink/fin_cal_img_20250207_141201.jpg"
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
img_gray, _ = morphology_diff(image)

height, width = img_gray.shape

edges = cv2.Canny(img_gray, 20, 80)

kernel = np.ones((5, 5), np.uint8)
morph = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# for contour in contours:
#     points = contour.squeeze()  # (N, 1, 2) -> (N, 2)
#     if len(points.shape) != 2 or points.shape[0] < 2:
#         continue

#     # X, Y 분리
#     X = points[:, 0].reshape(-1, 1)
#     Y = points[:, 1]

#     # RANSAC 적용
#     ransac = RANSACRegressor()
#     ransac.fit(X, Y)
#     line_X = np.array([X.min(), X.max()]).reshape(-1, 1)
#     line_Y = ransac.predict(line_X)
#     cv2.line(
#         image_rgb,
#         (int(line_X[0].item()), int(line_Y[0].item())),
#         (int(line_X[1].item()), int(line_Y[1].item())),
#         (0, 0, 255),
#         2,
#     )
img_binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
lines = cv2.HoughLinesP(
    img_binary, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10
)
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(image_rgb, (x1, y1), (x2, y2), (255, 255, 0), 2)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(img_binary, cmap="gray")
plt.title("Morphological Transformation")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(image_rgb)
plt.title("Detected Lines")
plt.axis("off")

plt.show()
