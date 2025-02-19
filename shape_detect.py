import cv2
import numpy as np

from simplification import morphology_diff
import matplotlib.pyplot as plt


# 이미지 로드
image_path = "C:/Users/UserK/Desktop/fin/fin_img.jpg"
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
img_gray, _, _ = morphology_diff(image)
# img_gray = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)
height, width = img_gray.shape

edges = cv2.Canny(img_gray, 50, 150)
mask = np.ones_like(edges)
lines = cv2.HoughLinesP(
    edges, 1, np.pi / 180, threshold=100, minLineLength=30, maxLineGap=10
)
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(mask, (x1, y1), (x2, y2), 0, 2)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(edges, cmap="gray")
plt.title("Morphological Transformation")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(mask, cmap="gray")
plt.title("Detected Lines")
plt.axis("off")

plt.show()
