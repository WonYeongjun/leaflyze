import cv2

img = cv2.imread("./image/white/fin_cal_img_20250207_132352.jpg")
img_filtered = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
enhanced_img = cv2.convertScaleAbs(img, alpha=2.0, beta=0)
import matplotlib.pyplot as plt

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (31, 31))
img_morphed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)


difference = cv2.absdiff(img, img_morphed)
difference_gray = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
difference_gray = 255 - difference_gray
plt.figure(figsize=(10, 5))
plt.title("Difference Image")
plt.imshow(difference_gray, cmap="gray")
plt.axis("off")
plt.show()
