import cv2
import numpy as np

# Read the image
image = cv2.imread("./image/marker_ideal.jpg", 0)
# image = 255 - image
# Define a kernel
kernel = np.ones((31, 31), np.uint8)

# Apply morphological operations
erosion = cv2.erode(image, kernel, iterations=1)
dilation = cv2.dilate(image, kernel, iterations=1)
opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

# Save or display the results
cv2.imwrite("erosion.jpg", erosion)
cv2.imwrite("dilation.jpg", dilation)
cv2.imwrite("opening.jpg", opening)
cv2.imwrite("closing.jpg", closing)
