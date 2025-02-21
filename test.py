import cv2
import numpy as np
import matplotlib.pyplot as pyplot
from shape_detect import line_detector, Hough, RANSAC
from simplification import morphology_diff

file_name = "black_back"
image_path = f"C:/Users/UserK/Desktop/fin/{file_name}.jpg"
img = cv2.imread(image_path)
img_gray, _ = morphology_diff(img)

edges = cv2.Canny(img_gray, 200, 300)
edges = RANSAC(edges) * 255

cv2.imwrite("output_image.png", edges)
