import cv2
import numpy as np

from simplification import morphology_diff
import matplotlib.pyplot as plt


def detect_shape(img_gray):
    height, width = img_gray.shape

    edges = cv2.Canny(img_gray, 50, 150)
    drawing_paper = np.ones_like(edges)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=100, minLineLength=30, maxLineGap=10
    )
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(drawing_paper, (x1, y1), (x2, y2), 0, 4)
    return drawing_paper


if __name__ == "__main__":
    image_path = "C:/Users/UserK/Desktop/fin/fin_img.jpg"
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_gray, _, _ = morphology_diff(image)
    shape_image = detect_shape(img_gray)
    plt.imshow(shape_image, cmap="gray")
    plt.title("Detected Lines")
    plt.axis("off")
    plt.show()
