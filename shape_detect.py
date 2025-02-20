import cv2
import numpy as np
from sklearn.linear_model import RANSACRegressor

from simplification import morphology_diff
import matplotlib.pyplot as plt


def detect_shape(img_gray):
    edges = cv2.Canny(img_gray, 70, 200)
    plt.imshow(edges, cmap="gray")
    plt.title("Detected Edges")
    plt.axis("off")
    plt.show()
    drawing_paper = np.ones_like(edges)
    lines = cv2.HoughLinesP(
        img_gray, 1, np.pi / 180, threshold=100, minLineLength=30, maxLineGap=10
    )
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(drawing_paper, (x1, y1), (x2, y2), 0, 4)
    return drawing_paper


def contours(img_gray):
    edges = cv2.Canny(img_gray, 50, 150)
    # edges_inversed = cv2.threshold(
    #     img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    # )[1]
    # edges = 255 - edges_inversed
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    drawing_paper = np.ones_like(img_gray)
    for contour in contours:
        points = contour.squeeze()  # (N, 1, 2) -> (N, 2)
        if len(points.shape) != 2 or points.shape[0] < 2:
            continue

        # X, Y 분리
        X = points[:, 0].reshape(-1, 1)
        Y = points[:, 1]

        # RANSAC 적용
        ransac = RANSACRegressor()
        ransac.fit(X, Y)

        # 직선 방정식 얻기
        line_X = np.array([X.min(), X.max()]).reshape(-1, 1)
        line_Y = ransac.predict(line_X)

        # 직선 그리기 (빨간색)
        cv2.line(
            drawing_paper,
            (int(line_X[0].item()), int(line_Y[0].item())),
            (int(line_X[1].item()), int(line_Y[1].item())),
            30,
            2,
        )
    return drawing_paper


if __name__ == "__main__":
    image_path = "C:/Users/UserK/Desktop/fin/fin_purple_img.jpg"
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_gray, _, _ = morphology_diff(image)
    shape_image = detect_shape(img_gray)
    contour_image = contours(img_gray)
    plt.imshow(contour_image, cmap="gray")
    plt.title("Detected Lines")
    plt.axis("off")
    plt.show()
