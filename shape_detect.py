import cv2
import numpy as np
from sklearn.linear_model import RANSACRegressor


from simplification import morphology_diff
import matplotlib.pyplot as plt


def detect_shape(img_gray):
    drawing_paper = np.zeros_like(img_gray)
    lines = cv2.HoughLinesP(
        img_gray, 1, np.pi / 180, threshold=50, minLineLength=20, maxLineGap=10
    )
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(drawing_paper, (x1, y1), (x2, y2), 255, 4)
    return drawing_paper


def contours(img_gray):

    contours, _ = cv2.findContours(img_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    drawing_paper = np.zeros_like(img_gray)

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


def detect_lines(img_gray):
    # 선 감지기 생성
    detector = cv2.createLineSegmentDetector()

    # 선 감지
    lines = detector.detect(img_gray)[0]
    print(lines[0, 0])
    # 감지된 선 그리기
    drawing_paper = np.ones_like(img_gray) * 255
    output_img = detector.drawSegments(drawing_paper, lines)
    output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2GRAY)
    output_img_binary = cv2.threshold(
        output_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )[1]
    output_img_binary_inversed = 255 - output_img_binary
    return output_img_binary_inversed


if __name__ == "__main__":
    image_path = "C:/Users/UserK/Desktop/fin/purple_back.jpg"
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_gray, _, _ = morphology_diff(image)
    # img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    line_image = detect_lines(img_gray)
    shape_image = detect_shape(line_image)
    # contour_image = contours(line_image)
    cv2.imwrite("output_image.png", line_image)
    plt.imshow(line_image, cmap="gray")
    plt.title("Detected Lines")
    plt.axis("off")
    plt.show()
