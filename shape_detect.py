import cv2
import numpy as np
from sklearn.linear_model import RANSACRegressor


from simplification import morphology_diff
import matplotlib.pyplot as plt
from line_merge import merge_all_segments
import sys

sys.setrecursionlimit(2000)  # 🔥 재귀 호출 한도 증가


def Hough(edges):
    drawing_paper = np.zeros_like(edges)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=100, minLineLength=30, maxLineGap=10
    )
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(drawing_paper, (x1, y1), (x2, y2), 255, 4)
    return drawing_paper


def RANSAC(edges):
    edges = cv2.blur(edges, (5, 5))

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    drawing_paper = np.zeros_like(edges)

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


def get_lines_not_in_merged(merged_lines, lines):
    # merged_lines와 lines을 (N, 4) 형태로 numpy 배열로 변환
    merged_lines = np.array(merged_lines)
    lines = np.array(lines)

    # 각 선분이 lines에 속하지 않는지 확인 (벡터화)
    # merged_lines와 lines의 각 선분을 비교하여 일치하지 않는 선분을 찾음
    merged_lines_set = set(map(tuple, merged_lines.tolist()))
    lines_set = set(map(tuple, lines.tolist()))

    non_matching_lines = np.array(
        [line for line in merged_lines if tuple(line) not in lines_set]
    )

    return non_matching_lines


def line_detector(img_gray):
    # 선 감지기 생성
    detector = cv2.createLineSegmentDetector()
    # 선 감지
    lines, _, _, _ = detector.detect(img_gray)
    lines = lines.squeeze()
    print(len(lines))
    merged_lines = merge_all_segments(lines)
    # 감지된 선 그리기
    drawing_paper = np.ones_like(img_gray) * 255
    print("그림 시작")
    print(len(merged_lines))
    for line in merged_lines:
        x1, y1, x2, y2 = line
        cv2.line(drawing_paper, (int(x1), int(y1)), (int(x2), int(y2)), 0, 2)
    output_img = drawing_paper
    output_img_binary = cv2.threshold(
        output_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )[1]
    output_img_binary_inversed = 255 - output_img_binary
    return output_img_binary_inversed


def line_detector_without_merge(img_gray):
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


def detect_SED(img_bgr):
    model = "model.yml.gz"  # 미리 학습된 모델 필요
    edge_detector = cv2.ximgproc.createStructuredEdgeDetection(model)
    img_bgr = cv2.resize(img_bgr, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    edges = edge_detector.detectEdges(np.float32(img_bgr) / 255.0)
    edges = edges * 255
    edges = edges.astype(np.uint8)
    edges = cv2.resize(edges, None, fx=0.5, fy=0.5)
    return edges


def sobel(img):
    # Sobel 필터 적용 (X, Y 방향)
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    sobel = cv2.magnitude(sobel_x, sobel_y)  # 크기 계산

    sobel = np.uint8(sobel)  # 정수형 변환
    return sobel


def laplacian(img):
    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    laplacian = np.uint8(np.abs(laplacian))  # 절댓값 변환
    return laplacian


def scharr(img):
    scharr_x = cv2.Scharr(img, cv2.CV_64F, 1, 0)
    scharr_y = cv2.Scharr(img, cv2.CV_64F, 0, 1)
    scharr = cv2.magnitude(scharr_x, scharr_y)

    scharr = np.uint8(scharr)
    return scharr


def canny(img):
    canny = cv2.Canny(img, 150, 250)
    return canny


if __name__ == "__main__":
    file_name = "black2"
    image_path = f"C:/Users/UserK/Desktop/fin/{file_name}.jpg"
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_gray, _ = morphology_diff(image)
    # img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    line_image = line_detector(img_gray)
    # contour_image = contours(line_image)
    # cv2.imwrite("output_image.png", line_image)
    plt.imshow(line_image, cmap="gray")
    plt.title("Detected Lines")
    plt.show()
