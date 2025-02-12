import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl
from calibration_marker_func import invariant_match_template  # ,template_crop
import time
import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = json.load(f)

start_time = time.time()


def correct_perspective(image_path):

    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Cannot load image at {image_path}")
        return

    h, w, _ = image.shape
    image_center = np.array([w / 2, h / 2])

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    corners, ids, _ = detector.detectMarkers(gray)

    desired_ids = [12, 18, 27, 5]

    if ids is not None:
        marker_points = []

        for marker_id in desired_ids:
            for i, detected_id in enumerate(ids):
                if detected_id[0] == marker_id:
                    marker_corners = corners[i][0]
                    closest_corner = min(
                        marker_corners, key=lambda pt: np.linalg.norm(pt - image_center)
                    )
                    marker_points.append(closest_corner)
                    break

        if len(marker_points) == 4:
            pts1 = np.array(marker_points, dtype="float32")
            width, height = 4200, 2970
            pts2 = np.array(
                [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
                dtype="float32",
            )

            matrix, _ = cv2.findHomography(pts1, pts2)

            dst = cv2.warpPerspective(image, matrix, (width, height))
        else:
            print("Not enough markers detected to calculate perspective.")
    else:
        print("No ArUco markers detected.")

    return dst


if __name__ == "__main__":
    test_image_path = config["pc_file_path"]

    img_bgr = correct_perspective(test_image_path)

    threshold = 130
    template_image_path = config["pc_marker_file_path"]
    template_bgr = plt.imread(template_image_path)
    template_bgr = cv2.resize(
        template_bgr, (0, 0), fx=0.27, fy=0.27
    )  # 템플릿 사이즈 조절(초기 설정 필요)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
    template_gray = cv2.cvtColor(template_bgr, cv2.COLOR_RGB2GRAY)
    _, template_gray = cv2.threshold(template_gray, threshold, 255, cv2.THRESH_BINARY)
    height, width = template_gray.shape
    points_list = invariant_match_template(
        grayimage=img_gray,
        graytemplate=template_gray,
        method="TM_CCOEFF",
        matched_thresh=0.35,
        rot_range=[-10, 10],
        rot_interval=2,
        scale_range=[90, 110],
        scale_interval=2,
        rm_redundant=True,
        minmax=True,
    )
    fig, ax = plt.subplots(1)
    plt.gcf().canvas.manager.set_window_title("Template Matching Results: Rectangles")
    ax.imshow(img_rgb)
    print(len(points_list))
    centers_list = []
    real_point = []
    for point_info in points_list:
        point = point_info[0]
        angle = point_info[1]
        scale = point_info[2]
        print(
            f"No.{str(points_list.index(point_info))} matched point: {point}, angle: {angle}, scale: {scale}, score: {point_info[3]}"
        )
        centers_list.append([point, scale])
        plt.scatter(
            point[0] + (width / 2) * scale[0] / 100,
            point[1] + (height / 2) * scale[1] / 100,
            s=20,
            color="red",
        )
        idx = (
            point[0] + (width / 2) * scale[0] / 100,
            point[1] + (height / 2) * scale[1] / 100,
        )
        real_point.append([idx])
        plt.scatter(point[0], point[1], s=20, color="green")
        rectangle = patches.Rectangle(
            (point[0], point[1]),
            width * scale[0] / 100,
            height * scale[1] / 100,
            color="red",
            alpha=0.50,
            label="Matched box",
        )
        plt.text(
            point[0],
            point[1] - 10,
            str(points_list.index(point_info)),
            color="blue",
            fontsize=12,
            weight="bold",
        )

        transform = (
            mpl.transforms.Affine2D().rotate_deg_around(
                point[0] + width / 2 * scale[0] / 100,
                point[1] + height / 2 * scale[1] / 100,
                angle,
            )
            + ax.transData
        )
        rectangle.set_transform(transform)
        ax.add_patch(rectangle)
        plt.legend(handles=[rectangle])

    end_time = time.time()
    plt.show()

    indices = input("원하는 점의 인덱스 4개를 입력하세요 (쉼표로 구분): ")
    indices = list(map(int, indices.split(",")))

    if len(indices) != 4:
        print("4개의 인덱스를 입력해야 합니다.")
    else:
        selected_points = [real_point[i][0] for i in indices]
        matrix = np.array(selected_points)
        print("선택된 점들의 좌표 행렬:")
        print(matrix)

        # 좌표를 오른쪽 위, 왼쪽 위, 왼쪽 아래, 오른쪽 아래 순서로 정렬
        def sort_points(points):
            points = sorted(points, key=lambda x: x[0])
            left_points = points[:2]
            right_points = points[2:]
            left_points = sorted(left_points, key=lambda x: x[1])
            right_points = sorted(right_points, key=lambda x: x[1])
            return [right_points[1], left_points[1], left_points[0], right_points[0]]

        sorted_matrix = sort_points(matrix)
        print("정렬된 좌표 행렬:")
        print(sorted_matrix)

        center_x = sum(point[0] for point in sorted_matrix) / 4
        center_y = sum(point[1] for point in sorted_matrix) / 4
        print(f"중심 좌표:({center_x}, {center_y})")

        def calculate_angle(p1, p2):
            return np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0]))

        angle01 = calculate_angle(sorted_matrix[1], sorted_matrix[0])
        angle23 = calculate_angle(sorted_matrix[2], sorted_matrix[3])
        if angle01 > 90:
            angle01 -= 180
        elif angle01 < -90:
            angle01 += 180

        if angle23 > 90:
            angle23 -= 180
        elif angle23 < -90:
            angle23 += 180
        angle_horizontal = (angle01 + angle23) / 2
        angle12 = calculate_angle(sorted_matrix[1], sorted_matrix[2])
        angle03 = calculate_angle(sorted_matrix[0], sorted_matrix[3])
        if angle12 < 0:
            angle12 += 180

        if angle03 < 0:
            angle03 += 180
        angle_vertical = (angle12 + angle03) / 2
        angle = ((90 - angle_vertical) + angle_horizontal) / 2
        print("직사각형의 회전 각도:", angle)

        fig2, ax2 = plt.subplots(1)
        plt.gcf().canvas.manager.set_window_title("Selected Rectangle")
        ax2.imshow(img_rgb)
        rect = patches.Polygon(
            sorted_matrix, closed=True, edgecolor="r", facecolor="none", linewidth=2
        )
        ax2.add_patch(rect)
        plt.scatter(center_x, center_y, s=50, color="blue")
        plt.show()
    elapsed_time = end_time - start_time
    print(f"작업에 걸린 시간: {elapsed_time} 초")
