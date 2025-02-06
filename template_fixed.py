import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl
import random
from InvariantTM1 import invariant_match_template  # ,template_crop
import time

# 시작 시간 기록
start_time = time.time()

if __name__ == "__main__":
    threshold = 130
    img_bgr = cv2.imread("./image/glass/pink_15.jpg")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    template_bgr = plt.imread("./image/marker_ideal.jpg")
    template_bgr = cv2.resize(
        template_bgr, (0, 0), fx=0.27, fy=0.27
    )  # 템플릿 사이즈 조절(초기 설정 필요)

    template_rgb = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2RGB)

    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    # alpha = 1.2  # 조정 강도 (1보다 크면 밝기 증가, 1보다 작으면 감소)
    # img_gray = cv2.convertScaleAbs(img_gray, alpha=alpha, beta=0)

    # def gamma_correction(img, gamma=0.5):
    #     lookup_table = np.array(
    #         [(i / 255.0) ** gamma * 255 for i in range(256)]
    #     ).astype(np.uint8)
    #     return cv2.LUT(img, lookup_table)

    # clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(8, 8))
    # # img_gray = clahe.apply(img_gray)
    # img_gray = gamma_correction(img_gray, gamma=0.5)

    # _, img_gray = cv2.threshold(
    #     img_gray, 80, 255, cv2.THRESH_BINARY
    # 실제 이미지 이진화
    im = img_gray
    img_gray = np.where(
        img_gray > threshold,
        np.random.randint(245, 256, img_gray.shape, dtype=np.uint8),
        np.random.randint(0, 11, img_gray.shape, dtype=np.uint8),
    )
    img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)

    template_gray = cv2.cvtColor(template_rgb, cv2.COLOR_RGB2GRAY)
    _, template_gray = cv2.threshold(template_gray, threshold, 255, cv2.THRESH_BINARY)
    template_rgb = cv2.cvtColor(template_gray, cv2.COLOR_GRAY2RGB)

    # img_rgb = cv2.cvtColor(cv2.Canny(img_rgb, 240, 240), cv2.COLOR_GRAY2RGB)
    # template_rgb = cv2.cvtColor(cv2.Canny(template_rgb, 240, 240), cv2.COLOR_GRAY2RGB)
    # canny = cv2.Canny(template_rgb, 240, 240)

    # cropped_template_rgb = template_crop(template_rgb)
    cropped_template_rgb = template_rgb
    cropped_template_rgb = np.array(cropped_template_rgb)
    cropped_template_gray = cv2.cvtColor(cropped_template_rgb, cv2.COLOR_RGB2GRAY)
    height, width = cropped_template_gray.shape
    # fig = plt.figure(num="Template - Close the Window to Continue >>>")
    # plt.imshow(cropped_template_rgb)
    # plt.show()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.imshow(im, cmap="gray")
    ax1.set_title("Original Grayscale Image")
    ax2.imshow(img_gray, cmap="gray")
    ax2.set_title("Processed Grayscale Image")
    plt.show()
    points_list = invariant_match_template(
        rgbimage=img_rgb,
        rgbtemplate=cropped_template_rgb,
        method="TM_SQDIFF",
        matched_thresh=0.5,
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
    # points_list = [point for point in points_list if point[4] != float("inf")]
    # points_list = points_list[:20]
    # reference_angle = points_list[0][1]

    # # 각도 차이를 기준으로 정렬
    # points_list = sorted(points_list, key=lambda x: abs(x[1] - reference_angle))
    # points_list = points_list[:20]
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
            point[0] + (width / 2) * scale / 100,
            point[1] + (height / 2) * scale / 100,
            s=20,
            color="red",
        )
        idx = (
            point[0] + (width / 2) * scale / 100,
            point[1] + (height / 2) * scale / 100,
        )
        real_point.append([idx])
        plt.scatter(point[0], point[1], s=20, color="green")
        rectangle = patches.Rectangle(
            (point[0], point[1]),
            width * scale / 100,
            height * scale / 100,
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
                point[0] + width / 2 * scale / 100,
                point[1] + height / 2 * scale / 100,
                angle,
            )
            + ax.transData
        )
        rectangle.set_transform(transform)
        ax.add_patch(rectangle)
        plt.legend(handles=[rectangle])
    # plt.grid(True)
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
            points = sorted(points, key=lambda x: x[0])  # x 좌표 기준으로 정렬
            left_points = points[:2]
            right_points = points[2:]
            left_points = sorted(
                left_points, key=lambda x: x[1]
            )  # y 좌표 기준으로 정렬
            right_points = sorted(right_points, key=lambda x: x[1])
            return [right_points[1], left_points[1], left_points[0], right_points[0]]

        sorted_matrix = sort_points(matrix)
        print("정렬된 좌표 행렬:")
        print(sorted_matrix)

        # 중심 좌표 계산
        center_x = sum(point[0] for point in sorted_matrix) / 4
        center_y = sum(point[1] for point in sorted_matrix) / 4
        print(f"중심 좌표:({center_x}, {center_y})")

        # 회전 각도 계산
        def calculate_angle(p1, p2):
            return np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0]))

        # 오른쪽 위와 왼쪽 위 점을 이용하여 각도 계산
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

        # 사각형 그리기
        fig2, ax2 = plt.subplots(1)
        plt.gcf().canvas.manager.set_window_title("Selected Rectangle")
        ax2.imshow(img_rgb)
        rect = patches.Polygon(
            sorted_matrix, closed=True, edgecolor="r", facecolor="none", linewidth=2
        )
        ax2.add_patch(rect)
        plt.scatter(center_x, center_y, s=50, color="blue")
        plt.show()
    # fig2, ax2 = plt.subplots(1)
    # plt.gcf().canvas.manager.set_window_title("Template Matching Results: Centers")
    # ax2.imshow(img_rgb)
    # for point_info in centers_list:
    #     point = point_info[0]
    #     scale = point_info[1]
    #     plt.scatter(
    #         point[0] + width / 2 * scale / 100,
    #         point[1] + height / 2 * scale / 100,
    #         s=20,
    #         color="red",
    #     )
    # plt.show()
    # 걸린 시간 계산
    elapsed_time = end_time - start_time
    print(f"작업에 걸린 시간: {elapsed_time} 초")
