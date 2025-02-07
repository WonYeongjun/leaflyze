import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl
import random
import time
import itertools

# 시작 시간 기록
start_time = time.time()

threshold = 130
img_bgr = cv2.imread("./exm/glass/NewPink.jpg")
template_bgr = plt.imread("./image/marker_ideal.jpg")
template_bgr = cv2.resize(
    template_bgr, (0, 0), fx=0.27, fy=0.27
)  # 템플릿 사이즈 조절(초기 설정 필요)

img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_RGB2GRAY)
im = img_gray
img_gray = np.where(
    img_gray > threshold,
    np.random.randint(245, 256, img_gray.shape, dtype=np.uint8),
    np.random.randint(0, 11, img_gray.shape, dtype=np.uint8),
)
img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)

template_gray = cv2.cvtColor(template_bgr, cv2.COLOR_RGB2GRAY)
_, template_gray = cv2.threshold(template_gray, threshold, 255, cv2.THRESH_BINARY)
height, width = template_gray.shape


def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, -angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


def scale_image(image, percent):
    width = int(image.shape[1] * percent / 100)
    height = int(image.shape[0] * percent / 100)
    result = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    return result, percent


def process_template(next_angle, next_scale):
    scaled_template_gray, actual_scale = scale_image(template_gray, next_scale)
    rotated_template = (
        scaled_template_gray
        if next_angle == 0
        else rotate_image(scaled_template_gray, next_angle)
    )
    result = cv2.matchTemplate(img_gray, rotated_template, cv2.TM_CCOEFF_NORMED)
    locations = np.where(result <= 0.4)
    matches = []
    x_coords = locations[1]
    y_coords = locations[0]
    scores = result[locations]
    angles = np.full_like(scores, next_angle)
    scales = np.full_like(scores, actual_scale)

    matches = np.column_stack((x_coords, y_coords, angles, scales, scores)).tolist()
    # 모든 (x, y) 좌표와 점수 저장
    print(next_angle, next_scale)
    return matches


def invariant_match_template(
    grayimage,
    graytemplate,
    method,
    matched_thresh,
    rot_range,
    rot_interval,
    scale_range,
    scale_interval,
    rm_redundant,
    minmax,
    rgbdiff_thresh=float("inf"),
):
    img_gray = grayimage
    template_gray = graytemplate
    image_maxwh = img_gray.shape
    height, width = template_gray.shape
    all_points = []

    print("동그라미")
    from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

    print(template_gray.shape)

    with ProcessPoolExecutor() as executor:
        tasks = [
            executor.submit(process_template, next_angle, next_scale)
            for next_angle in range(rot_range[0], rot_range[1], rot_interval)
            for next_scale in range(scale_range[0], scale_range[1], scale_interval)
        ]
        all_points = list(
            itertools.chain.from_iterable(
                result for future in tasks if (result := future.result())
            )
        )
        print(len(all_points))
    if method == "TM_CCOEFF":
        all_points = sorted(all_points, key=lambda x: -x[3])
    elif method == "TM_CCOEFF_NORMED":
        all_points = sorted(all_points, key=lambda x: -x[3])
    elif method == "TM_CCORR":
        all_points = sorted(all_points, key=lambda x: -x[3])
    elif method == "TM_CCORR_NORMED":
        all_points = sorted(all_points, key=lambda x: -x[3])
    elif method == "TM_SQDIFF":
        all_points = sorted(all_points, key=lambda x: x[3])
    elif method == "TM_SQDIFF_NORMED":
        all_points = sorted(all_points, key=lambda x: x[3])

    def filter_redundant_points(points_chunk):
        # """중복된 포인트를 제거하는 함수 (각 스레드에서 독립적으로 실행)"""
        lone_points_list = []
        visited_points_list = []

        for point_info in points_chunk:
            point = point_info[0]
            scale = point_info[2]
            all_visited_points_not_close = True

            for visited_point in visited_points_list:
                if (abs(visited_point[0] - point[0]) < (width * scale / 100)) and (
                    abs(visited_point[1] - point[1]) < (height * scale / 100)
                ):
                    all_visited_points_not_close = False
                    break  # 이미 가까운 점이 있으면 더 체크할 필요 없음

            if all_visited_points_not_close:
                lone_points_list.append(point_info)
                visited_points_list.append(point)

        return lone_points_list

    if rm_redundant:
        chunk_size = max(1, len(all_points) // 4)  # 스레드당 할당할 포인트 개수
        chunks = [
            all_points[i : i + chunk_size]
            for i in range(0, len(all_points), chunk_size)
        ]

        with ThreadPoolExecutor() as executor2:
            results = executor2.map(filter_redundant_points, chunks)

        # 여러 스레드 결과를 합치면서 최종 중복 제거
        final_lone_points = []
        final_visited_points = []

        for lone_points in results:
            for point_info in lone_points:
                point = point_info[0]
                scale = point_info[2]
                all_visited_points_not_close = True

                for visited_point in final_visited_points:
                    if (abs(visited_point[0] - point[0]) < (width * scale / 100)) and (
                        abs(visited_point[1] - point[1]) < (height * scale / 100)
                    ):
                        all_visited_points_not_close = False
                        break

                if all_visited_points_not_close:
                    final_lone_points.append(point_info)
                    final_visited_points.append(point)

        points_list = final_lone_points
    else:
        points_list = all_points
    return points_list


if __name__ == "__main__":

    # 전처리된 이미지 시각화
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.imshow(im, cmap="gray")
    ax1.set_title("Original Grayscale Image")
    ax2.imshow(img_gray, cmap="gray")
    ax2.set_title("Processed Grayscale Image")
    plt.show()
    points_list = invariant_match_template(
        grayimage=img_gray,
        graytemplate=template_gray,
        method="TM_CCOEFF_NORMED",
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
