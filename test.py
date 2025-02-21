import cv2
import numpy as np
import matplotlib.pyplot as plt

from get_point_of_interest import get_point_of_interest
from simplification import morphology_diff
from shape_detect import detect_lines


def sum_pixels_at_coordinates(image, coordinates):

    total = sum(image[int(co[1]), int(co[0])] for co in coordinates)
    return total


def get_white_pixel_coordinates(image):

    coords = np.argwhere(image == 255)  # 값이 255인 픽셀 좌표 찾기
    return [(y, x) for y, x in coords]  # (row, col) → (x, y) 변환


def get_rectangle_border_pixels(top_left, bottom_right, thickness=1):
    x1, y1 = top_left
    x2, y2 = bottom_right

    # 가로 변 (윗변과 아랫변)
    x_range = np.arange(x1, x2 + 1)
    top_edges = np.column_stack(
        [
            np.tile(x_range, thickness),
            np.repeat(np.arange(y1, y1 + thickness), len(x_range)),
        ]
    )
    bottom_edges = np.column_stack(
        [
            np.tile(x_range, thickness),
            np.repeat(np.arange(y2 - thickness + 1, y2 + 1), len(x_range)),
        ]
    )

    # 세로 변 (왼쪽과 오른쪽 변)
    y_range = np.arange(y1, y2 + 1)
    left_edges = np.column_stack(
        [
            np.repeat(np.arange(x1, x1 + thickness), len(y_range)),
            np.tile(y_range, thickness),
        ]
    )
    right_edges = np.column_stack(
        [
            np.repeat(np.arange(x2 - thickness + 1, x2 + 1), len(y_range)),
            np.tile(y_range, thickness),
        ]
    )

    # 네 변을 합치고 중복 제거
    border_pixels = np.vstack([top_edges, bottom_edges, left_edges, right_edges])
    border_pixels = np.unique(border_pixels, axis=0)

    return np.array([tuple(coord) for coord in border_pixels])


def rotate_points(points, angle, cx, cy):
    # 각도를 라디안으로 변환
    angle_rad = np.deg2rad(angle)

    # 회전 행렬
    rotation_matrix = np.array(
        [
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad), np.cos(angle_rad)],
        ]
    )

    # 모든 점을 기준으로 회전
    rotated_points = []
    for px, py in points:
        # 점 (px, py)을 (cx, cy)를 기준으로 이동 후 회전하고 다시 되돌리는 계산
        translated_point = np.array([px - cx, py - cy])
        rotated_point = np.dot(rotation_matrix, translated_point)

        # 다시 원래의 기준점 (cx, cy)으로 이동
        rotated_point = rotated_point + np.array([cx, cy])

        rotated_points.append(rotated_point)

    return np.array(rotated_points)


def translate_points(points, dx, dy):
    # 모든 점에 대해 dx, dy를 더하여 평행이동
    translated_points = points + np.array([dx, dy])
    return translated_points


# 예제 실행
if __name__ == "__main__":
    image_path = "C:/Users/UserK/Desktop/fin/purple_back.jpg"
    # 샘플 흑백 이미지 생성 (랜덤한 0, 255 값)
    img_bgr = cv2.imread(image_path)
    print(img_bgr.shape)
    # img_bgr = get_point_of_interest(img_bgr)
    img_gray, _, _ = morphology_diff(img_bgr)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    shape_image = detect_lines(img_gray)

    # 원하는 직사각형 크기
    rect_w, rect_h = 1686, 1378
    rect_pixels_list = get_rectangle_border_pixels(
        (5, 5), (rect_w + 5, rect_h + 5), thickness=11
    )
    rotated_rect_pixels_list = rotate_points(
        rect_pixels_list, 3, 5 + rect_w / 2, 5 + rect_h / 2
    )
    translated_rect_pixels_list = translate_points(rotated_rect_pixels_list, 1283, 788)

    total = sum_pixels_at_coordinates(shape_image, translated_rect_pixels_list)

    print(f"Total white pixels in the rotated rectangle: {total}")
    plt.imshow(
        cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    )  # BGR 이미지를 RGB로 변환하여 표시
    plt.scatter(
        translated_rect_pixels_list[:, 0],
        translated_rect_pixels_list[:, 1],
        color="red",
        label="Rotated Points",
    )
    plt.legend()
    plt.show()
