import time

import cv2
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from simplification import morphology_diff
from shape_detect import detect_lines

start_time = time.time()


def sum_pixels_at_coordinates(image, coordinates):

    total = sum(image[int(co[1]), int(co[0])] for co in coordinates)
    return total


def get_white_pixel_coordinates(image):

    coords = np.argwhere(image == 255)  # 값이 255인 픽셀 좌표 찾기
    return [(y, x) for y, x in coords]  # (row, col) → (x, y) 변환


def evaluate_configuration(image, initial_center, size, angle, dx, dy):
    center = (initial_center[0] + dx, initial_center[1] + dy)
    rotated_rect = ((center), size, angle)

    # Get the box points (corners of the rotated rectangle)
    box = cv2.boxPoints(rotated_rect)
    box = np.int32(box)

    # Precompute the mask shape for this configuration
    mask = np.zeros_like(image)
    cv2.polylines(mask, [box], isClosed=True, color=255, thickness=11)

    # Perform bitwise AND to get the white pixels inside the rotated rectangle
    masked_image = cv2.bitwise_and(image, image, mask=mask)

    # Count the number of white pixels using cv2.countNonZero (faster than np.argwhere)
    total = cv2.countNonZero(masked_image)
    return total, center, angle, box


def process_configuration(config, image, initial_center, size):
    return evaluate_configuration(image, initial_center, size, *config)


def find_optimal_rectangle(
    image, initial_center, size, angle_range=(3, 13), center_range=(-50, 50), step=5
):
    best_total = float("-inf")
    best_center = None
    best_angle = None
    best_box = None

    # Use ProcessPoolExecutor for parallel computation
    with ProcessPoolExecutor() as executor:
        # Prepare the configurations to evaluate
        configurations = [
            (angle, dx, dy)
            for angle in range(angle_range[0], angle_range[1], step)
            for dx in range(center_range[0], center_range[1], step)
            for dy in range(center_range[0], center_range[1], step)
        ]

        # Evaluate configurations in parallel using multiprocessing
        results = executor.map(
            process_configuration,
            configurations,
            [image] * len(configurations),
            [initial_center] * len(configurations),
            [size] * len(configurations),
        )

        # Update the best configuration based on the results
        for total, center, angle, box in results:
            print(f"Angle: {angle}, Center: {center} Total: {total}")
            if total > best_total:
                best_total = total
                best_center = center
                best_angle = angle
                best_box = box

    return best_center, best_angle, best_box, best_total


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

    initial_center = (2119, 1510)
    size = (1686, 1378)

    best_center, best_angle, best_box, best_total = find_optimal_rectangle(
        shape_image, initial_center, size
    )

    # Draw the rotated rectangle
    cv2.polylines(shape_image, [best_box], isClosed=True, color=255, thickness=11)

    print(f"Total white pixels in the rotated rectangle: {best_total}")
    end_time = time.time()
    plt.imshow(shape_image, cmap="gray")
    plt.show()

    elapsed_time = end_time - start_time
    print(f"작업에 걸린 시간: {elapsed_time} 초")
