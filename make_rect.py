import numpy as np
import cv2


def make_rect(size, angle):
    mask = np.zeros((int(size[1] * 1.2), int(size[0] * 1.2)), dtype=np.uint8)
    center = (size[0] * 1.2 // 2, size[1] * 1.2 // 2)
    rotated_rect = ((center), size, angle)

    # Get the box points (corners of the rotated rectangle)
    box = cv2.boxPoints(rotated_rect)
    box = np.int32(box)

    # Precompute the mask shape for this configuration
    cv2.polylines(mask, [box], isClosed=True, color=255, thickness=31)

    return mask


if __name__ == "__main__":
    # 사각형의 너비와 높이를 사용자로부터 입력받음
    width = 500  # 예시 너비
    height = 200  # 예시 높이
    angle = 10

    output_image = make_rect((width, height), angle)
    cv2.imwrite("rectangle_with_border.png", output_image)
    cv2.imshow("Generated Rectangle with Border", output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
