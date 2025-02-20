import numpy as np
import cv2


def make_rect(width, height):
    img = np.zeros((int(height * 1.2), int(width * 1.2)), dtype=np.uint8)
    mask = np.zeros((int(height * 1.2), int(width * 1.2)), dtype=np.uint8)
    # 사각형의 테두리 색깔 (파란색으로 설정)
    rect_color = (255, 0, 0)

    # 사각형 그리기 (위치: (50, 50)에서 시작, 너비: width, 높이: height)
    top_left = (int(width * 0.1), int(height * 0.1))  # 사각형의 좌상단 좌표
    bottom_right = (int(width * 1.1), int(height * 1.1))  # 사각형의 우하단 좌표
    cv2.rectangle(img, top_left, bottom_right, rect_color, 3)  # 두께 3px
    cv2.rectangle(mask, top_left, bottom_right, rect_color, 10)  # 내부 채우기
    return img, mask


if __name__ == "__main__":
    # 사각형의 너비와 높이를 사용자로부터 입력받음
    width = 400  # 예시 너비
    height = 200  # 예시 높이

    output_image = make_rect(width, height)
    cv2.imwrite("rectangle_with_border.png", output_image)
    cv2.imshow("Generated Rectangle with Border", output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
