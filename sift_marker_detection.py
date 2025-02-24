import numpy as np
import cv2


def detect_and_show_keypoints(image):
    # SIFT 특징점 탐지기 생성
    sift = cv2.SIFT_create()

    # 특징점 탐지
    keypoints, descriptors = sift.detectAndCompute(image, None)

    # 특징점 그리기
    image_with_keypoints = cv2.drawKeypoints(
        image,
        keypoints,
        None,
        color=(0, 255, 0),
        flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS,
    )

    return image_with_keypoints


file_name = "black_back"
image_path = f"C:/Users/UserK/Desktop/fin/{file_name}.jpg"

template_path = "rectangle_with_border.png"
# 예시 이미지 생성 (너비 400, 높이 300)
width = 400
height = 300
image = cv2.imread(template_path)
# image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

# 특징점 탐지 및 표시
image_with_keypoints = detect_and_show_keypoints(image)

# 결과 이미지 표시
cv2.imwrite("image_with_keypoints.png", image_with_keypoints)
cv2.imshow("Image with Keypoints", image_with_keypoints)
cv2.waitKey(0)
cv2.destroyAllWindows()
