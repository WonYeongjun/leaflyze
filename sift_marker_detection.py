import cv2
import matplotlib.pyplot as plt
import numpy as np
from image_segmentation import point_of_interest


def extract_sift_features(img):
    # 이미지 읽기

    # SIFT 객체 생성 (OpenCV 4.4 이후부터 cv2.SIFT_create 사용)
    sift = cv2.SIFT_create()
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif img.dtype != np.uint8:
        img = img.astype(np.uint8)

    kernel_x = np.array([[1, 0], [0, -1]], dtype=np.float32)
    kernel_y = np.array([[0, 1], [-1, 0]], dtype=np.float32)

    # 필터 적용
    roberts_x = cv2.filter2D(img, cv2.CV_32F, kernel_x)
    roberts_y = cv2.filter2D(img, cv2.CV_32F, kernel_y)

    # 결과 합성
    roberts_edge = cv2.magnitude(roberts_x, roberts_y)
    roberts_edge = cv2.convertScaleAbs(roberts_edge)

    # 이미지에서 키포인트와 기술자 계산 (키포인트: 특징점 위치, 기술자: 특징점 주변의 특징 벡터)
    keypoints, descriptors = sift.detectAndCompute(roberts_edge, None)
    # 키포인트를 이미지에 그리기
    # DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS 플래그를 사용하면, 각 키포인트의 크기와 방향이 함께 표시됩니다.
    img_with_keypoints = cv2.drawKeypoints(
        roberts_edge, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    return img_with_keypoints, keypoints, descriptors


if __name__ == "__main__":
    # 이미지 파일 경로를 입력하세요.
    image_path = "./image/pink/fin_cal_img_20250207_141129.jpg"  # 예: "sample.jpg"
    template_path = "./image/marker_4.png"
    image = cv2.imread(image_path)
    image = point_of_interest(image)
    template = cv2.imread(template_path)

    # Convert the template image to grayscale
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    # # Apply binary thresholding to the grayscale template image
    _, template_binary = cv2.threshold(template_gray, 128, 255, cv2.THRESH_BINARY)
    result_img, keypoints, descriptors = extract_sift_features(image)
    result_img_template, keypoints_template, descriptors_template = (
        extract_sift_features(template_binary)
    )
    bf = cv2.BFMatcher(cv2.NORM_L2)

    # KNN 매칭 수행 (각 descriptor에 대해 가장 가까운 2개의 매칭을 찾음)
    matches = bf.knnMatch(descriptors_template, descriptors, k=4)
    good_matches = []
    for m, n, o, p in matches:
        if m.distance <= 400:
            good_matches.append(m)
            print(m.distance)
        if n.distance <= 400:
            good_matches.append(n)
            print(n.distance)
        if o.distance <= 400:
            good_matches.append(o)
            print(o.distance)
    matched_image = cv2.drawMatches(
        template_binary,
        keypoints_template,
        image,
        keypoints,
        good_matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    plt.figure(figsize=(10, 5))
    plt.imshow(matched_image)
    plt.title("SIFT Template Matching")
    plt.show()

    if result_img is not None:
        # 결과 이미지 창 띄우기

        # 이미지를 RGB 형식으로 변환 (OpenCV는 BGR 형식을 사용)
        result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

        # 이미지 시각화
        result_img_template_rgb = cv2.cvtColor(result_img_template, cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.imshow(result_img_rgb)
        plt.title("SIFT Keypoints - Image")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(result_img_template_rgb)
        plt.title("SIFT Keypoints - Template")
        plt.axis("off")

        plt.show()
