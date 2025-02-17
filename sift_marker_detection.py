import cv2


def extract_sift_features(image_path):
    # 이미지 읽기
    img = cv2.imread(image_path)
    if img is None:
        print("이미지를 불러올 수 없습니다.")
        return None, None, None

    # SIFT 객체 생성 (OpenCV 4.4 이후부터 cv2.SIFT_create 사용)
    sift = cv2.SIFT_create()

    # 이미지에서 키포인트와 기술자 계산 (키포인트: 특징점 위치, 기술자: 특징점 주변의 특징 벡터)
    keypoints, descriptors = sift.detectAndCompute(img, None)
    print(descriptors[0])

    # 키포인트를 이미지에 그리기
    # DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS 플래그를 사용하면, 각 키포인트의 크기와 방향이 함께 표시됩니다.
    img_with_keypoints = cv2.drawKeypoints(
        img, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    return img_with_keypoints, keypoints, descriptors


if __name__ == "__main__":
    # 이미지 파일 경로를 입력하세요.
    image_path = "./image/pink/fin_cal_img_20250207_141129.jpg"  # 예: "sample.jpg"

    result_img, keypoints, descriptors = extract_sift_features(image_path)

    if result_img is not None:
        # 결과 이미지 창 띄우기
        import matplotlib.pyplot as plt

        # 이미지를 RGB 형식으로 변환 (OpenCV는 BGR 형식을 사용)
        result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

        # 이미지 시각화
        plt.imshow(result_img_rgb)
        plt.title("SIFT Keypoints")
        plt.axis("off")
        plt.show()
