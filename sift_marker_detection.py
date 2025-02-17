import cv2
import numpy as np
import matplotlib.pyplot as plt
from image_segmentation import point_of_interest


def extract_sift_features(img):
    """SIFT 특징점과 기술자 추출 함수"""
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif img.dtype != np.uint8:
        img = img.astype(np.uint8)

    # SIFT 객체 생성 (파라미터 튜닝)
    sift = cv2.SIFT_create(
        nOctaveLayers=7, contrastThreshold=0.1, edgeThreshold=20, sigma=1.9
    )

    # 이미지 전처리 (가우시안 블러로 노이즈 제거)
    # img_blurred = cv2.GaussianBlur(img, (5, 5), 1.5)

    # 특징점과 디스크립터 계산
    keypoints, descriptors = sift.detectAndCompute(img, None)

    # 특징점 그리기
    img_with_keypoints = cv2.drawKeypoints(
        img, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    return img_with_keypoints, keypoints, descriptors


def match_sift_features(descriptors_template, descriptors_image):
    """SIFT 특징점 매칭 (FLANN과 Lowe's ratio test 적용)"""
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # FLANN 매칭 수행
    matches = flann.knnMatch(descriptors_template, descriptors_image, k=4)

    # Lowe's ratio test: 비율 검사를 통해 잘못된 매칭 제거
    print(len(matches))
    good_matches = []
    for m, n, o, p in matches:
        if m.distance < 250:
            good_matches.extend([m, n, o, p])
    print(f"정답{len(good_matches)}")

    return good_matches


def apply_ransac_filter(keypoints_template, keypoints_image, good_matches):
    """RANSAC 후처리를 통해 잘못된 매칭을 제거"""
    src_pts = np.float32(
        [keypoints_template[m.queryIdx].pt for m in good_matches]
    ).reshape(-1, 1, 2)
    dst_pts = np.float32(
        [keypoints_image[m.trainIdx].pt for m in good_matches]
    ).reshape(-1, 1, 2)

    # Homography 찾기 및 RANSAC 적용
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 15.0)

    # 잘못된 매칭을 제외한 좋은 매칭만 반환
    good_matches_filtered = [
        good_matches[i] for i in range(len(good_matches)) if mask[i]
    ]

    return good_matches_filtered, M


def find_multiple_markers(
    image, template_binary, descriptors_template, keypoints_template
):
    """여러 마커를 찾는 함수"""
    result_img, keypoints, descriptors = extract_sift_features(image)

    # 특징점 매칭 수행
    good_matches = match_sift_features(descriptors_template, descriptors)

    # RANSAC 후처리
    # good_matches_filtered, M = apply_ransac_filter(
    #     keypoints_template, keypoints, good_matches
    # )

    # 매칭된 특징점들 시각화
    matched_image = cv2.drawMatches(
        template_binary,
        keypoints_template,
        image,
        keypoints,
        good_matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    # 결과 이미지를 화면에 출력
    plt.figure(figsize=(10, 5))
    plt.imshow(matched_image)
    plt.title("SIFT Multi-Template Matching")
    plt.show()

    # 여기에 추가로 다중 마커를 식별하는 방법 (예: 컨투어, Hough 변환 등) 추가
    # 예: 특정 패턴을 인식하거나, 마커들을 연결하여 마커 위치 파악


if __name__ == "__main__":
    # 이미지 파일 경로
    image_path = "./image/pink/fin_cal_img_20250207_141129.jpg"  # 예: "sample.jpg"
    template_path = "./image/marker_4.png"

    # 이미지 및 템플릿 로드
    image = cv2.imread(image_path)
    image = point_of_interest(image)
    template = cv2.imread(template_path)

    # 템플릿을 그레이스케일로 변환
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    _, template_binary = cv2.threshold(template_gray, 128, 255, cv2.THRESH_BINARY)

    # 템플릿에서 SIFT 특징점 및 기술자 추출
    result_img_template, keypoints_template, descriptors_template = (
        extract_sift_features(template_binary)
    )

    # 타깃 이미지에서 다중 마커 찾기
    find_multiple_markers(
        image, template_binary, descriptors_template, keypoints_template
    )
