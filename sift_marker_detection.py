import cv2
import numpy as np


def detect_marker_sift(template_path, image_path):
    """SIFT를 이용해 마커를 찾고 RANSAC으로 정제된 매칭 결과를 출력"""

    # 1️⃣ 이미지 로드
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # 2️⃣ SIFT 객체 생성
    sift = cv2.SIFT_create()

    # 3️⃣ 특징점 및 디스크립터 검출
    keypoints_template, descriptors_template = sift.detectAndCompute(template, None)
    keypoints_image, descriptors_image = sift.detectAndCompute(image, None)

    # 4️⃣ BFMatcher로 특징점 매칭
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors_template, descriptors_image, k=2)

    # 5️⃣ Lowe’s Ratio Test 적용 (잘못된 매칭 제거)
    good_matches = [m for m, n in matches if m.distance < 1 * n.distance]

    if len(good_matches) < 4:
        print(f"⚠ 매칭된 특징점이 부족합니다. ({len(good_matches)}개)")
        return

    # 6️⃣ RANSAC을 이용한 Homography 계산 및 이상치 제거
    src_pts = np.float32(
        [keypoints_template[m.queryIdx].pt for m in good_matches]
    ).reshape(-1, 1, 2)
    dst_pts = np.float32(
        [keypoints_image[m.trainIdx].pt for m in good_matches]
    ).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    if M is None:
        print("⚠ Homography 행렬을 찾을 수 없습니다. 매칭된 특징점이 부족합니다.")
        return

    good_matches_filtered = [
        good_matches[i] for i in range(len(good_matches)) if mask[i]
    ]

    # 7️⃣ 마커 영역을 사각형으로 표시
    h, w = template.shape
    pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)  # 변환된 좌표 계산

    # 원본 이미지에 사각형을 그려 마커 위치 표시
    image_with_box = cv2.polylines(
        cv2.cvtColor(image, cv2.COLOR_GRAY2BGR),
        [np.int32(dst)],
        isClosed=True,
        color=(0, 255, 0),
        thickness=3,
    )

    # 8️⃣ 매칭 결과 시각화
    result = cv2.drawMatches(
        template,
        keypoints_template,
        image_with_box,
        keypoints_image,
        good_matches_filtered,
        None,
        flags=2,
    )

    # 9️⃣ 결과 출력
    cv2.imwrite("./output/matched_result.jpg", result)
    cv2.imshow("Matched Features", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 예제 실행
detect_marker_sift("./image/marker_4.png", "./image/fin_cal_img_one_marker.jpg")
