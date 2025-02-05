# 저장된 이미지를 가져와서 원근법 보정함
import cv2
import numpy as np
import matplotlib.pyplot as plt


# 원근 보정 함수
def correct_perspective(image_path):
    # 이미지 로드
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Cannot load image at {image_path}")
        return

    # 이미지 크기 및 중심 계산
    h, w, _ = image.shape
    image_center = np.array([w / 2, h / 2])

    # 그레이스케일 변환
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 아루코 마커 검출
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    corners, ids, _ = detector.detectMarkers(gray)

    # 원하는 마커 ID 순서
    desired_ids = [12, 18, 27, 5]

    if ids is not None:
        marker_points = []

        # 검출된 마커에서 원하는 ID 찾기
        for marker_id in desired_ids:
            for i, detected_id in enumerate(ids):
                if detected_id[0] == marker_id:
                    marker_corners = corners[i][0]  # 4개의 꼭짓점 좌표
                    # 이미지 중심과 가장 가까운 꼭짓점 선택
                    closest_corner = min(
                        marker_corners, key=lambda pt: np.linalg.norm(pt - image_center)
                    )
                    marker_points.append(closest_corner)
                    break

        # 모든 마커가 감지된 경우 원근 변환 수행
        if len(marker_points) == 4:
            pts1 = np.array(marker_points, dtype="float32")
            print(f"pts1 (selected marker corners): {pts1}")

            # 변환 후 기준이 될 좌표
            width, height = 4200, 2970
            pts2 = np.array(
                [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
                dtype="float32",
            )

            # 호모그래피 행렬 계산
            matrix, _ = cv2.findHomography(pts1, pts2)

            # 원근 변환 적용
            dst = cv2.warpPerspective(image, matrix, (width, height))
        else:
            print("Not enough markers detected to calculate perspective.")
    else:
        print("No ArUco markers detected.")

    return dst


# 실행
test_image_path = "./image/cal+marker.jpg"  # 원본 이미지 경로
image = correct_perspective(test_image_path)
# plt.imshow(image)
# plt.show()
