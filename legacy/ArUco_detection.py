import cv2
import numpy as np

# 아루코 표식의 크기 (실제 크기 또는 픽셀 크기)
marker_length = 0.03  # 실제 크기 (예: 10cm)
arUco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

# DetectorParameters 생성 및 최적화
detector_params = cv2.aruco.DetectorParameters()
detector_params.adaptiveThreshWinSizeMin = 3
detector_params.adaptiveThreshWinSizeMax = 23
detector_params.adaptiveThreshWinSizeStep = 10
detector_params.polygonalApproxAccuracyRate = 0.05
detector_params.minMarkerPerimeterRate = 0.005  # 더 작은 마커를 검출할 수 있도록 설정
detector_params.maxMarkerPerimeterRate = 10.0  # 더 큰 마커를 검출할 수 있도록 설정
detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
detector_params.cornerRefinementWinSize = 5
detector_params.cornerRefinementMaxIterations = 50
detector_params.cornerRefinementMinAccuracy = 0.05
detector_params.minCornerDistanceRate = 0.01  # 코너 간 최소 거리 비율을 낮춤
detector_params.maxErroneousBitsInBorderRate = (
    0.5  # 경계에서 허용되는 최대 오류 비율을 높임
)
detector_params.errorCorrectionRate = 0.7  # 오류 수정 비율을 높임

aruco_detector = cv2.aruco.ArucoDetector(arUco_dict, detector_params)

# 이미지를 불러오기
img = cv2.imread("./raw/raw_img.jpg")

# 그레이스케일로 변환
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.equalizeHist(gray)
# 이미지를 보기 편하도록 작게 만들기
scale_percent = 200  # 이미지 크기를 50%로 축소
width = int(gray.shape[1] * scale_percent / 100)
height = int(gray.shape[0] * scale_percent / 100)
dim = (width, height)

# 이미지 크기 조정
gray = cv2.resize(gray, dim, interpolation=cv2.INTER_AREA)
cv2.imshow("gray", gray)
cv2.waitKey(0)
cv2.destroyAllWindows()
# 어두운 픽셀을 검정색으로 변환
threshold_value = 150  # 임계값 설정 (0-255 사이의 값)
_, gray = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)

cv2.imshow("gray", gray)
cv2.waitKey(0)
cv2.destroyAllWindows()
# 아루코 마커 검출
corners, ids, rejectedImgPoints = aruco_detector.detectMarkers(gray)
print(corners)
print(ids)

# 원하는 마커 ID 순서
desired_ids = [12, 18, 27, 5]

if ids is not None:
    marker_centers = []  # 마커 중심 좌표를 저장할 리스트

    # 아루코 표식이 인식되었을 경우
    for i in range(len(ids)):
        marker_id = ids[i][0]
        if marker_id in desired_ids:
            # 각 마커의 중심 좌표 계산
            marker_corners = corners[i][0]  # 4개의 모서리 좌표
            center_x = np.mean(marker_corners[:, 0])  # x 좌표의 평균
            center_y = np.mean(marker_corners[:, 1])  # y 좌표의 평균
            print(f"Marker {marker_id} center: ({center_x}, {center_y})")

            # 중심 좌표를 리스트에 추가
            marker_centers.append([center_x, center_y])

    # 마커들의 순서를 desired_ids에 맞게 재정렬
    marker_centers_sorted = []
    for marker_id in desired_ids:
        for i, id in enumerate(ids):
            if id[0] == marker_id:
                marker_corners = corners[i][0]
                center_x = np.mean(marker_corners[:, 0])
                center_y = np.mean(marker_corners[:, 1])
                marker_centers_sorted.append([center_x, center_y])
                break

    # pts1을 마커의 중심 좌표들로 설정 (desired_ids 순서대로)
    if len(marker_centers_sorted) == 4:
        pts1 = np.array(marker_centers_sorted, dtype="float32")
        print(f"pts1 (marker centers): {pts1}")

        # 정면에서 본 이미지의 4개의 좌표 (임의로 설정한 예시)
        width = 500  # 변환 후 출력 이미지의 너비
        height = 500  # 변환 후 출력 이미지의 높이
        pts2 = np.array(
            [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
            dtype="float32",
        )

        # 호모그래피 행렬 계산
        matrix, mask = cv2.findHomography(pts1, pts2)

        # 원근 변환 적용
        dst = cv2.warpPerspective(img, matrix, (width, height))

        # 변환된 이미지 저장
        save_path = "./fin/fin_cal_img.jpg"
        cv2.imwrite(save_path, dst)
        print(f"Corrected image saved to {save_path}")

    else:
        print("Not enough markers detected to calculate perspective.")
else:
    print("No ArUco markers detected.")
