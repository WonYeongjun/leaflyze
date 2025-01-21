import cv2
import numpy as np

# 아루코 표식의 크기 (실제 크기 또는 픽셀 크기)
marker_length = 0.03  # 실제 크기 (예: 10cm)
arUco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)  # 아루코 사전 정의
parameters = cv2.aruco.DetectorParameters_create()

# 이미지를 불러오기
img = cv2.imread("/home/userk/per_img/per_raw.jpg")

# 그레이스케일로 변환
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 아루코 마커 검출
corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, arUco_dict, parameters=parameters)
print(corners)
print(ids)

if ids is not None:
    marker_centers = []  # 마커 중심 좌표를 저장할 리스트

    # 아루코 표식이 인식되었을 경우
    for i in range(len(ids)):
        # 각 마커의 중심 좌표 계산
        marker_corners = corners[i][0]  # 4개의 모서리 좌표
        center_x = np.mean(marker_corners[:, 0])  # x 좌표의 평균
        center_y = np.mean(marker_corners[:, 1])  # y 좌표의 평균
        print(f"Marker {ids[i]} center: ({center_x}, {center_y})")

        # 중심 좌표를 리스트에 추가
        marker_centers.append([center_x, center_y])

        # 마커를 이미지에 그리기
        cv2.aruco.drawDetectedMarkers(img, corners, ids)

    # pts1을 마커의 중심 좌표들로 설정 (마커 4개가 있을 경우)
    if len(marker_centers) == 4:
        pts1 = np.array(marker_centers, dtype="float32")
        print(f"pts1 (marker centers): {pts1}")

        # 정면에서 본 이미지의 4개의 좌표 (임의로 설정한 예시)
        width = 500  # 변환 후 출력 이미지의 너비
        height = 500  # 변환 후 출력 이미지의 높이
        pts2 = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")

        # 호모그래피 행렬 계산
        matrix, mask = cv2.findHomography(pts1, pts2)

        # 원근 변환 적용
        dst = cv2.warpPerspective(img, matrix, (width, height))

        # 결과 이미지 출력
        cv2.imshow("Corrected Image", dst)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Not enough markers detected to calculate perspective.")
else:
    print("No ArUco markers detected.")
