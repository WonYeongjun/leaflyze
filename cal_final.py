#ArUco 마커는 우상단 좌상단 좌하단 우하단 에 12, 18, 27, 5 순으로 배치(6*6크기의 마커)
import subprocess
import cv2
import numpy as np
import matplotlib.pyplot as plt

#1 사진 촬영
subprocess.run(["libcamera-jpeg", "-o", "/home/userk/cal_img/raw/raw_img.jpg", "--width", "1920", "--height", "1440"])

print("사진 촬영 완료")

#1.5 사진 회전(카메라가 뒤집혔을 때)
# 이미지를 불러오기
img0 = cv2.imread("/home/userk/cal_img/raw/raw_img.jpg")

# 180도 회전
rotated_img = cv2.rotate(img0, cv2.ROTATE_180)

# 이미지를 저장
save_path = "/home/userk/cal_img/raw/raw_img.jpg"
cv2.imwrite(save_path, rotated_img)

#2 왜곡보정
def undistort_image(image_path, camera_matrix, dist_coeffs, target_size=(640*3, 480*3)):
    # 원본 이미지 로드
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Cannot load image at {image_path}")
        return None, None

    # 이미지 크기 변경 (640x480)
    image_resized = cv2.resize(image, target_size)

    # 왜곡 보정
    h, w = image_resized.shape[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
    undistorted_image = cv2.undistort(image_resized, camera_matrix, dist_coeffs, None, new_camera_matrix)

    return image_resized, undistorted_image

# MAIN: 실행
if __name__ == "__main__":
    # 카메라 매트릭스 및 왜곡 계수
    camera_matrix = np.array(
        [
            [1.90296778e+03, 0.00000000e+00, 9.62538876e+02],
            [0.00000000e+00, 1.90259449e+03, 6.99587164e+02],
            [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
        ]
    )

    dist_coeffs = np.array([-0.00159553,  0.25363958, -0.00156397,  0.00185892, -0.41956662])

    # 테스트 이미지 경로
    image_path = "/home/userk/cal_img/raw/raw_img.jpg"  # 촬영한 이미지 경로

    # 왜곡 보정
    original, undistorted = undistort_image(image_path, camera_matrix, dist_coeffs)

    if original is not None and undistorted is not None:
        # 보정된 이미지 저장
        save_path = "/home/userk/cal_img/cal/cal_img.jpg"
        cv2.imwrite(save_path, undistorted)
        print(f"Undistorted image saved to {save_path}")


#3 원근감 보정
# 아루코 표식의 크기 (실제 크기 또는 픽셀 크기)
marker_length = 0.03  # 실제 크기 (예: 10cm)
arUco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)  # 아루코 사전 정의
parameters = cv2.aruco.DetectorParameters_create()

# 이미지를 불러오기
img = cv2.imread("/home/userk/cal_img/cal/cal_img.jpg")

# 그레이스케일로 변환
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 아루코 마커 검출
corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, arUco_dict, parameters=parameters)
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
        width = 300*4  # 변환 후 출력 이미지의 너비
        height = 355*4  # 변환 후 출력 이미지의 높이
        pts2 = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")

        # 호모그래피 행렬 계산
        matrix, mask = cv2.findHomography(pts1, pts2)

        # 원근 변환 적용
        dst = cv2.warpPerspective(img, matrix, (width, height))

        # 변환된 이미지 저장
        save_path = "/home/userk/cal_img/fin/fin_cal_img.jpg"
        cv2.imwrite(save_path, dst)
        print(f"Corrected image saved to {save_path}")

    else:
        print("Not enough markers detected to calculate perspective.")
else:
    print("No ArUco markers detected.")


