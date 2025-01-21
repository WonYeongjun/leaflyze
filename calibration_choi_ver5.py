import cv2
import numpy as np
import time
from picamera2 import Picamera2, Preview
import os

# -----------------------------
# STEP 1: 체스보드 사진 촬영 및 체스보드 찾기 반복
# -----------------------------
def capture_and_find_chessboard(calibration_dir, num_images=20, checkerboard_size=(7, 7)):
    # 저장 디렉토리 확인 및 생성
    if not os.path.exists(calibration_dir):
        os.makedirs(calibration_dir)
        print(f"Created directory: {calibration_dir}")

    # Picamera2 설정
    camera = Picamera2()

    # 해상도 설정 (4608x2592) 
    camera.configure(camera.create_still_configuration(raw={'size': (1920, 1440)}))

    # 미리보기 해상도를 캡처 해상도와 동일하게 설정
    preview_config = camera.create_preview_configuration(main={"size": (1920, 1440)})
    camera.configure(preview_config)

    # 화면에 미리보기 활성화
    camera.start_preview(Preview.QTGL)
    camera.start()
    time.sleep(1)  # 카메라 초기화 대기

    objpoints = []  # 3D 점
    imgpoints = []  # 2D 점

    # 3D 체스보드 점 생성
    objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)

    print(f"Starting chessboard capture and detection... ({num_images} images to be captured)")

    while len(imgpoints) < num_images:
        # 사진 촬영
        filename = os.path.join(calibration_dir, f"calibration_{len(imgpoints)}.jpg")
        camera.capture_file(filename)
        print(f"{filename} saved.")

        # 촬영된 이미지 로드 및 체스보드 코너 찾기
        img = cv2.imread(filename)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 체스보드 코너 검출
        ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)

        if ret:
            # 체스보드를 찾은 경우
            print(f"Chessboard found in image: {filename}")
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            imgpoints.append(corners2)
            objpoints.append(objp)

            # 검출된 코너를 이미지에 표시
            cv2.drawChessboardCorners(img, checkerboard_size, corners2, ret)
            cv2.imshow("Calibration", img)
            cv2.waitKey(100)
        else:
            # 체스보드를 찾지 못한 경우
            print(f"Chessboard not found in image: {filename}")

        # 2초 대기 후 다시 시도
        time.sleep(2)

    # 캘리브레이션 이미지가 충분히 모였으면 종료
    camera.stop_preview()
    camera.close()
    print(f"Captured {len(imgpoints)} valid chessboard images.")

    cv2.destroyAllWindows()

    return objpoints, imgpoints, gray.shape[::-1]

# -----------------------------
# STEP 2: 카메라 캘리브레이션
# -----------------------------
def calibrate_camera(objpoints, imgpoints, gray_shape):
    # 카메라 매트릭스 및 왜곡 계수 계산
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray_shape, None, None)
    if not ret:
        raise RuntimeError("Camera calibration failed. Please check your images and try again.")
    
    return mtx, dist


# -----------------------------
# MAIN: 실행 흐름
# -----------------------------
if __name__ == "__main__":
    # 저장 디렉토리 설정
    calibration_dir = "/home/userk/cal_img/chess"  # 체스보드 사진 저장 경로

    # STEP 1: 체스보드 사진 촬영 및 찾기 반복
    objpoints, imgpoints, gray_shape = capture_and_find_chessboard(calibration_dir, num_images=20)

    # STEP 2: 카메라 캘리브레이션
    print("Calibrating camera...")
    mtx, dist = calibrate_camera(objpoints, imgpoints, gray_shape)
    print("Camera matrix:\n", mtx)
    print("Distortion coefficients:\n", dist)

    print("Processing complete.")
