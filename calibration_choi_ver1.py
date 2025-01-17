import cv2
import numpy as np
import time
from picamera2 import Picamera2, Preview
import os

# -----------------------------
# STEP 1: 체스보드 사진 촬영 (2초 간격)
# -----------------------------
def capture_chessboard_images(calibration_dir, num_images=10):
    # 저장 디렉토리 확인 및 생성
    if not os.path.exists(calibration_dir):
        os.makedirs(calibration_dir)
        print(f"Created directory: {calibration_dir}")

    # Picamera2 설정
    camera = Picamera2()
    camera.configure(camera.create_preview_configuration(main={"size": (640, 480)}))

    # 화면에 미리보기 활성화
    camera.start_preview(Preview.QTGL)
    camera.start()

    print(f"Starting chessboard image capture... ({num_images} images)")
    for i in range(num_images):
        filename = os.path.join(calibration_dir, f"calibration_{i}.jpg")
        camera.capture_file(filename)
        print(f"{filename} saved.")
        time.sleep(2)  # 2초 대기

    camera.stop_preview()
    camera.close()
    print("Chessboard image capture complete.")

# -----------------------------
# STEP 2: 카메라 캘리브레이션
# -----------------------------
def calibrate_camera(calibration_dir, checkerboard_size=(7, 6)):
    objpoints = []  # 3D 점
    imgpoints = []  # 2D 점

    # 3D 체스보드 점 생성
    objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)

    # 체스보드 이미지 로드
    images = [os.path.join(calibration_dir, f) for f in os.listdir(calibration_dir) if f.startswith("calibration_")]
    if len(images) == 0:
        raise FileNotFoundError(f"No calibration images found in {calibration_dir}. Please capture images first.")

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 체스보드 코너 검출
        ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)
        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            imgpoints.append(corners2)

            # 검출된 코너를 이미지에 표시
            cv2.drawChessboardCorners(img, checkerboard_size, corners2, ret)
            cv2.imshow("Calibration", img)
            cv2.waitKey(100)

    cv2.destroyAllWindows()

    # 카메라 매트릭스 및 왜곡 계수 계산
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    return mtx, dist

# -----------------------------
# STEP 3: 왜곡 보정 수행
# -----------------------------
def undistort_images(calibration_dir, mtx, dist):
    output_dir = os.path.join(calibration_dir, "undistorted")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    images = [os.path.join(calibration_dir, f) for f in os.listdir(calibration_dir) if f.startswith("calibration_")]
    for i, fname in enumerate(images):
        img = cv2.imread(fname)
        h, w = img.shape[:2]

        # 왜곡 보정 맵 생성
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

        # ROI(Region of Interest) 자르기
        x, y, w, h = roi
        dst = dst[y:y + h, x:x + w]

        # 결과 저장
        output_path = os.path.join(output_dir, f"undistorted_{i}.jpg")
        cv2.imwrite(output_path, dst)
        print(f"{output_path} saved.")

# -----------------------------
# MAIN: 실행 흐름
# -----------------------------
if __name__ == "__main__":
    # 저장 디렉토리 설정
    calibration_dir = "/home/userk/cal_img"  # 체스보드 사진 저장 경로

    # STEP 1: 체스보드 사진 촬영
    capture_chessboard_images(calibration_dir, num_images=10)

    # STEP 2: 카메라 캘리브레이션
    print("Calibrating camera...")
    mtx, dist = calibrate_camera(calibration_dir)
    print("Camera matrix:\n", mtx)
    print("Distortion coefficients:\n", dist)

    # STEP 3: 왜곡 보정 수행
    print("Undistorting images...")
    undistort_images(calibration_dir, mtx, dist)

    print("Processing complete.")
