import cv2
import numpy as np
import time
from picamera2 import Picamera2, Preview
import os

# -----------------------------
# STEP 1: 체스보드 사진 촬영 (10장, 2초 간격)
# -----------------------------
def capture_chessboard_images(output_dir, num_images=10, delay=2):
    """
    체스보드 사진을 지정된 디렉토리에 저장합니다.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    camera = Picamera2()
    camera.configure(camera.create_preview_configuration(main={"size": (640, 480)}))
    camera.start_preview(Preview.QTGL)
    camera.start()

    print(f"Starting chessboard image capture... ({num_images} images)")
    for i in range(num_images):
        filename = os.path.join(output_dir, f"calibration_{i}.jpg")
        camera.capture_file(filename)
        print(f"{filename} saved.")
        time.sleep(delay)  # 2초 간격

    camera.stop_preview()
    camera.close()
    print("Initial chessboard image capture complete.")

# -----------------------------
# STEP 2: 5초 후 사진 한 장 촬영
# -----------------------------
def capture_single_image(output_dir, filename="single_capture.jpg"):
    """
    5초 대기 후 사진 한 장을 촬영하여 저장합니다.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    camera = Picamera2()
    camera.configure(camera.create_preview_configuration(main={"size": (640, 480)}))
    camera.start_preview(Preview.QTGL)
    camera.start()

    print("Waiting 5 seconds before capturing single image...")
    time.sleep(5)

    filepath = os.path.join(output_dir, filename)
    camera.capture_file(filepath)
    print(f"Captured and saved single image: {filepath}")

    camera.stop_preview()
    camera.close()

# -----------------------------
# STEP 3: 카메라 캘리브레이션
# -----------------------------
def calibrate_camera(image_dir, checkerboard_size=(7, 6)):
    """
    카메라 캘리브레이션을 수행하고 카메라 매트릭스와 왜곡 계수를 반환합니다.
    """
    objpoints = []  # 3D 점
    imgpoints = []  # 2D 점

    # 3D 체스보드 점 생성
    objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)

    # 체스보드 이미지 로드
    images = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(".jpg")]
    if len(images) == 0:
        raise FileNotFoundError(f"No images found in {image_dir}. Please capture images first.")

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

    # 카메라 매트릭스 및 왜곡 계수 계산
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    return mtx, dist

# -----------------------------
# STEP 4: 왜곡 보정 수행
# -----------------------------
def undistort_image(image_path, output_path, mtx, dist):
    """
    주어진 이미지를 왜곡 보정하고 결과를 저장합니다.
    """
    img = cv2.imread(image_path)
    h, w = img.shape[:2]

    # 왜곡 보정 맵 생성
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

    # ROI(Region of Interest) 자르기
    x, y, w, h = roi
    dst = dst[y:y + h, x:x + w]

    # 결과 저장
    cv2.imwrite(output_path, dst)
    print(f"Undistorted image saved: {output_path}")

# -----------------------------
# MAIN: 실행 흐름
# -----------------------------
if __name__ == "__main__":
    # 디렉토리 설정
    calibration_dir = "/home/userk/cal_img"
    single_image_dir = "/home/userk/ex_img"

    # STEP 1: 체스보드 사진 촬영
    capture_chessboard_images(calibration_dir, num_images=10)

    # STEP 2: 5초 후 사진 한 장 촬영
    capture_single_image(single_image_dir, filename="single_capture.jpg")

    # STEP 3: 카메라 캘리브레이션
    print("Calibrating camera...")
    mtx, dist = calibrate_camera(calibration_dir)
    print("Camera matrix:\n", mtx)
    print("Distortion coefficients:\n", dist)

    # STEP 4: 왜곡 보정 수행
    print("Undistorting single captured image...")
    single_image_path = os.path.join(single_image_dir, "single_capture.jpg")
    undistorted_image_path = os.path.join(single_image_dir, "undistorted_single_capture.jpg")
    undistort_image(single_image_path, undistorted_image_path, mtx, dist)

    print("Processing complete.")
