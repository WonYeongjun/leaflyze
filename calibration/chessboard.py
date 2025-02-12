#20장의 체스보드 패턴을 찾을때까지 반복, 20장이 모이면 계수 도출
import cv2
import numpy as np
import time
from picamera2 import Picamera2, Preview
import os

def capture_and_find_chessboard(calibration_dir, num_images=20, checkerboard_size=(7, 7)):
    if not os.path.exists(calibration_dir):
        os.makedirs(calibration_dir)
        print(f"Created directory: {calibration_dir}")

    camera = Picamera2()
    camera.configure(camera.create_still_configuration(raw={'size': (1920, 1440)}))
    preview_config = camera.create_preview_configuration(main={"size": (1920, 1440)})
    camera.configure(preview_config)
    camera.start_preview(Preview.QTGL)
    camera.start()
    time.sleep(1)

    objpoints = []
    imgpoints = []
    objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)

    print(f"Starting chessboard capture and detection... ({num_images} images to be captured)")

    while len(imgpoints) < num_images:
        filename = os.path.join(calibration_dir, f"calibration_{len(imgpoints)}.jpg")
        camera.capture_file(filename)
        print(f"{filename} saved.")
        img = cv2.imread(filename)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)

        if ret:
            print(f"Chessboard found in image: {filename}")
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            imgpoints.append(corners2)
            objpoints.append(objp)
            cv2.drawChessboardCorners(img, checkerboard_size, corners2, ret)
            cv2.imshow("Calibration", img)
            cv2.waitKey(100)
        else:
            print(f"Chessboard not found in image: {filename}")

        time.sleep(2)

    camera.stop_preview()
    camera.close()
    print(f"Captured {len(imgpoints)} valid chessboard images.")
    cv2.destroyAllWindows()

    return objpoints, imgpoints, gray.shape[::-1]

def calibrate_camera(objpoints, imgpoints, gray_shape):
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray_shape, None, None)
    if not ret:
        raise RuntimeError("Camera calibration failed. Please check your images and try again.")
    
    return mtx, dist

if __name__ == "__main__":
    calibration_dir = "/home/userk/cal_img/chess"
    objpoints, imgpoints, gray_shape = capture_and_find_chessboard(calibration_dir, num_images=20)
    print("Calibrating camera...")
    mtx, dist = calibrate_camera(objpoints, imgpoints, gray_shape)
    print("Camera matrix:\n", mtx)
    print("Distortion coefficients:\n", dist)
    print("Processing complete.")