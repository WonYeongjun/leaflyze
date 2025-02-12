#이미지 불러와서 왜곡보정 후 저장
#ArUco 마커는 좌상단 우상단 우하단 좌하단 에 12, 18, 27, 5 순으로 배치(4x4크기의 마커)
import cv2
import numpy as np
import json

with open("config.json", "r") as f:
    config = json.load(f)

def undistort_image(image_path, camera_matrix, dist_coeffs, target_size=(640*3, 480*3)):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Cannot load image at {image_path}")
        return None

    image_resized = cv2.resize(image, target_size)

    h, w = image_resized.shape[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
    undistorted_image = cv2.undistort(image_resized, camera_matrix, dist_coeffs, None, new_camera_matrix)

    return undistorted_image

if __name__ == "__main__":
    camera_matrix = np.array(config["matrix_coef"])
    dist_coeffs = np.array(config["dist_coeffs"])
    image_path = config["pc_file_path"]

    undistorted = undistort_image(image_path, camera_matrix, dist_coeffs)

    if undistorted is not None:
        save_path = config["pc_modified_file_path"]
        cv2.imwrite(save_path, undistorted)
        print(f"Undistorted image saved to {save_path}")
