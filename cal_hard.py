import cv2
import numpy as np

def undistort_image(image_path, camera_matrix, dist_coeffs, target_size=(640, 480)):
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
