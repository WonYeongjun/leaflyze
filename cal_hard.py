import cv2
import numpy as np
import matplotlib.pyplot as plt

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

def display_images(original, undistorted):
    # 보정 전후 이미지 비교
    plt.figure(figsize=(12, 6))

    # 원본 이미지
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    # 보정된 이미지
    plt.subplot(1, 2, 2)
    plt.title("Undistorted Image")
    plt.imshow(cv2.cvtColor(undistorted, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.tight_layout()
    plt.show()

# MAIN: 실행
if __name__ == "__main__":
    # 카메라 매트릭스 및 왜곡 계수
    camera_matrix = np.array([[942.61474276 ,  0.     ,    297.72929912],
 [  0.      ,   940.60838882, 222.08543923],
 [  0.         ,  0.      ,     1.        ]])

    dist_coeffs = np.array([-0.04801603,  0.15976978 ,-0.00377979 ,-0.00425458,  0.15109397])

    # 테스트 이미지 경로
    image_path = "/home/userk/cal_img/raw/raw_img.jpg"  # 촬영한 이미지 경로

    # 왜곡 보정
    original, undistorted = undistort_image(image_path, camera_matrix, dist_coeffs)

    if original is not None and undistorted is not None:
        # 이미지 비교
        display_images(original, undistorted)
