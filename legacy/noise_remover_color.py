import cv2
import numpy as np
import matplotlib.pyplot as plt


def segment_image_with_spatial_info(image, k=3, spatial_weight=0.01):
    # 이미지를 HSV 색상 공간으로 변환
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # H와 S 채널만 사용 (밝기(V) 채널 무시)
    hs_image = hsv_image[:, :, :2]
    pixel_values = hs_image.reshape((-1, 2))
    pixel_values = np.float32(pixel_values)

    # 이미지 크기
    h, w, c = image.shape

    # 공간 정보 추가
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    spatial_info = np.stack((x, y), axis=-1).reshape((-1, 2))
    spatial_info = np.float32(spatial_info)
    h_norm = h / np.max([h, w])
    w_norm = w / np.max([h, w])
    spatial_weight_norm = spatial_weight / np.max([h, w])

    # 거리 정보의 가중치 조정
    spatial_info[:, 0] *= w_norm * spatial_weight_norm
    spatial_info[:, 1] *= h_norm * spatial_weight_norm

    # 색상 정보와 공간 정보를 결합
    combined_data = np.hstack((pixel_values, spatial_info))

    # K-means 클러스터링 적용
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(
        combined_data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
    )

    # 각 픽셀에 클러스터 중심 색상 할당
    centers = centers[:, :2]  # 색상 정보만 사용
    centers = np.uint8(centers)
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape((image.shape[0], image.shape[1], 2))

    # HSV 이미지의 H와 S 채널을 클러스터 중심 색상으로 대체
    hsv_segmented_image = np.copy(hsv_image)
    hsv_segmented_image[:, :, :2] = segmented_image

    # HSV 이미지를 다시 RGB 색상 공간으로 변환
    segmented_image_rgb = cv2.cvtColor(hsv_segmented_image, cv2.COLOR_HSV2RGB)
    hsv_segmented_image[:, :, :2] = segmented_image
    hsv_segmented_image[:, :, 2] = 255  # 밝기 채널을 최대값으로 설정
    segmented_image_rgb = cv2.cvtColor(hsv_segmented_image, cv2.COLOR_HSV2RGB)

    return segmented_image_rgb, labels, centers


if __name__ == "__main__":
    # 이미지 로드
    img = cv2.imread("C:/Users/UserK/Desktop/fin/fin_purple_img.jpg")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 이미지 세그멘테이션
    k = 5  # 클러스터 수
    spatial_weight = 0.1  # 거리 정보의 가중치
    segmented_image, labels, centers = segment_image_with_spatial_info(
        img_rgb, k, spatial_weight
    )

    # 결과 출력
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(img_rgb)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Segmented Image")
    plt.imshow(segmented_image)
    plt.axis("off")

    plt.show()
