import cv2
import numpy as np
import matplotlib.pyplot as plt


def generate_dog(image, num_intervals=5, sigma=1.6):
    """
    DoG(Difference of Gaussian) 계산 함수
    """
    k = 2 ** (1.0 / num_intervals)  # 스케일 인자
    gaussian_images = [image]

    for i in range(num_intervals + 1):  # Gaussian 이미지 생성
        sigma_i = sigma * (k**i)
        blurred = cv2.GaussianBlur(image, (0, 0), sigma_i)
        gaussian_images.append(blurred)

    dog_images = [
        gaussian_images[i + 1] - gaussian_images[i] for i in range(num_intervals)
    ]
    return dog_images


def show_images(images, title="DoG Images"):
    plt.figure(figsize=(12, 6))
    for i, img in enumerate(images):
        plt.subplot(1, len(images), i + 1)
        plt.imshow(img, cmap="gray")
        plt.title(f"DoG {i+1}")
        plt.axis("off")
    plt.suptitle(title)
    plt.show()


# 이미지 불러오기
image = cv2.imread(
    "./image/pink/fin_cal_img_20250207_141129.jpg", cv2.IMREAD_GRAYSCALE
)  # 이미지를 불러와서 그레이스케일 변환

if image is None:
    raise ValueError("이미지를 불러올 수 없습니다. 'sample.jpg' 파일을 확인하세요.")

# DoG 계산
dog_images = generate_dog(image)

# 결과 출력
show_images(dog_images)
