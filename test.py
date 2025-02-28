import glob

import cv2
import numpy as np
import matplotlib.pyplot as plt
from shape_detect import detect_SED, canny, line_detector


def morphology_diff(image):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (101, 101))
    img_eroded = cv2.erode(image, kernel, iterations=1)
    img_morphed = cv2.dilate(img_eroded, kernel, iterations=1)

    difference_with_morph = cv2.absdiff(image, img_morphed)
    difference_with_morph_gray = cv2.cvtColor(difference_with_morph, cv2.COLOR_BGR2GRAY)
    difference_with_morph_gray_inversed = 255 - difference_with_morph_gray
    blurred = cv2.GaussianBlur(difference_with_morph_gray_inversed, (5, 5), 10)
    contour_emphasized = cv2.addWeighted(
        difference_with_morph_gray_inversed, 1.5, blurred, -0.5, 0
    )
    sharp_blurred = cv2.GaussianBlur(contour_emphasized, (5, 5), 10)
    return (
        img_eroded,
        img_morphed,
        difference_with_morph_gray,
        difference_with_morph_gray_inversed,
        contour_emphasized,
        sharp_blurred,
    )


def morphology_diff_binary(image):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (101, 101))
    img_morphed = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

    difference = cv2.absdiff(image, img_morphed)
    difference_gray = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
    # difference_gray = 255 - difference_gray

    blurred = cv2.GaussianBlur(difference_gray, (9, 9), 10)
    sharp = cv2.addWeighted(difference_gray, 1.5, blurred, -0.5, 0)

    img_binary = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return difference_gray, sharp, img_binary


def morph(image):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((5, 5), np.uint8)
    morph_img = cv2.morphologyEx(image_gray, cv2.MORPH_CLOSE, kernel)

    return morph_img


def blur(image):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur_img = cv2.GaussianBlur(image_gray, (5, 5), 0)

    return blur_img


def nothing(image):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image_gray


def process_image(image_path):
    img = cv2.imread(image_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = 255 - img_gray

    blurred = blur(img)
    morphed_grad = morph(img)
    morphed, _, _ = morphology_diff_binary(img)
    result_image = detect_SED(cv2.cvtColor(morphed, cv2.COLOR_GRAY2BGR))
    canny_image = canny(morphed)
    merged_image, not_merged_image = line_detector(morphed)
    combined_image = np.hstack(
        [
            img,
            cv2.cvtColor(canny_image, cv2.COLOR_GRAY2RGB),
            cv2.cvtColor(result_image, cv2.COLOR_GRAY2RGB),
            cv2.cvtColor(merged_image, cv2.COLOR_GRAY2RGB),
            cv2.cvtColor(not_merged_image, cv2.COLOR_GRAY2RGB),
        ]
    )
    return combined_image


if __name__ == "__main__":

    folder_path = "C:/Users/UserK/Desktop/fin"
    image_paths = glob.glob(f"{folder_path}/*.jpg")

    result_images = []
    for image_path in image_paths:
        result_image = process_image(image_path)
        result_images.append(result_image)
        print(f"Processed {image_path}")
    combined_image = np.vstack(result_images)

    # 결합된 이미지 저장
    combined_image_path = "C:/Users/UserK/Desktop/fin/combined_result.png"
    cv2.imwrite(combined_image_path, combined_image)

    # 결합된 이미지 시각화
    plt.imshow(combined_image, cmap="gray")
    plt.title("Combined Result Image")
    plt.show()
