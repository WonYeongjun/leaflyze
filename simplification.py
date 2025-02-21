import cv2
import numpy as np
import matplotlib.pyplot as plt


def morphology_diff(image):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (31, 31))
    img_morphed = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

    difference_with_morph = cv2.absdiff(image, img_morphed)
    difference_with_morph_gray = cv2.cvtColor(difference_with_morph, cv2.COLOR_BGR2GRAY)
    difference_with_morph_gray_inversed = 255 - difference_with_morph_gray

    blurred = cv2.GaussianBlur(difference_with_morph_gray_inversed, (9, 9), 10)
    contour_emphasized = cv2.addWeighted(
        difference_with_morph_gray_inversed, 1.5, blurred, -0.5, 0
    )

    return difference_with_morph_gray_inversed, contour_emphasized


def morph(image):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((5, 5), np.uint8)
    morph_img = cv2.morphologyEx(image_gray, cv2.MORPH_GRADIENT, kernel)

    return morph_img


def blur(image):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur_img = cv2.GaussianBlur(image_gray, (5, 5), 0)

    return blur_img


def nothing(image):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image_gray


if __name__ == "__main__":

    img = cv2.imread("C:/Users/UserK/Desktop/fin/fin_purple_img.jpg")
    # img = cv2.imread("./image/pink/fin_cal_img_20250207_141129.jpg")
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    threshold_value = 90  # You can change this value to set your own threshold
    _, img_2jin = cv2.threshold(img_gray, threshold_value, 255, cv2.THRESH_BINARY)
    difference_gray, sharp, img_binary = morphology_diff(img)

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(cv2.cvtColor(difference_gray, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Sharped")
    plt.imshow(cv2.cvtColor(img_2jin, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("binary")
    plt.imshow(img_binary, cmap="gray")
    plt.axis("off")

    plt.show()
