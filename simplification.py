import cv2
import numpy as np
import matplotlib.pyplot as plt


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
    return difference_with_morph_gray_inversed, contour_emphasized, sharp_blurred


def morphology_diff_binary(image):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (31, 31))
    img_morphed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

    difference = cv2.absdiff(image, img_morphed)
    difference_gray = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
    difference_gray = 255 - difference_gray

    blurred = cv2.GaussianBlur(difference_gray, (9, 9), 10)
    sharp = cv2.addWeighted(difference_gray, 1.5, blurred, -0.5, 0)

    img_binary = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return difference_gray, sharp, img_binary


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
    file_name = "pink1"
    image_path = f"C:/Users/UserK/Desktop/fin/{file_name}.jpg"
    img = cv2.imread(image_path)
    # img = cv2.imread("./image/pink/fin_cal_img_20250207_141129.jpg")
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    difference_gray, sharp, blurred = morphology_diff(img)

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    plt.subplot(1, 3, 2)
    plt.title("Sharped")
    plt.imshow(cv2.cvtColor(difference_gray, cv2.COLOR_GRAY2RGB))

    plt.subplot(1, 3, 3)
    plt.title("blur")
    plt.imshow(blurred, cmap="gray")

    plt.show()
