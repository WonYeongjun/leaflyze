import cv2
import matplotlib.pyplot as plt


def make_mask_of_noise(image):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    img_morphed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

    difference_with_morph = cv2.absdiff(image, img_morphed)
    difference_with_morph_gray = cv2.cvtColor(difference_with_morph, cv2.COLOR_BGR2GRAY)
    mask_for_noise = cv2.inRange(difference_with_morph_gray, 0, 10)
    plt.imshow(mask_for_noise, cmap="gray")
    plt.show()

    return mask_for_noise


def morphology_diff(image):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (31, 31))
    img_morphed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

    difference_with_morph = cv2.absdiff(image, img_morphed)
    difference_with_morph_gray = cv2.cvtColor(difference_with_morph, cv2.COLOR_BGR2GRAY)
    difference_with_morph_gray_inversed = 255 - difference_with_morph_gray
    mask = make_mask_of_noise(image)
    difference_with_morph_gray_inversed = cv2.bitwise_and(
        difference_with_morph_gray_inversed,
        difference_with_morph_gray_inversed,
        mask=mask,
    )
    difference_with_morph_gray_inversed[difference_with_morph_gray_inversed == 0] = 255
    blurred = cv2.GaussianBlur(difference_with_morph_gray_inversed, (9, 9), 10)
    contour_emphasized = cv2.addWeighted(
        difference_with_morph_gray_inversed, 1.5, blurred, -0.5, 0
    )

    # threshold_value = 210  # You can change this value to set your own threshold
    # _, img_binary = cv2.threshold(
    #     contour_emphasized, threshold_value, 255, cv2.THRESH_BINARY
    # )
    img_binary = cv2.threshold(
        contour_emphasized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )[1]

    return difference_with_morph_gray_inversed, contour_emphasized, img_binary


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
