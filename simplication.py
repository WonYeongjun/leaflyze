import cv2
import matplotlib.pyplot as plt


def morphlogy_diff(image):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (31, 31))
    img_morphed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

    difference_with_morph = cv2.absdiff(image, img_morphed)
    difference_with_morph_gray = cv2.cvtColor(difference_with_morph, cv2.COLOR_BGR2GRAY)
    difference_with_morph_gray_inversed = 255 - difference_with_morph_gray

    blurred = cv2.GaussianBlur(difference_with_morph_gray_inversed, (9, 9), 10)
    contour_emphasized = cv2.addWeighted(
        difference_with_morph_gray_inversed, 1.5, blurred, -0.5, 0
    )

    img_binary = cv2.threshold(
        contour_emphasized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )[1]
    return difference_with_morph_gray_inversed, contour_emphasized, img_binary


if __name__ == "__main__":

    img = cv2.imread("./image/pink/fin_cal_img_20250207_141129.jpg")
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    difference_gray, sharp, img_binary = morphlogy_diff(img)
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(cv2.cvtColor(img_gray, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Sharped")
    plt.imshow(cv2.cvtColor(sharp, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("binary")
    plt.imshow(img_binary, cmap="gray")
    plt.axis("off")

    plt.show()
