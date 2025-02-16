import cv2
import matplotlib.pyplot as plt


def morphlogy_diff(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (31, 31))
    img_morphed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    difference = cv2.absdiff(img, img_morphed)
    difference_gray = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
    difference_gray = 255 - difference_gray

    return difference_gray


if __name__ == "__main__":

    img = cv2.imread("./image/pink/fin_cal_img_20250207_141129.jpg")
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    difference_gray = morphlogy_diff(img)
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(cv2.cvtColor(img_gray, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Morphed Image")
    plt.imshow(cv2.cvtColor(img_gray, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Difference Image (Gray)")
    plt.imshow(difference_gray, cmap="gray")
    plt.axis("off")

    plt.show()
