import cv2
import numpy as np
import matplotlib.pyplot as plt

from simplification import morphology_diff
from shape_detect import line_detector, detect_SED
from outline import center_emphasize
from get_point_of_interest import get_point_of_interest


def masking_honeycomb(image):

    shape_image = detect_SED(image)
    # Apply morphological operations to enlarge white areas
    kernel = np.ones((5, 5), np.uint8)
    morph_image = cv2.dilate(shape_image, kernel, iterations=2)
    morph_image = cv2.threshold(
        morph_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )[1]

    contours, _ = cv2.findContours(morph_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    top_10_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
    colors = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
        (128, 0, 0),
        (0, 128, 0),
        (0, 0, 128),
        (128, 128, 0),
    ]
    mask = np.ones_like(image[:, :, 0]) * 255

    cv2.drawContours(mask, top_10_contours[:2], -1, 0, thickness=cv2.FILLED)

    # for i, contour in enumerate(top_10_contours):
    #     cv2.drawContours(image, [contour], -1, colors[i % len(colors)], 2)
    result_image = cv2.bitwise_and(image, image, mask=mask)
    return result_image


if __name__ == "__main__":
    file_name = "white2_rot"
    image_path = f"C:/Users/UserK/Desktop/fin/{file_name}.jpg"

    # 6. 이미지 로드
    image = cv2.imread(image_path)
    result_image = masking_honeycomb(image)
    plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
    plt.show()
