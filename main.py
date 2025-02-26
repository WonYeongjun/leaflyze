import time

import numpy as np
import cv2
import matplotlib.pyplot as pyplot

from simplification import morphology_diff
from shape_detect import line_detector, detect_SED, line_detector_without_merge
from get_point_of_interest import get_point_of_interest
from get_contours_of_honeycomb import masking_honeycomb


def make_rect(size, angle, thickness=31):
    mask = np.zeros((int(size[1] * 1.2), int(size[0] * 1.2)), dtype=np.uint8)
    center = (size[0] * 1.2 // 2, size[1] * 1.2 // 2)
    rotated_rect = ((center), size, angle)

    box = cv2.boxPoints(rotated_rect)
    box = np.int32(box)

    cv2.polylines(mask, [box], isClosed=True, color=255, thickness=thickness)

    return mask


class PointInfo:
    def __init__(self, x, y, angle, score):
        self.x = x
        self.y = y
        self.angle = angle
        self.score = score


if __name__ == "__main__":
    start_time = time.time()
    file_name = "black2"
    image_path = f"C:/Users/UserK/Desktop/fin/{file_name}.jpg"

    img_bgr = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_bgr, mask = masking_honeycomb(img_bgr)
    # img_bgr = get_point_of_interest(img_bgr)
    # _, _, _, _, img_gray, _ = morphology_diff(img_bgr)
    img_gray = 255 - cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    shape_image = line_detector(img_gray)

    kernel = np.ones((5, 5), np.uint8)
    shape_image = cv2.dilate(shape_image, kernel, iterations=1)

    kernel_mask = np.ones((15, 15), np.uint8)
    mask = cv2.erode(mask, kernel_mask, iterations=1)
    shape_image = cv2.bitwise_and(shape_image, mask)
    pyplot.imshow(shape_image, cmap="gray")
    pyplot.show()

    # shape_image = detect_SED(img_bgr)
    # shape_image = cv2.threshold(
    #     shape_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    # )[1]
    # pyplot.imshow(shape_image, cmap="gray")
    # pyplot.show()

    result_rough = []
    width = 1800  # 1686 #TODO: Change this to the actual size of the square
    height = 1300  # 1378 #TODO: Change this to the actual size of the square
    template = np.ones((int(height * 1.2), int(width * 1.2)), dtype=np.uint8) * 255
    for angle in range(0, 351, 25):
        template = make_rect((width, height), angle / 10, 31)
        template = cv2.blur(template, (5, 5))
        template_mask = make_rect((width, height), angle / 10, 51)
        hotmap = cv2.matchTemplate(
            shape_image, template, cv2.TM_CCORR, mask=template_mask
        )

        max_value = np.max(hotmap)
        max_index = np.unravel_index(np.argmax(hotmap), hotmap.shape)
        print(f"Max value: {max_value}, Max index: {max_index}, angle: {angle/10}")
        result_rough.append(
            [
                max_index,
                angle,
                max_value,
            ]
        )
    point_info_list_rough = [
        PointInfo(
            x=point_info[0][1],
            y=point_info[0][0],
            angle=point_info[1],
            score=point_info[2],
        )
        for point_info in result_rough
    ]

    point_info_list_rough.sort(key=lambda point: point.score, reverse=True)
    point_info_rough = point_info_list_rough[:2]
    result = []
    point_info_rough.sort(key=lambda point: point.angle)
    print(point_info_rough[0].angle, point_info_rough[1].angle)
    template = np.ones((int(height * 1.2), int(width * 1.2)), dtype=np.uint8) * 255
    for angle in range(point_info_rough[0].angle, point_info_rough[1].angle + 1, 1):
        template = make_rect((width, height), angle / 10, 31)
        template = cv2.blur(template, (5, 5))
        template_mask = make_rect((width, height), angle / 10, 51)
        hotmap = cv2.matchTemplate(
            shape_image, template, cv2.TM_CCORR, mask=template_mask
        )
        max_value = np.max(hotmap)
        max_index = np.unravel_index(np.argmax(hotmap), hotmap.shape)
        print(f"Max value: {max_value}, Max index: {max_index}, angle: {angle/10}")
        result.append(
            [
                max_index,
                angle / 10,
                max_value,
            ]
        )

    end_time = time.time()

    point_info_list = [
        PointInfo(
            x=point_info[0][1],
            y=point_info[0][0],
            angle=point_info[1],
            score=point_info[2],
        )
        for point_info in result
    ]
    point_info_list.sort(key=lambda point: point.score, reverse=True)
    point_info = point_info_list[0]

    center = (point_info.x + 0.6 * width, point_info.y + 0.6 * height)
    rotated_rect = ((center), (width, height), point_info.angle)

    box = cv2.boxPoints(rotated_rect)
    box = np.int32(box)

    cv2.polylines(img_rgb, [box], isClosed=True, color=(255, 0, 0), thickness=5)

    combined_image = np.hstack((cv2.cvtColor(shape_image, cv2.COLOR_GRAY2RGB), img_rgb))

    cv2.imwrite(
        f"./output/back/{file_name}_result.png",
        cv2.cvtColor(combined_image, cv2.COLOR_RGB2BGR),
    )

    elapsed_time = end_time - start_time
    print(f"작업에 걸린 시간: {elapsed_time} 초")
