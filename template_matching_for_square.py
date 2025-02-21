import time

import numpy as np
import cv2


from simplification import morphology_diff
from shape_detect import line_detector


def make_rect(size, angle):
    mask = np.zeros((int(size[1] * 1.2), int(size[0] * 1.2)), dtype=np.uint8)
    center = (size[0] * 1.2 // 2, size[1] * 1.2 // 2)
    rotated_rect = ((center), size, angle)

    box = cv2.boxPoints(rotated_rect)
    box = np.int32(box)

    cv2.polylines(mask, [box], isClosed=True, color=255, thickness=31)

    return mask


class PointInfo:
    def __init__(self, x, y, angle, score):
        self.x = x
        self.y = y
        self.angle = angle
        self.score = score


if __name__ == "__main__":

    file_name = "purple_back"
    image_path = f"C:/Users/UserK/Desktop/fin/{file_name}.jpg"

    img_bgr = cv2.imread(image_path)

    _, img_gray = morphology_diff(img_bgr)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    shape_image = line_detector(img_gray)

    result_rough = []
    width = 1800  # 1686 #TODO: Change this to the actual size of the square
    height = 1300  # 1378 #TODO: Change this to the actual size of the square
    template = np.ones((int(height * 1.2), int(width * 1.2)), dtype=np.uint8) * 255
    start_time = time.time()

    for angle in range(-300, 301, 25):
        template_mask = make_rect((width, height), angle / 10)
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

    for angle in range(point_info_rough[0].angle, point_info_rough[1].angle, 1):
        template_mask = make_rect((width, height), angle / 10)
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

    cv2.polylines(shape_image, [box], isClosed=True, color=255, thickness=31)

    combined_image = np.hstack((img_rgb, cv2.cvtColor(shape_image, cv2.COLOR_GRAY2RGB)))

    cv2.imwrite(
        f"./output/back/{file_name}_result.png",
        cv2.cvtColor(combined_image, cv2.COLOR_RGB2BGR),
    )

    elapsed_time = end_time - start_time
    print(f"작업에 걸린 시간: {elapsed_time} 초")
