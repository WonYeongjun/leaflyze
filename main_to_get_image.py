import time

import numpy as np
import cv2

from simplification import morphology_diff_binary
from shape_detect import detect_SED, line_detector_without_merge
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
    file_name = "black1"
    image_path = f"C:/Users/UserK/Desktop/fin/{file_name}.jpg"

    img_bgr = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_bgr, mask, edges_image = masking_honeycomb(img_bgr)
    cv2.imwrite(
        f"C:\\Users\\UserK\\Desktop\\{file_name}\\SED.png",
        cv2.cvtColor(edges_image, cv2.COLOR_RGB2BGR),
    )

    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_binary = cv2.threshold(img_gray, 220, 255, cv2.THRESH_BINARY)[1]
    cv2.imwrite(
        f"C:\\Users\\UserK\\Desktop\\{file_name}\\Binary Image 220.png", img_binary
    )

    _, morphed_image, _ = morphology_diff_binary(img_bgr)
    cv2.imwrite(
        f"C:\\Users\\UserK\\Desktop\\{file_name}\\Morphology Difference.png",
        cv2.cvtColor(morphed_image, cv2.COLOR_RGB2BGR),
    )

    morph_edges_image = detect_SED(cv2.cvtColor(morphed_image, cv2.COLOR_GRAY2BGR))

    morphed_image_binary = cv2.threshold(morphed_image, 30, 255, cv2.THRESH_BINARY)[1]
    cv2.imwrite(
        f"C:\\Users\\UserK\\Desktop\\{file_name}\\Morphology Difference Binary.png",
        morphed_image_binary,
    )

    image_binary_and_morphed_image_binary = cv2.bitwise_and(
        img_binary, morphed_image_binary
    )
    cv2.imwrite(
        f"C:\\Users\\UserK\\Desktop\\{file_name}\\Binary Image 220 & Morphology Difference.png",
        image_binary_and_morphed_image_binary,
    )

    kernel_mask = np.ones((15, 15), np.uint8)
    mask = cv2.erode(mask, kernel_mask, iterations=1)

    cv2.imwrite(
        f"C:\\Users\\UserK\\Desktop\\{file_name}\\morphed_image_SED.png",
        morph_edges_image,
    )

    edges_image_combined = cv2.bitwise_or(edges_image, morph_edges_image)
    cv2.imwrite(
        f"C:\\Users\\UserK\\Desktop\\{file_name}\\SED or morphed_image_SED.png",
        edges_image_combined,
    )

    edges_image_combined = cv2.bitwise_and(edges_image_combined, mask)
    edges_image_combined = cv2.threshold(
        edges_image_combined, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )[1]
    edges_image_combined = cv2.bitwise_and(
        edges_image_combined, image_binary_and_morphed_image_binary
    )
    cv2.imwrite(
        f"C:\\Users\\UserK\\Desktop\\{file_name}\\SED_combined & Binary Image 220 & Morphology Difference.png",
        edges_image_combined,
    )

    lines = line_detector_without_merge(edges_image_combined)

    cv2.imwrite(f"C:\\Users\\UserK\\Desktop\\{file_name}\\Lines.png", lines)

    edges_image = cv2.bitwise_and(edges_image, mask)
    edges_image = cv2.threshold(
        edges_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )[1]
    edges_image = cv2.bitwise_and(edges_image, image_binary_and_morphed_image_binary)
    cv2.imwrite(
        f"C:\\Users\\UserK\\Desktop\\{file_name}\\SED & Binary Image 220 & Morphology Difference.png",
        edges_image,
    )

    result_rough = []
    width = 1800  # 1686 #TODO: Change this to the actual size of the square
    height = 1300  # 1378 #TODO: Change this to the actual size of the square
    template = np.ones((int(height * 1.2), int(width * 1.2)), dtype=np.uint8) * 255
    for angle in range(-350, 351, 25):
        template = make_rect((width, height), angle / 10, 31)
        template = cv2.blur(template, (5, 5))
        template_mask = make_rect((width, height), angle / 10, 51)
        hotmap = cv2.matchTemplate(lines, template, cv2.TM_CCORR, mask=template_mask)

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

    best_point_in_rough = point_info_list_rough[0]

    cropped = lines[
        int(best_point_in_rough.y - 0.2 * height) : int(
            best_point_in_rough.y + 1.4 * height
        ),
        int(best_point_in_rough.x - 0.2 * width) : int(
            best_point_in_rough.x + 1.4 * width
        ),
    ]
    is_cropped = True
    if cropped.size <= 0:
        is_cropped = False
        print(cropped.size)
        cropped = lines
    cv2.imwrite(f"C:\\Users\\UserK\\Desktop\\{file_name}\\Cropped.png", cropped)
    template = np.ones((int(height * 1.2), int(width * 1.2)), dtype=np.uint8) * 255
    for angle in range(point_info_rough[0].angle, point_info_rough[1].angle + 1, 1):
        template = make_rect((width, height), angle / 10, 31)
        template = cv2.blur(template, (5, 5))
        template_mask = make_rect((width, height), angle / 10, 51)
        hotmap = cv2.matchTemplate(cropped, template, cv2.TM_CCORR, mask=template_mask)
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
    end_time = time.time()
    if is_cropped:
        center = (
            point_info.x + 0.4 * width + best_point_in_rough.x,
            point_info.y + 0.4 * height + best_point_in_rough.y,
        )
    else:
        center = (point_info.x + 0.6 * width, point_info.y + 0.6 * height)

    rotated_rect = ((center), (width, height), point_info.angle)

    box = cv2.boxPoints(rotated_rect)
    box = np.int32(box)

    cv2.polylines(img_rgb, [box], isClosed=True, color=(255, 0, 0), thickness=5)

    combined_image = np.hstack((cv2.cvtColor(lines, cv2.COLOR_GRAY2RGB), img_rgb))

    cv2.imwrite(
        f"C:\\Users\\UserK\\Desktop\\{file_name}\\result.png",
        cv2.cvtColor(combined_image, cv2.COLOR_RGB2BGR),
    )

    elapsed_time = end_time - start_time
    print(f"작업에 걸린 시간: {elapsed_time} 초")
