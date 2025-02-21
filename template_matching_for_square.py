import time

import numpy as np
import cv2
from matplotlib import pyplot


from simplification import morphology_diff
from shape_detect import detect_lines

start_time = time.time()


def make_rect(size, angle):
    mask = np.zeros((int(size[1] * 1.2), int(size[0] * 1.2)), dtype=np.uint8)
    center = (size[0] * 1.2 // 2, size[1] * 1.2 // 2)
    rotated_rect = ((center), size, angle)

    # Get the box points (corners of the rotated rectangle)
    box = cv2.boxPoints(rotated_rect)
    box = np.int32(box)

    # Precompute the mask shape for this configuration
    cv2.polylines(mask, [box], isClosed=True, color=255, thickness=31)

    return mask


class PointInfo:
    def __init__(self, x, y, angle, score):
        self.x = x
        self.y = y
        self.angle = angle
        self.score = score


if __name__ == "__main__":
    file_name = "white_back"
    image_path = f"C:/Users/UserK/Desktop/fin/{file_name}.jpg"

    img_bgr = cv2.imread(image_path)
    # img_bgr = get_point_of_interest(img_bgr)
    img_gray, _, _ = morphology_diff(img_bgr)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    shape_image = detect_lines(img_gray)
    pyplot.imshow(shape_image, cmap="gray")
    pyplot.show()
    result = []
    width = 1800  # 1686 #TODO: Change this to the actual size of the square
    height = 1300  # 1378 #TODO: Change this to the actual size of the square
    for angle in range(-10, 11):
        template_mask = make_rect((width, height), angle)
        template = np.ones_like(template_mask) * 255
        hotmap = cv2.matchTemplate(
            shape_image, template, cv2.TM_CCORR, mask=template_mask
        )
        # pyplot.imshow(hotmap, cmap="hot")
        # pyplot.show()
        max_value = np.max(hotmap)  # 최댓값
        max_index = np.unravel_index(np.argmax(hotmap), hotmap.shape)  # 좌표 변환
        print(f"Max value: {max_value}, Max index: {max_index}, angle: {angle}")
        result.append(
            [
                max_index,
                angle,
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
    point_info_list = point_info_list[:1]

    for point_info in point_info_list:
        center = (point_info.x + 0.6 * width, point_info.y + 0.6 * height)
        rotated_rect = ((center), (width, height), point_info.angle)
        # Get the box points (corners of the rotated rectangle)
        box = cv2.boxPoints(rotated_rect)
        box = np.int32(box)

        # Precompute the mask shape for this configuration
        cv2.polylines(shape_image, [box], isClosed=True, color=255, thickness=31)
    # Combine the original image and the shape image side by side
    combined_image = np.hstack((img_rgb, cv2.cvtColor(shape_image, cv2.COLOR_GRAY2RGB)))

    # Save the combined image
    cv2.imwrite(
        f"./output/back/{file_name}_result.png",
        cv2.cvtColor(combined_image, cv2.COLOR_RGB2BGR),
    )
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"작업에 걸린 시간: {elapsed_time} 초")
