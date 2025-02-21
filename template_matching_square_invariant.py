import time

import numpy as np
import cv2
import matplotlib
from matplotlib import pyplot, patches

from invariant_match_template_for_square import invariant_match_template
from get_point_of_interest import get_point_of_interest
from simplification import morphology_diff
from shape_detect import detect_lines

start_time = time.time()


class PointInfo:
    def __init__(self, x, y, angle, scale, score):
        self.x = x
        self.y = y
        self.angle = angle
        self.scale = scale
        self.score = score


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


if __name__ == "__main__":
    image_path = "C:/Users/UserK/Desktop/fin/white_back.jpg"

    img_bgr = cv2.imread(image_path)
    # img_bgr = get_point_of_interest(img_bgr)
    img_gray, _, _ = morphology_diff(img_bgr)
    shape_image = detect_lines(img_gray)

    img_rgb = cv2.cvtColor(shape_image, cv2.COLOR_GRAY2RGB)
    width = 1800  # 1686 #TODO: Change this to the actual size of the square
    height = 1300  # 1378 #TODO: Change this to the actual size of the square
    mask = make_rect((width, height), 0)
    template = np.ones_like(mask) * 255
    print(template.shape)
    result = invariant_match_template(
        grayimage=shape_image,
        graytemplate=template,
        matched_thresh=100000000,
        rot_range=[-10, 10],
        rot_interval=2,
        scale_range=[100, 101],
        scale_interval=2,
        mask=mask,
    )

    point_info_list = [
        PointInfo(
            x=point_info[0][0],
            y=point_info[0][1],
            angle=point_info[1],
            scale=point_info[2],
            score=point_info[3],
        )
        for point_info in result
    ]
    fig, axes = pyplot.subplots(1)
    axes.imshow(img_rgb)
    pyplot.gcf().canvas.manager.set_window_title("Template Matching Results: Grid")
    point_info_list = point_info_list[:2]
    for i, point_info in enumerate(point_info_list):
        print(
            f"No.{i} matched point: {(point_info.x,point_info.y)}, angle: {point_info.angle}, scale: {point_info.scale}, score: {point_info.score}"
        )
        axes.scatter(
            point_info.x + (width / 2) * point_info.scale[0] / 100,
            point_info.y + (height / 2) * point_info.scale[1] / 100,
            s=20,
            color="red",
        )
        idx = (
            point_info.x + (width / 2) * point_info.scale[0] / 100,
            point_info.y + (height / 2) * point_info.scale[1] / 100,
        )
        axes.scatter(point_info.x, point_info.y, s=20, color="green")
        rectangle = patches.Rectangle(
            (point_info.x, point_info.y),
            width * point_info.scale[0] / 100,
            height * point_info.scale[1] / 100,
            color="red",
            alpha=0.50,
            label="Matched box",
        )
        axes.text(
            point_info.x,
            point_info.y - 10,
            f"{str(i)} : {point_info.score:.3f}",
            color="blue",
            fontsize=12,
            weight="bold",
        )
        transform = (
            matplotlib.transforms.Affine2D().rotate_deg_around(
                point_info.x + width / 2 * point_info.scale[0] / 100,
                point_info.y + height / 2 * point_info.scale[1] / 100,
                point_info.angle,
            )
            + axes.transData
        )
        rectangle.set_transform(transform)
        axes.add_patch(rectangle)
    axes.grid(True)
    axes.legend(handles=[rectangle])

    end_time = time.time()
    pyplot.show()

    elapsed_time = end_time - start_time
    print(f"작업에 걸린 시간: {elapsed_time} 초")
