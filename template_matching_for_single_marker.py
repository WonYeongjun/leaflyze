import glob
import os
import time

import cv2
import matplotlib
from matplotlib import pyplot, patches
import math

from invariant_match_template import invariant_match_template
from get_point_of_interest import get_point_of_interest
from simplification import morphology_diff

start_time = time.time()


example_fabric_type = "pink"


class PointInfo:
    def __init__(self, x, y, angle, scale, score):
        self.x = x
        self.y = y
        self.angle = angle
        self.scale = scale
        self.score = score


if __name__ == "__main__":
    ans_list = []
    # image_files = glob.glob(f"./image/{example_fabric_type}/*.jpg")
    image_files = glob.glob("C:/Users/UserK/Desktop/fin/*.jpg")

    grid_cols = 2
    grid_rows = math.ceil(len(image_files) / grid_cols)
    fig, axes = pyplot.subplots(grid_rows, grid_cols, figsize=(15, 15))

    pyplot.gcf().canvas.manager.set_window_title("Template Matching Results: Grid")
    for index, file in enumerate(image_files):
        img_bgr = cv2.imread(file)
        img_bgr = get_point_of_interest(img_bgr)
        _, _, img_gray = morphology_diff(img_bgr)
        img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)

        template_bgr = cv2.imread("./image/marker_4.png")
        template_bgr = cv2.resize(
            template_bgr, (0, 0), fx=1, fy=1
        )  # TODO: 템플릿 사이즈 조절
        template_gray = cv2.cvtColor(template_bgr, cv2.COLOR_RGB2GRAY)
        template_blur = cv2.GaussianBlur(template_gray, (11, 11), 0)
        height, width = template_blur.shape

        result = invariant_match_template(
            grayimage=img_gray,
            graytemplate=template_blur,
            matched_thresh=0.5,
            rot_range=[-10, 10],
            rot_interval=2,
            scale_range=[90, 110],
            scale_interval=2,
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

        local_axes = axes[index // grid_cols, index % grid_cols]
        local_axes.imshow(img_rgb)
        point_info_list = point_info_list[:10]
        for i, point_info in enumerate(point_info_list):
            local_axes.scatter(
                point_info.x + (width / 2) * point_info.scale[0] / 100,
                point_info.y + (height / 2) * point_info.scale[1] / 100,
                s=20,
                color="red",
            )
            idx = (
                point_info.x + (width / 2) * point_info.scale[0] / 100,
                point_info.y + (height / 2) * point_info.scale[1] / 100,
            )
            local_axes.scatter(point_info.x, point_info.y, s=20, color="green")
            rectangle = patches.Rectangle(
                (point_info.x, point_info.y),
                width * point_info.scale[0] / 100,
                height * point_info.scale[1] / 100,
                color="red",
                alpha=0.50,
                label="Matched box",
            )
            local_axes.text(
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
                + local_axes.transData
            )
            rectangle.set_transform(transform)
            local_axes.add_patch(rectangle)
            local_axes.legend(handles=[rectangle])
            local_axes.grid(True)

        ans_list.append([point_info.score for point_info in point_info_list[:5]])

    pyplot.tight_layout()
    file_name = os.path.basename(file)
    image_save_path = f"./output/{example_fabric_type}/output_{example_fabric_type}.png"
    pyplot.savefig(image_save_path, dpi=300)
    pyplot.show()

    file_path = f"./output/{example_fabric_type}/output_{example_fabric_type}.txt"
    with open(file_path, "w") as file:
        for item in ans_list:
            file.write(f"{item}\n")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"작업에 걸린 시간: {elapsed_time} 초")
