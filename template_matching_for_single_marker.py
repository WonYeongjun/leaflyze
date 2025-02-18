import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl
from func_for_single_marker import invariant_match_template  # ,template_crop
import time
from image_segmentation import point_of_interest
from simplication import morphlogy_diff
import glob
import os

start_time = time.time()


example_fabric_type = "pink"


if __name__ == "__main__":
    ans_list = []
    image_files = glob.glob(f"./image/{example_fabric_type}/*.jpg")
    for file in image_files:
        img_bgr = cv2.imread(file)
        img_bgr = point_of_interest(img_bgr)
        _, _, img_gray = morphlogy_diff(img_bgr)
        img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)

        template_bgr = cv2.imread("./image/marker_4.png")
        template_bgr = cv2.resize(
            template_bgr, (0, 0), fx=1, fy=1
        )  # 템플릿 사이즈 조절(초기 설정 필요)
        template_gray = cv2.cvtColor(template_bgr, cv2.COLOR_RGB2GRAY)
        template_blur = cv2.GaussianBlur(template_gray, (11, 11), 0)
        height, width = template_blur.shape

        points_list = invariant_match_template(
            grayimage=img_gray,
            graytemplate=template_blur,
            matched_thresh=0.5,
            rot_range=[-10, 10],
            rot_interval=2,
            scale_range=[90, 110],
            scale_interval=2,
        )
        fig, ax = plt.subplots(1)
        plt.gcf().canvas.manager.set_window_title(
            "Template Matching Results: Rectangles"
        )
        ax.imshow(img_rgb)
        points_list = points_list[:10]
        centers_list = []
        real_point = []
        for point_info in points_list:
            point = point_info[0]
            angle = point_info[1]
            scale = point_info[2]
            centers_list.append([point, scale])
            plt.scatter(
                point[0] + (width / 2) * scale[0] / 100,
                point[1] + (height / 2) * scale[1] / 100,
                s=20,
                color="red",
            )
            idx = (
                point[0] + (width / 2) * scale[0] / 100,
                point[1] + (height / 2) * scale[1] / 100,
            )
            real_point.append([idx])
            plt.scatter(point[0], point[1], s=20, color="green")
            rectangle = patches.Rectangle(
                (point[0], point[1]),
                width * scale[0] / 100,
                height * scale[1] / 100,
                color="red",
                alpha=0.50,
                label="Matched box",
            )
            plt.text(
                point[0],
                point[1] - 10,
                f"{str(points_list.index(point_info))} : {point_info[3]:.3f}",
                color="blue",
                fontsize=12,
                weight="bold",
            )

            transform = (
                mpl.transforms.Affine2D().rotate_deg_around(
                    point[0] + width / 2 * scale[0] / 100,
                    point[1] + height / 2 * scale[1] / 100,
                    angle,
                )
                + ax.transData
            )
            rectangle.set_transform(transform)
            ax.add_patch(rectangle)
            plt.legend(handles=[rectangle])
        ans_list.append([point_info[3] for point_info in points_list[:5]])
        plt.grid(True)
        file_name = os.path.basename(file)
        image_save_path = f"./output/{example_fabric_type}/output_{file_name}"
        plt.savefig(
            image_save_path, dpi=300
        )  # dpi를 300으로 설정하여 고해상도 이미지 저장
    end_time = time.time()

    file_path = f"./output/{example_fabric_type}/output_{example_fabric_type}.txt"

    # 리스트를 텍스트 파일로 저장
    with open(file_path, "w") as file:
        for item in ans_list:
            file.write(f"{item}\n")  # 각 항목을 한 줄에 저장

    elapsed_time = end_time - start_time
    print(f"작업에 걸린 시간: {elapsed_time} 초")
