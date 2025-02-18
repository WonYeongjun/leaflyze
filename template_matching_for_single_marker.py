import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl
from func_for_single_marker import invariant_match_template  # ,template_crop
import time
from image_segmentation import point_of_interest
from simplication import morphlogy_diff

# 시작 시간 기록
start_time = time.time()

if __name__ == "__main__":
    threshold = 130
    img_bgr = cv2.imread("./image/pink/fin_cal_img_20250207_141129.jpg")
    img_bgr = point_of_interest(img_bgr)

    template_bgr = cv2.imread("./image/marker_4.png")
    template_bgr = cv2.resize(
        template_bgr, (0, 0), fx=1, fy=1
    )  # 템플릿 사이즈 조절(초기 설정 필요)
    _, _, img_gray = morphlogy_diff(img_bgr)
    # img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # 실제 이미지 이진화
    im = img_gray
    img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
    template_gray = cv2.cvtColor(template_bgr, cv2.COLOR_RGB2GRAY)
    _, template_gray = cv2.threshold(template_gray, threshold, 255, cv2.THRESH_BINARY)
    template_gray = cv2.GaussianBlur(template_gray, (11, 11), 0)
    height, width = template_gray.shape

    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    # ax1.imshow(template_gray, cmap="gray")
    # ax1.set_title("Original Grayscale Image")
    # ax2.imshow(img_gray, cmap="gray")
    # ax2.set_title("Processed Grayscale Image")
    # plt.show()
    points_list = invariant_match_template(
        grayimage=img_gray,
        graytemplate=template_gray,
        method="TM_CCOEFF",
        matched_thresh=0.3,
        rot_range=[-10, 10],
        rot_interval=2,
        scale_range=[90, 110],
        scale_interval=2,
        rm_redundant=True,
        minmax=True,
    )
    fig, ax = plt.subplots(1)
    plt.gcf().canvas.manager.set_window_title("Template Matching Results: Rectangles")
    ax.imshow(img_rgb)
    print(len(points_list))
    points_list = points_list[:10]
    centers_list = []
    real_point = []
    for point_info in points_list:
        point = point_info[0]
        angle = point_info[1]
        scale = point_info[2]
        print(
            f"No.{str(points_list.index(point_info))} matched point: {point}, angle: {angle}, scale: {scale}, score: {point_info[3]}"
        )
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
            str(points_list.index(point_info)),
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
    # plt.grid(True)
    end_time = time.time()
    # plt.show()
    elapsed_time = end_time - start_time
    print(f"작업에 걸린 시간: {elapsed_time} 초")
