import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl

from InvariantTM_mask import invariant_match_template  # ,template_crop
import time

# 시작 시간 기록
start_time = time.time()

if __name__ == "__main__":
    img_bgr = cv2.imread("./image/cloth5.jpg")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    template_bgr = cv2.imread("./image/marker_ideal.jpg", cv2.IMREAD_UNCHANGED)
    template_bgr = cv2.resize(
        template_bgr, (0, 0), fx=0.2, fy=0.2
    )  # 템플릿 사이즈 조절(촬영 후에 조정필요)
    formask = cv2.imread("./image/marker_ideal_back.jpg", cv2.IMREAD_UNCHANGED)
    formask = cv2.resize(formask, (0, 0), fx=0.2, fy=0.2)
    print(formask.shape)
    formask = formask[:, :, 3]
    print(formask.shape)
    # template_rgb = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2RGB)
    # img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    # _, img_binary = cv2.threshold(
    #     img_gray, 127, 255, cv2.THRESH_BINARY
    # )  # 실제 이미지 이진화
    # img_rgb = cv2.cvtColor(img_binary, cv2.COLOR_GRAY2RGB)
    # img_rgb = cv2.cvtColor(cv2.Canny(img_rgb, 240, 240), cv2.COLOR_GRAY2RGB)
    # template_rgb = cv2.cvtColor(cv2.Canny(template_rgb, 240, 240), cv2.COLOR_GRAY2RGB)
    # canny = cv2.Canny(template_rgb, 240, 240)

    # cropped_template_rgb = template_crop(template_rgb)
    # cropped_template_rgb = template_rgb

    # cropped_template_rgb = np.array(cropped_template_rgb)
    # cropped_template_gray = cv2.cvtColor(cropped_template_rgb, cv2.COLOR_RGB2GRAY)
    height, width = template_bgr.shape[:2]
    # fig = plt.figure(num="Template - Close the Window to Continue >>>")
    # plt.imshow(cropped_template_rgb)
    # plt.show()

    points_list = invariant_match_template(
        rgbimage=img_rgb,
        # rgbtemplate=cropped_template_rgb,
        maskmaker=formask,
        rgbtemplate=template_bgr,
        method="TM_CCOEFF_NORMED",
        # method="TM_CCORR",
        matched_thresh=0.5,
        rot_range=[-30, 30],
        rot_interval=3,
        scale_range=[90, 110],
        scale_interval=1,
        rm_redundant=True,
        minmax=True,
    )
    fig, ax = plt.subplots(1)
    plt.gcf().canvas.manager.set_window_title("Template Matching Results: Rectangles")
    ax.imshow(img_rgb)
    # points_list = points_list[:7]
    # reference_angle = points_list[0][1]

    # 각도 차이를 기준으로 정렬
    # points_list = sorted(points_list, key=lambda x: abs(x[1] - reference_angle))
    # points_list = points_list[:7]
    centers_list = []

    for point_info in points_list:
        point = point_info[0]
        angle = point_info[1]
        scale = point_info[2]
        print(
            f"matched point: {point}, angle: {angle}, scale: {scale}, score: {point_info[3]}"
        )
        centers_list.append([point, scale])
        plt.scatter(
            point[0] + (width / 2) * scale / 100,
            point[1] + (height / 2) * scale / 100,
            s=20,
            color="red",
        )
        plt.scatter(point[0], point[1], s=20, color="green")
        rectangle = patches.Rectangle(
            (point[0], point[1]),
            width * scale / 100,
            height * scale / 100,
            color="red",
            alpha=0.50,
            label="Matched box",
        )

        transform = (
            mpl.transforms.Affine2D().rotate_deg_around(
                point[0] + width / 2 * scale / 100,
                point[1] + height / 2 * scale / 100,
                angle,
            )
            + ax.transData
        )
        rectangle.set_transform(transform)
        ax.add_patch(rectangle)
        plt.legend(handles=[rectangle])
    # plt.grid(True)
    end_time = time.time()
    plt.show()
    fig2, ax2 = plt.subplots(1)
    plt.gcf().canvas.manager.set_window_title("Template Matching Results: Centers")
    ax2.imshow(img_rgb)
    for point_info in centers_list:
        point = point_info[0]
        scale = point_info[1]
        plt.scatter(
            point[0] + width / 2 * scale / 100,
            point[1] + height / 2 * scale / 100,
            s=20,
            color="red",
        )
    plt.show()
    # 걸린 시간 계산
    elapsed_time = end_time - start_time
    print(f"작업에 걸린 시간: {elapsed_time} 초")
