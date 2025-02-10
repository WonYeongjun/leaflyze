import cairosvg
import cv2
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
import matplotlib as mpl
import time

start_time = time.time()
# SVG 파일 경로와 출력할 PNG 파일 경로 지정
input_svg = "C:\\Users\\UserK\\Desktop\\image\\example.svg"
output_png = "C:\\Users\\UserK\\Desktop\\image\\output.png"

# SVG 파일을 PNG로 변환
cairosvg.svg2png(url=input_svg, write_to=output_png)

# PNG 파일을 OpenCV로 읽어오기 (알파 채널 포함)
image = cv2.imread(output_png, cv2.IMREAD_UNCHANGED)

# 알파 채널이 있을 경우, 투명한 부분을 흰색으로 바꾸기
if image.shape[2] == 4:  # 알파 채널이 있는 이미지
    # 알파 채널을 분리
    alpha_channel = image[:, :, 3]

    # 투명한 부분을 흰색으로 변경
    image[alpha_channel == 0] = [
        255,
        255,
        255,
        255,
    ]  # RGB를 흰색(255)으로 설정, 알파는 255로 유지

# BGR을 RGB로 변환 (matplotlib은 RGB 색 공간을 사용)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 이미지를 matplotlib으로 표시
template = cv2.imread("C:/Users/UserK/Desktop/image/sim_template.png")
if template.shape[2] == 4:  # 알파 채널이 있는 이미지
    # 알파 채널을 분리
    alpha_channel = template[:, :, 3]

    # 투명한 부분을 흰색으로 변경
    template[alpha_channel == 0] = [
        255,
        255,
        255,
        255,
    ]  # RGB를 흰색(255)으로 설정, 알파는 255로 유지

# BGR을 RGB로 변환 (matplotlib은 RGB 색 공간을 사용)
template_rgb = cv2.cvtColor(template, cv2.COLOR_BGR2RGB)
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
# ax1.imshow(image_rgb, cmap="gray")
# ax1.set_title("Original Grayscale Image")
# ax2.imshow(template_rgb, cmap="gray")
# ax2.set_title("Processed Grayscale Image")
# plt.show()

start_time = time.time()

if __name__ == "__main__":
    img_bgr = image

    template_bgr = template
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    im = img_gray

    img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
    template_gray = cv2.cvtColor(template_bgr, cv2.COLOR_RGB2GRAY)
    height, width = template_gray.shape

    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    # ax1.imshow(im, cmap="gray")
    # ax1.set_title("Original Grayscale Image")
    # ax2.imshow(img_gray, cmap="gray")
    # ax2.set_title("Processed Grayscale Image")
    # plt.show()
    result = cv2.matchTemplate(img_gray, template_gray, cv2.TM_CCOEFF_NORMED)

    # numpy 벡터화 연산을 사용해서 조건을 만족하는 인덱스들을 한 번에 추출
    ys, xs = np.where(result >= 0.65)
    matches = []
    # 추출된 좌표와 점수를 list comprehension으로 matches 리스트에 저장
    matches = [
        (
            (int(x), int(y)),
            0,
            (100, 100),
            float(result[y, x]),
        )
        for y, x in zip(ys, xs)
    ]
all_points = sorted(matches, key=lambda x: -x[3])
from concurrent.futures import ThreadPoolExecutor

chunk_size = max(1, len(all_points) // 4)  # 스레드당 할당할 포인트 개수
chunks = [all_points[i : i + chunk_size] for i in range(0, len(all_points), chunk_size)]


def filter_redundant_points(points_chunk):
    # """중복된 포인트를 제거하는 함수 (각 스레드에서 독립적으로 실행)"""
    lone_points_list = []
    visited_points_list = []

    for point_info in points_chunk:
        point = point_info[0]
        scale = point_info[2]
        all_visited_points_not_close = True

        for visited_point in visited_points_list:
            if (abs(visited_point[0] - point[0]) < (width * scale[0] / 100)) and (
                abs(visited_point[1] - point[1]) < (height * scale[1] / 100)
            ):
                all_visited_points_not_close = False
                break  # 이미 가까운 점이 있으면 더 체크할 필요 없음

        if all_visited_points_not_close:
            lone_points_list.append(point_info)
            visited_points_list.append(point)

    return lone_points_list


with ThreadPoolExecutor() as executor2:
    results = executor2.map(filter_redundant_points, chunks)

# 여러 스레드 결과를 합치면서 최종 중복 제거
final_lone_points = []
final_visited_points = []

for lone_points in results:
    for point_info in lone_points:
        point = point_info[0]
        scale = point_info[2]
        all_visited_points_not_close = True

        for visited_point in final_visited_points:
            if (abs(visited_point[0] - point[0]) < (width * scale[0] / 100)) and (
                abs(visited_point[1] - point[1]) < (height * scale[1] / 100)
            ):
                all_visited_points_not_close = False
                break

        if all_visited_points_not_close:
            final_lone_points.append(point_info)
            final_visited_points.append(point)

points_list = final_lone_points
print("중복 지점 제거 완료")
print(points_list)
fig, ax = plt.subplots(1)
plt.gcf().canvas.manager.set_window_title("Template Matching Results: Rectangles")
ax.imshow(img_rgb)
print(len(points_list))
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
end_time = time.time()
elapsed_time = end_time - start_time
print(f"작업에 걸린 시간: {elapsed_time} 초")
plt.show()
