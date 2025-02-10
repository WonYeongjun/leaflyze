import ezdxf
from matplotlib.patches import Polygon
import cv2
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
import matplotlib as mpl
import time


def dxf_to_png(dxf_file, png_file):
    # DXF 파일 읽기
    doc = ezdxf.readfile(dxf_file)

    # 새 이미지 생성 (빈 그림)
    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    ax.set_axis_off()  # 축 숨기기

    # DXF의 모든 엔티티를 PNG로 그리기
    for entity in doc.modelspace():
        if entity.dxftype() == "LINE":
            # 선 그리기
            ax.plot(
                [entity.dxf.start.x, entity.dxf.end.x],
                [entity.dxf.start.y, entity.dxf.end.y],
                "b-",
            )
        elif entity.dxftype() == "CIRCLE":
            # 원 그리기
            circle = plt.Circle(
                (entity.dxf.center.x, entity.dxf.center.y),
                entity.dxf.radius,
                fill=False,
                edgecolor="b",
            )
            ax.add_artist(circle)
        elif entity.dxftype() == "LWPOLYLINE":
            # LWPOLYLINE 그리기
            points = [(vertex[0], vertex[1]) for vertex in entity.vertices()]
            ax.plot(*zip(*points), "b-")
        elif entity.dxftype() == "SPLINE":
            # SPLINE 그리기 (이때는 각 점을 이어서 그리는 방법으로 처리)
            points = np.array(
                [point for point in entity.control_points]
            )  # control_points로 수정
            ax.plot(points[:, 0], points[:, 1], "r-")
        elif entity.dxftype() == "HATCH":
            # HATCH 그리기 (채우기 다각형 처리)
            for path in entity.paths:
                if hasattr(path, "vertices"):  # path가 EdgePath일 경우
                    points = [(point.x, point.y) for point in path.vertices]
                    polygon = Polygon(
                        points, closed=True, edgecolor="blue", facecolor="lightblue"
                    )
                    ax.add_patch(polygon)

    # 이미지 저장
    plt.savefig(png_file, format="png", bbox_inches="tight", pad_inches=0.1)
    plt.close()


# 예시 사용법
dxf_to_png("비전재단테스트.dxf", "output.png")

image = cv2.imread("./output.png")
template = cv2.imread("C:/Users/UserK/Desktop/dxf_template.png")
template_rgb = cv2.cvtColor(template, cv2.COLOR_BGR2RGB)

start_time = time.time()

if __name__ == "__main__":
    img_bgr = image

    template_bgr = template
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
    template_gray = cv2.cvtColor(template_bgr, cv2.COLOR_RGB2GRAY)
    height, width = template_gray.shape
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

    lone_points_list = []
    visited_points_list = []
    for point_info in all_points:
        print(point_info)
        point = point_info[0]
        scalex = point_info[2][0]
        scaley = point_info[2][1]
        all_visited_points_not_close = True
        if len(visited_points_list) != 0:
            for visited_point in visited_points_list:
                if (abs(visited_point[0] - point[0]) < (width * scalex / 100)) and (
                    abs(visited_point[1] - point[1]) < (height * scaley / 100)
                ):
                    all_visited_points_not_close = False
            if all_visited_points_not_close == True:
                lone_points_list.append(point_info)
                visited_points_list.append(point)
        else:
            lone_points_list.append(point_info)
            visited_points_list.append(point)
    points_list = lone_points_list

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
