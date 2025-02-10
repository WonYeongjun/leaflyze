import ezdxf
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon


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
