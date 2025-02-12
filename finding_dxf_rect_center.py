import ezdxf
import math


def calculate_area(rectangle):
    points = rectangle.get_points()

    if len(points) != 4:
        return 0

    x1, y1 = points[0][0], points[0][1]
    x2, y2 = points[1][0], points[1][1]
    x3, y3 = points[2][0], points[2][1]
    x4, y4 = points[3][0], points[3][1]
    centerx = (x1 + x2 + x3 + x4) / 4
    centery = (y1 + y2 + y3 + y4) / 4

    length = math.dist((x1, y1), (x2, y2))
    width = math.dist((x2, y2), (x3, y3))

    area = length * width
    return area, (centerx, centery)


def remove_largest_rectangle(dxf_path):
    doc = ezdxf.readfile(dxf_path)
    modelspace = doc.modelspace()

    rectangles = [
        entity
        for entity in modelspace
        if entity.dxftype() == "LWPOLYLINE" and len(entity.get_points()) == 4
    ]

    if not rectangles:
        print("직사각형 LWPOLYLINE 객체가 없습니다.")
        return

    largest_rectangle = None
    largest_area = 0
    largest_center = (0, 0)
    for rectangle in rectangles:
        area, center = calculate_area(rectangle)
        if area > largest_area:
            largest_center = center
            largest_area = area
            largest_rectangle = rectangle

    if largest_rectangle:
        print(f"가장 넓이가 큰 직사각형을 삭제합니다. 넓이: {largest_area}")
        print(f"그 직사각형의 중심은 {largest_center}")
        radius = 3.0
        modelspace.add_circle(center=largest_center, radius=radius)
        doc.saveas("./image/updated_심포유강아지_생산_얼굴x12.dxf")
    else:
        print("직사각형을 찾을 수 없습니다.")


dxf_file = "./image/심포유강아지_생산_얼굴x12.dxf"
remove_largest_rectangle(dxf_file)
