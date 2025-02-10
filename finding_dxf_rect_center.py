import ezdxf
import math


def calculate_area(rectangle):
    """주어진 직사각형 객체의 넓이를 계산하는 함수"""
    points = rectangle.get_points()

    if len(points) != 4:
        return 0  # 직사각형이 아닌 경우

    # 각 점에서 (x, y) 값 추출
    x1, y1 = points[0][0], points[0][1]
    x2, y2 = points[1][0], points[1][1]
    x3, y3 = points[2][0], points[2][1]
    x4, y4 = points[3][0], points[3][1]
    centerx = (x1 + x2 + x3 + x4) / 4
    centery = (y1 + y2 + y3 + y4) / 4

    # 거리 계산: 두 점 사이의 유클리드 거리
    length = math.dist((x1, y1), (x2, y2))
    width = math.dist((x2, y2), (x3, y3))

    # 직사각형의 넓이: 길이 * 너비
    area = length * width
    return area, (centerx, centery)


def remove_largest_rectangle(dxf_path):
    doc = ezdxf.readfile(dxf_path)
    modelspace = doc.modelspace()

    # LWPOLYLINE 객체 중 직사각형을 찾아 넓이를 계산
    rectangles = [
        entity
        for entity in modelspace
        if entity.dxftype() == "LWPOLYLINE" and len(entity.get_points()) == 4
    ]

    if not rectangles:
        print("직사각형 LWPOLYLINE 객체가 없습니다.")
        return

    # 가장 넓이가 큰 직사각형을 찾기
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
        radius = 3.0  # 반지름 설정 (원 크기 조정 가능)
        modelspace.add_circle(
            center=largest_center, radius=radius
        )  # 다각형 내부 영역의 중심
        doc.saveas("updated_" + dxf_path)  # 변경된 파일 저장
    else:
        print("직사각형을 찾을 수 없습니다.")


# DXF 파일 경로 지정
dxf_file = "심포유강아지_생산_얼굴x12.dxf"  # 예시 DXF 파일 경로
remove_largest_rectangle(dxf_file)
