import ezdxf
from shapely.geometry import Polygon


def draw_centroid_circle(dxf_path, output_path):
    # DXF 파일 열기
    doc = ezdxf.readfile(dxf_path)
    modelspace = doc.modelspace()

    # SPLINE 객체를 찾기
    splines = [entity for entity in modelspace if entity.dxftype() == "SPLINE"]

    if not splines:
        print("SPLINE 객체가 없습니다.")
        return

    for spline in splines:
        if spline.closed:
            # 제어점 (Control Points) 출력 및 평균 계산
            control_points = spline.control_points
            if control_points:
                # 제어점들의 평균 계산 (기하학적 중심)
                avg_x = sum(point[0] for point in control_points) / len(control_points)
                avg_y = sum(point[1] for point in control_points) / len(control_points)

                # 제어점들의 기하학적 중심 (평균)
                print(f"기하학적 중심 (X, Y): ({avg_x}, {avg_y})")

                # 제어점들로 다각형 생성
                polygon = Polygon([(point[0], point[1]) for point in control_points])

                if polygon.is_valid:
                    # 다각형의 무게중심 계산
                    centroid = polygon.centroid
                    print(f"다각형의 무게중심: X: {centroid.x}, Y: {centroid.y}")

                    # 원을 그리기 위한 좌표와 반지름 설정
                    radius = 1.0  # 반지름 설정 (원 크기 조정 가능)
                    modelspace.add_circle(
                        center=(centroid.x, centroid.y), radius=radius
                    )  # 다각형 내부 영역의 중심
                    # modelspace.add_circle(
                    #     center=(avg_x, avg_y), radius=radius
                    # )  # 윤곽선에 있는 점들의 중심

            break  # 첫 번째 SPLINE에 대해서만 처리

    # 수정된 DXF 파일 저장
    doc.saveas(output_path)
    print(f"동그라미가 그려진 DXF 파일을 저장했습니다: {output_path}")


dxf_file = "./image/비전재단테스트.dxf"  # 원본 DXF 파일 경로
output_file = "./image/output_with_circle.dxf"  # 수정된 DXF 파일 경로
draw_centroid_circle(dxf_file, output_file)
