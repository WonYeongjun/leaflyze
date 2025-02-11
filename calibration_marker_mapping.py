# 저장된 이미지를 가져와서 원근법 보정함
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl
from calibration_marker_func import invariant_match_template  # ,template_crop
import ezdxf
from matplotlib.patches import Polygon
from shapely.geometry import Polygon


# 원근 보정 함수
def correct_perspective(image_path):
    # 이미지 로드
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Cannot load image at {image_path}")
        return

    # 이미지 크기 및 중심 계산
    h, w, _ = image.shape
    image_center = np.array([w / 2, h / 2])

    # 그레이스케일 변환
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 아루코 마커 검출
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    corners, ids, _ = detector.detectMarkers(gray)

    # 원하는 마커 ID 순서
    desired_ids = [12, 18, 27, 5]

    if ids is not None:
        marker_points = []

        # 검출된 마커에서 원하는 ID 찾기
        for marker_id in desired_ids:
            for i, detected_id in enumerate(ids):
                if detected_id[0] == marker_id:
                    marker_corners = corners[i][0]  # 4개의 꼭짓점 좌표
                    # 이미지 중심과 가장 가까운 꼭짓점 선택
                    closest_corner = min(
                        marker_corners, key=lambda pt: np.linalg.norm(pt - image_center)
                    )
                    marker_points.append(closest_corner)
                    break

        # 모든 마커가 감지된 경우 원근 변환 수행
        if len(marker_points) == 4:
            pts1 = np.array(marker_points, dtype="float32")

            # 변환 후 기준이 될 좌표
            width, height = 4200, 2970
            pts2 = np.array(
                [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
                dtype="float32",
            )

            # 호모그래피 행렬 계산
            matrix, _ = cv2.findHomography(pts1, pts2)

            # 원근 변환 적용
            dst = cv2.warpPerspective(image, matrix, (width, height))
        else:
            print("Not enough markers detected to calculate perspective.")
    else:
        print("No ArUco markers detected.")

    return dst


def real_marker_detector():
    test_image_path = "C:/Users/UserK/Desktop/raw/raw_img.jpg"  # 원본 이미지 경로

    img_bgr = correct_perspective(test_image_path)

    threshold = 130
    template_bgr = plt.imread("./image/marker_ideal.jpg")
    template_bgr = cv2.resize(
        template_bgr, (0, 0), fx=0.27, fy=0.27
    )  # 템플릿 사이즈 조절(초기 설정 필요)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
    template_gray = cv2.cvtColor(template_bgr, cv2.COLOR_RGB2GRAY)
    _, template_gray = cv2.threshold(template_gray, threshold, 255, cv2.THRESH_BINARY)
    height, width = template_gray.shape
    points_list = invariant_match_template(
        grayimage=img_gray,
        graytemplate=template_gray,
        method="TM_CCOEFF",
        matched_thresh=0.35,
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

    plt.show()

    indices = input("원하는 점의 인덱스 4개를 입력하세요 (쉼표로 구분): ")
    indices = list(map(int, indices.split(",")))

    if len(indices) != 4:
        print("4개의 인덱스를 입력해야 합니다.")
    else:
        selected_points = [real_point[i][0] for i in indices]
        matrix = np.array(selected_points)

        def sort_points(points):
            points = sorted(points, key=lambda x: x[0])  # x 좌표 기준으로 정렬
            left_points = points[:2]
            right_points = points[2:]
            left_points = sorted(
                left_points, key=lambda x: x[1]
            )  # y 좌표 기준으로 정렬
            right_points = sorted(right_points, key=lambda x: x[1])
            return [right_points[1], left_points[1], left_points[0], right_points[0]]

        sorted_matrix = sort_points(matrix)
        print("정렬된 좌표 행렬:")
        print(sorted_matrix)
        # sorted_matrix = np.array(sorted_matrix) / 10
    return sorted_matrix


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


def sim_marker_detector(image, template):
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
    plt.show()
    indices = [0, 1, 2, 3]

    if len(indices) != 4:
        print("4개의 인덱스를 입력해야 합니다.")
    else:
        selected_points = [real_point[i][0] for i in indices]
        matrix = np.array(selected_points)

        # 좌표를 오른쪽 위, 왼쪽 위, 왼쪽 아래, 오른쪽 아래 순서로 정렬
        def sort_points(points):
            points = sorted(points, key=lambda x: x[0])  # x 좌표 기준으로 정렬
            left_points = points[:2]
            right_points = points[2:]
            left_points = sorted(
                left_points, key=lambda x: x[1]
            )  # y 좌표 기준으로 정렬
            right_points = sorted(right_points, key=lambda x: x[1])
            return [right_points[1], left_points[1], left_points[0], right_points[0]]

        sorted_matrix = sort_points(matrix)

        print("정렬된 좌표 행렬:")
        print(sorted_matrix)
        # sorted_matrix = np.array(sorted_matrix) / 8
    return sorted_matrix


def translation_mapping(dst_pts, src_pts, center):
    src_pts = np.float32(src_pts)
    dst_pts = np.float32(dst_pts)
    M = cv2.getPerspectiveTransform(dst_pts, src_pts)

    def transform_point(x, y):
        """왜곡된 사각형 내의 좌표 (x, y)를 직사각형 필드 내의 대응 좌표로 변환"""
        input_point = np.array([[[x, y]]], dtype=np.float32)
        transformed_point = cv2.perspectiveTransform(input_point, M)
        return transformed_point[0][0]  # 변환된 (x, y) 좌표 반환

    # 테스트할 점 (왜곡된 사각형 내부의 점)
    test_points = []
    test_points.append(center)

    # 변환된 점 계산
    transformed_points = [transform_point(x, y) for x, y in test_points]

    # 플롯 생성
    plt.figure(figsize=(6, 4))
    ax = plt.gca()

    # 두 개의 사각형을 그림
    original_polygon = np.vstack([src_pts, src_pts[0]])  # 원본 직사각형
    distorted_polygon = np.vstack([dst_pts, dst_pts[0]])  # 왜곡된 사각형

    plt.plot(original_polygon[:, 0], original_polygon[:, 1], "bo-", label="real_marker")
    plt.plot(
        distorted_polygon[:, 0], distorted_polygon[:, 1], "ro-", label="DXF_marker"
    )

    # 점과 변환된 점을 표시
    for (dx, dy), (sx, sy) in zip(test_points, transformed_points):
        plt.plot(dx, dy, "ro")  # 왜곡된 사각형 내의 점
        plt.plot(sx, sy, "bo")  # 변환된 직사각형 필드 내의 점
        plt.arrow(
            dx, dy, sx - dx, sy - dy, head_width=2, head_length=3, fc="gray", ec="gray"
        )

    # 축 설정
    plt.xlim(
        min(min(src_pts[:, 0]), min(dst_pts[:, 0])) - 10,
        max(max(src_pts[:, 0]), max(dst_pts[:, 0])) + 10,
    )
    plt.ylim(
        min(min(src_pts[:, 1]), min(dst_pts[:, 1])) - 10,
        max(max(src_pts[:, 1]), max(dst_pts[:, 1])) + 10,
    )
    plt.gca().invert_yaxis()  # 이미지 좌표계처럼 y축 방향 반전

    plt.legend()
    plt.title("Translation_mapping")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.show()


def draw_centroid_circle(dxf_path):
    # DXF 파일 열기
    doc = ezdxf.readfile(dxf_path)
    modelspace = doc.modelspace()

    # SPLINE 객체를 찾기
    splines = [entity for entity in modelspace if entity.dxftype() == "SPLINE"]

    if not splines:
        print("SPLINE 객체가 없습니다.")
        return

    # 무게중심을 계산하고 동그라미 추가하기
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
                    centroidy = centroid.y * (-1)
                    print(f"다각형의 무게중심: X: {centroid.x}, Y: {centroid.y}")

            break  # 첫 번째 SPLINE에 대해서만 처리
    return (centroid.x, centroidy)


print(__name__)
if __name__ == "__main__":
    src_pts = real_marker_detector()
    dxf_path = "./image/비전재단테스트.dxf"
    dxf_file_png_path = "./image/output.png"
    dxf_to_png(dxf_path, dxf_file_png_path)

    image = cv2.imread(dxf_file_png_path)
    template = cv2.imread("./image/dxf_template.png")
    template_rgb = cv2.cvtColor(template, cv2.COLOR_BGR2RGB)

    dst_pts = sim_marker_detector(image, template)
    center = draw_centroid_circle("./image/비전재단테스트.dxf")
    translation_mapping(dst_pts, src_pts, center)
