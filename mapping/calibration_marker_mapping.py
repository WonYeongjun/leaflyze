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


def correct_perspective(image_path):

    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Cannot load image at {image_path}")
        return

    h, w, _ = image.shape
    image_center = np.array([w / 2, h / 2])

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    corners, ids, _ = detector.detectMarkers(gray)

    # 원하는 마커 ID 순서
    desired_ids = [12, 18, 27, 5]

    if ids is not None:
        marker_points = []

        for marker_id in desired_ids:
            for i, detected_id in enumerate(ids):
                if detected_id[0] == marker_id:
                    marker_corners = corners[i][0]

                    closest_corner = min(
                        marker_corners, key=lambda pt: np.linalg.norm(pt - image_center)
                    )
                    marker_points.append(closest_corner)
                    break

        if len(marker_points) == 4:
            pts1 = np.array(marker_points, dtype="float32")

            width, height = 4200, 2970
            pts2 = np.array(
                [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
                dtype="float32",
            )

            matrix, _ = cv2.findHomography(pts1, pts2)

            dst = cv2.warpPerspective(image, matrix, (width, height))
        else:
            print("Not enough markers detected to calculate perspective.")
    else:
        print("No ArUco markers detected.")

    return dst


def real_marker_detector():
    test_image_path = "C:/Users/UserK/Desktop/raw/raw_img.jpg"

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
        matched_thresh=0.4,
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

    indices = input("원하는 점의 인덱스 4개를 입력하세요 (쉼표로 구분): ")
    indices = list(map(int, indices.split(",")))

    if len(indices) != 4:
        print("4개의 인덱스를 입력해야 합니다.")
    else:
        selected_points = [real_point[i][0] for i in indices]
        matrix = np.array(selected_points)

        def sort_points(points):
            points = sorted(points, key=lambda x: x[0])
            left_points = points[:2]
            right_points = points[2:]
            left_points = sorted(left_points, key=lambda x: x[1])
            right_points = sorted(right_points, key=lambda x: x[1])
            return [right_points[1], left_points[1], left_points[0], right_points[0]]

        sorted_matrix = sort_points(matrix)
    return sorted_matrix


def dxf_to_png(dxf_file, png_file):
    doc = ezdxf.readfile(dxf_file)
    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    ax.set_axis_off()
    for entity in doc.modelspace():
        if entity.dxftype() == "LINE":
            ax.plot(
                [entity.dxf.start.x, entity.dxf.end.x],
                [entity.dxf.start.y, entity.dxf.end.y],
                "b-",
            )
        elif entity.dxftype() == "CIRCLE":
            circle = plt.Circle(
                (entity.dxf.center.x, entity.dxf.center.y),
                entity.dxf.radius,
                fill=False,
                edgecolor="b",
            )
            ax.add_artist(circle)
        elif entity.dxftype() == "LWPOLYLINE":
            points = [(vertex[0], vertex[1]) for vertex in entity.vertices()]
            ax.plot(*zip(*points), "b-")
        elif entity.dxftype() == "SPLINE":
            points = np.array([point for point in entity.control_points])
            ax.plot(points[:, 0], points[:, 1], "r-")
        elif entity.dxftype() == "HATCH":
            for path in entity.paths:
                if hasattr(path, "vertices"):
                    points = [(point.x, point.y) for point in path.vertices]
                    polygon = Polygon(
                        points, closed=True, edgecolor="blue", facecolor="lightblue"
                    )
                    ax.add_patch(polygon)
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

    ys, xs = np.where(result >= 0.65)
    matches = []
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

        def sort_points(points):
            points = sorted(points, key=lambda x: x[0])
            left_points = points[:2]
            right_points = points[2:]
            left_points = sorted(left_points, key=lambda x: x[1])
            right_points = sorted(right_points, key=lambda x: x[1])
            return [right_points[1], left_points[1], left_points[0], right_points[0]]

        sorted_matrix = sort_points(matrix)
    return sorted_matrix


def translation_mapping(dst_pts, src_pts, center):
    src_pts = np.float32(src_pts)
    dst_pts = np.float32(dst_pts)
    M = cv2.getPerspectiveTransform(dst_pts, src_pts)

    def transform_point(x, y):
        input_point = np.array([[[x, y]]], dtype=np.float32)
        transformed_point = cv2.perspectiveTransform(input_point, M)
        return transformed_point[0][0]

    test_points = []
    test_points.append(center)

    transformed_points = [transform_point(x, y) for x, y in test_points]

    plt.figure(figsize=(6, 4))
    ax = plt.gca()

    original_polygon = np.vstack([src_pts, src_pts[0]])
    distorted_polygon = np.vstack([dst_pts, dst_pts[0]])

    plt.plot(original_polygon[:, 0], original_polygon[:, 1], "bo-", label="real_marker")
    plt.plot(
        distorted_polygon[:, 0], distorted_polygon[:, 1], "ro-", label="DXF_marker"
    )

    for (dx, dy), (sx, sy) in zip(test_points, transformed_points):
        plt.plot(dx, dy, "ro")
        plt.plot(sx, sy, "bo")
        plt.arrow(
            dx, dy, sx - dx, sy - dy, head_width=2, head_length=3, fc="gray", ec="gray"
        )

    plt.xlim(
        min(min(src_pts[:, 0]), min(dst_pts[:, 0])) - 10,
        max(max(src_pts[:, 0]), max(dst_pts[:, 0])) + 10,
    )
    plt.ylim(
        min(min(src_pts[:, 1]), min(dst_pts[:, 1])) - 10,
        max(max(src_pts[:, 1]), max(dst_pts[:, 1])) + 10,
    )
    plt.gca().invert_yaxis()

    plt.legend()
    plt.title("Translation_mapping")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.show()


def draw_centroid_circle(dxf_path):

    doc = ezdxf.readfile(dxf_path)
    modelspace = doc.modelspace()

    splines = [entity for entity in modelspace if entity.dxftype() == "SPLINE"]

    if not splines:
        print("SPLINE 객체가 없습니다.")
        return

    for spline in splines:
        if spline.closed:

            control_points = spline.control_points
            if control_points:
                polygon = Polygon([(point[0], point[1]) for point in control_points])

                if polygon.is_valid:

                    centroid = polygon.centroid
                    centroidy = centroid.y * (-1)
                    print(f"다각형의 무게중심: X: {centroid.x}, Y: {centroid.y}")

            break
    return (centroid.x, centroidy)


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
