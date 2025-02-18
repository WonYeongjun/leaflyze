import cv2
import numpy as np
import itertools


def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, -angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


def scale_image(image, percent):
    width = int(image.shape[1] * percent[0] / 100)
    height = int(image.shape[0] * percent[1] / 100)
    result = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    result = rotate_image(result, percent[2])
    return result


def invariant_match_template(
    grayimage,
    graytemplate,
    matched_thresh,
    rot_range,
    rot_interval,
    scale_range,
    scale_interval,
):
    start = scale_range[0]
    end = scale_range[1]
    scale_angle_combine = []
    for i in range(start, end + 1, scale_interval):
        for j in range(
            i - (scale_interval * 2), i + (scale_interval * 3), scale_interval
        ):
            for k in range(rot_range[0], rot_range[1] + 1, rot_interval):
                scale_angle_combine.append([i, j, k])

    height, width = graytemplate.shape
    all_points = []

    from concurrent.futures import ThreadPoolExecutor

    def process_template(scale_and_angle):
        processed_template = scale_image(graytemplate, scale_and_angle)
        result = cv2.matchTemplate(grayimage, processed_template, cv2.TM_CCOEFF_NORMED)
        ys, xs = np.where(result >= matched_thresh)
        matches = []
        matches = [
            (
                (int(x), int(y)),
                scale_and_angle[2],
                (scale_and_angle[0], scale_and_angle[1]),
                float(result[y, x]),
            )
            for y, x in zip(ys, xs)
        ]
        return matches

    with ThreadPoolExecutor() as executor:
        tasks = [
            executor.submit(process_template, scale_and_angle)
            for scale_and_angle in scale_angle_combine
        ]
        all_points = list(
            itertools.chain.from_iterable(
                future.result() for future in tasks if future.result()
            )
        )

    all_points = sorted(all_points, key=lambda x: -x[3])

    def filter_redundant_points(points_chunk):
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
                    break

            if all_visited_points_not_close:
                lone_points_list.append(point_info)
                visited_points_list.append(point)

        return lone_points_list

    chunk_size = max(1, len(all_points) // 4)
    chunks = [
        all_points[i : i + chunk_size] for i in range(0, len(all_points), chunk_size)
    ]

    with ThreadPoolExecutor() as executor2:
        results = executor2.map(filter_redundant_points, chunks)

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

    return points_list
