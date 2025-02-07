import cv2
import numpy as np
import itertools

box_points = []
button_down = False


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
    method,
    matched_thresh,
    rot_range,
    rot_interval,
    scale_range,
    scale_interval,
    rm_redundant,
    minmax,
    rgbdiff_thresh=float("inf"),
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

    img_gray = grayimage
    template_gray = graytemplate
    image_maxwh = img_gray.shape
    height, width = template_gray.shape
    all_points = []

    print("템플릿 매칭 시작")
    from concurrent.futures import ThreadPoolExecutor

    def process_template(scale_and_angle):
        # 이미지 스케일링 및 회전
        processed_template = scale_image(template_gray, scale_and_angle)
        # 템플릿 매칭 (결과는 2차원 numpy 배열)
        result = cv2.matchTemplate(img_gray, processed_template, cv2.TM_CCOEFF_NORMED)

        # numpy 벡터화 연산을 사용해서 조건을 만족하는 인덱스들을 한 번에 추출
        print(1)
        ys, xs = np.where(result >= matched_thresh)
        matches = []
        # 추출된 좌표와 점수를 list comprehension으로 matches 리스트에 저장
        matches = [
            (
                (int(x), int(y)),
                scale_and_angle[2],
                (scale_and_angle[0], scale_and_angle[1]),
                float(result[y, x]),
            )
            for y, x in zip(ys, xs)
        ]
        print(matches)
        return matches

    # ThreadPoolExecutor를 이용한 병렬 처리
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
        print(len(all_points))
    if method == "TM_CCOEFF":
        all_points = sorted(all_points, key=lambda x: -x[3])
    elif method == "TM_CCOEFF_NORMED":
        all_points = sorted(all_points, key=lambda x: -x[3])
    elif method == "TM_CCORR":
        all_points = sorted(all_points, key=lambda x: -x[3])
    elif method == "TM_CCORR_NORMED":
        all_points = sorted(all_points, key=lambda x: -x[3])
    elif method == "TM_SQDIFF":
        all_points = sorted(all_points, key=lambda x: x[3])
    elif method == "TM_SQDIFF_NORMED":
        all_points = sorted(all_points, key=lambda x: x[3])
    print("템플릿 매칭 완료")
    print("중복 지점 제거 시작")

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

    if rm_redundant:
        chunk_size = max(1, len(all_points) // 4)  # 스레드당 할당할 포인트 개수
        chunks = [
            all_points[i : i + chunk_size]
            for i in range(0, len(all_points), chunk_size)
        ]

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
                    if (
                        abs(visited_point[0] - point[0]) < (width * scale[0] / 100)
                    ) and (
                        abs(visited_point[1] - point[1]) < (height * scale[1] / 100)
                    ):
                        all_visited_points_not_close = False
                        break

                if all_visited_points_not_close:
                    final_lone_points.append(point_info)
                    final_visited_points.append(point)

        points_list = final_lone_points
    else:
        points_list = all_points
    print("중복 지점 제거 완료")

    return points_list
