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


def scale_image(image, percent, maxwh):
    max_width = maxwh[1]
    max_height = maxwh[0]
    max_percent_width = max_width / image.shape[1] * 100
    max_percent_height = max_height / image.shape[0] * 100
    max_percent = min(max_percent_width, max_percent_height)
    percent = min(percent, max_percent)
    width = int(image.shape[1] * percent / 100)
    height = int(image.shape[0] * percent / 100)
    result = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    return result, percent


# def click_and_crop(event, x, y, flags, param):
#     global box_points, button_down
#     if (button_down == False) and (event == cv2.EVENT_LBUTTONDOWN):
#         button_down = True
#         box_points = [(x, y)]
#     elif (button_down == True) and (event == cv2.EVENT_MOUSEMOVE):
#         image_copy = param.copy()
#         point = (x, y)
#         cv2.rectangle(image_copy, box_points[0], point, (0, 255, 0), 2)
#         cv2.imshow("Template Cropper - Press C to Crop", image_copy)
#     elif event == cv2.EVENT_LBUTTONUP:
#         button_down = False
#         box_points.append((x, y))
#         cv2.rectangle(param, box_points[0], box_points[1], (0, 255, 0), 2)
#         cv2.imshow("Template Cropper - Press C to Crop", param)


# # GUI template cropping tool
# def template_crop(image):
#     clone = image.copy()
#     cv2.namedWindow("Template Cropper - Press C to Crop")
#     param = image
#     cv2.setMouseCallback("Template Cropper - Press C to Crop", click_and_crop, param)
#     while True:
#         cv2.imshow("Template Cropper - Press C to Crop", image)
#         key = cv2.waitKey(1)
#         if key == ord("c"):
#             cv2.destroyAllWindows()
#             break
#     if len(box_points) == 2:
#         cropped_region = clone[
#             box_points[0][1] : box_points[1][1], box_points[0][0] : box_points[1][0]
#         ]
#     return cropped_region


def invariant_match_template(
    rgbimage,
    rgbtemplate,
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
    """
    rgbimage: RGB image where the search is running.
    rgbtemplate: RGB searched template. It must be not greater than the source image and have the same data type.
    method: [String] Parameter specifying the comparison method
    matched_thresh: [Float] Setting threshold of matched results(0~1).
    rot_range: [Integer] Array of range of rotation angle in degrees. Example: [0,360]
    rot_interval: [Integer] Interval of traversing the range of rotation angle in degrees.
    scale_range: [Integer] Array of range of scaling in percentage. Example: [50,200]
    scale_interval: [Integer] Interval of traversing the range of scaling in percentage.
    rm_redundant: [Boolean] Option for removing redundant matched results based on the width and height of the template.
    minmax:[Boolean] Option for finding points with minimum/maximum value.
    rgbdiff_thresh: [Float] Setting threshold of average RGB difference between template and source image. Default: +inf threshold (no rgbdiff)

    Returns: List of satisfied matched points in format [[point.x, point.y], angle, scale].
    """
    img_gray = cv2.cvtColor(rgbimage, cv2.COLOR_RGB2GRAY)
    template_gray = cv2.cvtColor(rgbtemplate, cv2.COLOR_RGB2GRAY)
    image_maxwh = img_gray.shape
    height, width = template_gray.shape
    all_points = []

    print("동그라미")
    from concurrent.futures import ThreadPoolExecutor

    print(template_gray.shape)

    def process_template(next_angle, next_scale):
        # 이미지 스케일링 및 회전
        scaled_template_gray, actual_scale = scale_image(
            template_gray, next_scale, image_maxwh
        )
        if next_angle == 0:
            rotated_template = scaled_template_gray
        else:
            rotated_template = rotate_image(scaled_template_gray, next_angle)

        # 템플릿 매칭 (결과는 2차원 numpy 배열)
        result = cv2.matchTemplate(img_gray, rotated_template, cv2.TM_CCOEFF_NORMED)

        # numpy 벡터화 연산을 사용해서 조건을 만족하는 인덱스들을 한 번에 추출
        print(1)
        ys, xs = np.where(result >= matched_thresh)
        matches = []
        # 추출된 좌표와 점수를 list comprehension으로 matches 리스트에 저장
        matches = [
            ((int(x), int(y)), next_angle, next_scale, float(result[y, x]))
            for y, x in zip(ys, xs)
        ]
        print(matches)
        return matches

    # ThreadPoolExecutor를 이용한 병렬 처리
    with ThreadPoolExecutor() as executor:
        tasks = [
            executor.submit(process_template, next_angle, next_scale)
            for next_angle in range(rot_range[0], rot_range[1], rot_interval)
            for next_scale in range(scale_range[0], scale_range[1], scale_interval)
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

    def filter_redundant_points(points_chunk):
        # """중복된 포인트를 제거하는 함수 (각 스레드에서 독립적으로 실행)"""
        lone_points_list = []
        visited_points_list = []

        for point_info in points_chunk:
            point = point_info[0]
            scale = point_info[2]
            all_visited_points_not_close = True

            for visited_point in visited_points_list:
                if (abs(visited_point[0] - point[0]) < (width * scale / 100)) and (
                    abs(visited_point[1] - point[1]) < (height * scale / 100)
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
                    if (abs(visited_point[0] - point[0]) < (width * scale / 100)) and (
                        abs(visited_point[1] - point[1]) < (height * scale / 100)
                    ):
                        all_visited_points_not_close = False
                        break

                if all_visited_points_not_close:
                    final_lone_points.append(point_info)
                    final_visited_points.append(point)

        points_list = final_lone_points
    else:
        points_list = all_points
    if rgbdiff_thresh != float("inf"):
        print(">>>RGBDiff Filtering>>>")
        color_filtered_list = []
        template_channels = cv2.mean(rgbtemplate)
        template_channels = np.array(
            [template_channels[0], template_channels[1], template_channels[2]]
        )
        for point_info in points_list:
            point = point_info[0]
            cropped_img = rgbimage[
                point[1] : point[1] + height, point[0] : point[0] + width
            ]
            cropped_channels = cv2.mean(cropped_img)
            cropped_channels = np.array(
                [cropped_channels[0], cropped_channels[1], cropped_channels[2]]
            )
            diff_observation = cropped_channels - template_channels
            total_diff = np.sum(np.absolute(diff_observation))
            print(total_diff)
            if total_diff < rgbdiff_thresh:
                color_filtered_list.append(
                    [point_info[0], point_info[1], point_info[2], point_info[3]]
                )
        return color_filtered_list
    else:
        return points_list
