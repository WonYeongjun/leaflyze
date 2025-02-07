import cv2
import numpy as np

box_points = []
button_down = False


def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, -angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


def scale_image(image, percent, maxwh):
    width = int(image.shape[1] * percent[0] / 100)
    height = int(image.shape[0] * percent[1] / 100)
    result = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    result = rotate_image(result, percent[2])

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
    start = scale_range[0]
    end = scale_range[1]
    scale_tuple = []
    for i in range(start, end + 1, scale_interval):
        for j in range(
            i - (scale_interval * 2), i + (scale_interval * 3), scale_interval
        ):
            for k in range(rot_range[0], rot_range[1] + 1, rot_interval):
                scale_tuple.append([i, j, k])
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

    print("circle")
    from concurrent.futures import ThreadPoolExecutor

    def process_template(next_scale, n=5):  # 상위 n개 선택 가능
        rotated_template, actual_scale = scale_image(
            template_gray, next_scale, image_maxwh
        )

        matched_points = cv2.matchTemplate(img_gray, rotated_template, cv2.TM_CCOEFF)

        # 상위 n개 찾기
        flat_indices = np.argsort(matched_points.ravel())[::-1]  # 높은 값부터 정렬
        top_n_indices = flat_indices[:n]
        top_n_locs = [
            np.unravel_index(idx, matched_points.shape) for idx in top_n_indices
        ]
        top_n_vals = [matched_points[loc] for loc in top_n_locs]

        # 임계값 이상만 반환
        results = [
            (loc, actual_scale[2], actual_scale[0], actual_scale[1], val)
            for loc, val in zip(top_n_locs, top_n_vals)
            if val >= matched_thresh
        ]

        return results  # 리스트로 반환

    with ThreadPoolExecutor() as executor:
        tasks = [
            executor.submit(process_template, next_scale, 5)  # 상위 5개 가져오기
            for next_scale in scale_tuple
        ]

        # 모든 결과 모으기
        all_points = [result for future in tasks for result in future.result()]
    all_points = sorted(all_points, key=lambda x: -x[4])
    if rm_redundant == True:
        lone_points_list = []
        visited_points_list = []
        for point_info in all_points:
            point = point_info[0]
            scalex = point_info[2]
            scaley = point_info[3]
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
