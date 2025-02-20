import cv2
import numpy as np
import matplotlib.pyplot as plt

box_points = []
button_down = False


def remove_background(image, bg_color=(255, 255, 255)):
    """
    배경을 제거하고 마커만 남기는 함수.
    image: 입력 이미지 (BGR 형식)
    bg_color: 배경 색상 (기본값: 흰색)
    """
    # BGR에서 RGBA로 변환 (알파 채널 추가)
    if image.shape[-1] == 3:  # BGR 포맷
        image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)

    # 배경 마스크 생성
    lower_bound = np.array([bg_color[0] - 10, bg_color[1] - 10, bg_color[2] - 10, 0])
    upper_bound = np.array([bg_color[0] + 10, bg_color[1] + 10, bg_color[2] + 10, 255])
    mask = cv2.inRange(image, lower_bound, upper_bound)

    # 배경 투명화
    image[mask == 255] = (0, 0, 0, 0)  # 배경을 투명하게 설정

    return image


def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, -angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


def scale_image(image, percent, maxwh):
    width = int(image.shape[1] * percent[0] / 100)
    height = int(image.shape[0] * percent[0] / 100)
    result = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    result = rotate_image(result, percent[1])
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
    maskmaker,
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
        for j in range(rot_range[0], rot_range[1] + 1, rot_interval):
            scale_tuple.append([i, j])
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
    # img_gray = cv2.Canny(img_gray, 240, 240)
    # template_gray = cv2.Canny(template_gray, 240, 240)
    image_maxwh = img_gray.shape
    height, width = template_gray.shape
    all_points = []

    print("동그라미")
    from concurrent.futures import ThreadPoolExecutor

    print(template_gray.shape)

    def process_template(next_angle):
        rotated_template, actual_scale = scale_image(
            template_gray, next_angle, image_maxwh
        )
        rotated_maskmaker, actual_scale = scale_image(
            maskmaker, next_angle, image_maxwh
        )
        matched_points = cv2.matchTemplate(
            img_gray, rotated_template, cv2.TM_CCOEFF_NORMED, mask=rotated_maskmaker
        )
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(matched_points)
        if max_val >= matched_thresh:
            return [max_loc, actual_scale[1], actual_scale[0], max_val]
        return None

    with ThreadPoolExecutor(max_workers=6) as executor:
        tasks = [
            executor.submit(process_template, next_angle) for next_angle in scale_tuple
        ]
        all_points = [result for future in tasks if (result := future.result())]
        print(result)
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

    if rm_redundant == True:
        lone_points_list = []
        visited_points_list = []
        for point_info in all_points:
            point = point_info[0]
            scale = point_info[2]
            all_visited_points_not_close = True
            if len(visited_points_list) != 0:
                for visited_point in visited_points_list:
                    if (abs(visited_point[0] - point[0]) < (width * scale / 100)) and (
                        abs(visited_point[1] - point[1]) < (height * scale / 100)
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
