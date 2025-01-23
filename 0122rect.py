import cv2
import math
from sklearn.cluster import DBSCAN

filename = "./image/box"


# 이미지 4분할
def split_image(image, point, angle):
    x, y = point
    h, w = image.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((x, y), angle, 1)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h))
    top_left = rotated_image[0:y, 0:x]
    top_right = rotated_image[0:y, x:]
    bottom_left = rotated_image[y:, 0:x]
    bottom_right = rotated_image[y:, x:]
    return top_left, top_right, bottom_left, bottom_right


import cv2
import numpy as np

count = 0
img = cv2.imread(f"{filename}.jpg", cv2.IMREAD_COLOR)
size = img.shape[:2]
print(size)
img = cv2.resize(img, (800, 600))  # 이미지 크기 조정-카메라 설치 후 하이퍼파라미터 조정


def remove_overlapping_contours(contours, img_size=(500, 500)):
    # 컨투어 리스트 정렬 (면적 기준 내림차순)
    contours = sorted(contours, key=lambda cnt: cv2.contourArea(cnt), reverse=True)
    remaining_contours = []
    mask = np.zeros(img_size, dtype=np.uint8)

    for cnt in contours:
        # 현재 컨투어를 그린 마스크 생성
        current_mask = np.zeros(img_size, dtype=np.uint8)
        cv2.drawContours(current_mask, [cnt], -1, 255, -1)
        overlap = cv2.bitwise_and(mask, current_mask)
        if not np.any(overlap):  # 겹치는 영역이 없으면
            remaining_contours.append(cnt)
            cv2.drawContours(mask, [cnt], -1, 255, -1)
    return remaining_contours


# Canny Edge Detection
edges = cv2.Canny(img, 240, 240)


kernel = np.ones((4, 4), np.uint8)
morph = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
edges = cv2.blur(edges, (3, 3))
# 컨투어 검출 (Canny후에 블러처리까지 한 결과를 입력으로 사용)
contours, hierarchy = cv2.findContours(morph, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)
realcontour = []
for i, cnt in enumerate(contours):
    area = cv2.contourArea(cnt)
    peri = cv2.arcLength(cnt, True)
    if 400000 > area > 5000:  # 면적 필터링-카메라 설치 후 하이퍼파리미터 조정
        # 컨투어 내부 마스크 생성
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        mean_val, stddev_val = cv2.meanStdDev(img, mask=mask)
        # 색상이 비슷한지 판별 (임의 기준: 표준편차가 50 미만)-카메라 설치 후 하이퍼파라미터 조정
        if np.all(stddev_val < 50):
            cnt = cv2.approxPolyDP(cnt, 0.0001 * peri, True)
            print(cnt.shape)
            realcontour.append(cnt)
filtered_contours = remove_overlapping_contours(realcontour)

for i, cnt in enumerate(filtered_contours):
    cv2.drawContours(img, [cnt], -1, (255, 0, 255), 2)
    M = cv2.moments(cnt, False)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    mu20 = M["mu20"]
    mu02 = M["mu02"]
    mu11 = M["mu11"]
    angle = 0.5 * math.atan2(2 * mu11, mu20 - mu02)  # 라디안 단위
    angle = angle * (180 / np.pi)  # 각도로 변환
    cv2.circle(img, (cX, cY), radius=5, color=(0, 255, 0), thickness=-1)  # 녹색 점
    # 컨투어를 내부 면적 기준으로 정렬
    sorted_contours = sorted(filtered_contours, key=lambda cnt: cv2.contourArea(cnt))

print("ans")
print(len(sorted_contours))
point = (cX, cY)
print(point)
image = cv2.imread(f"{filename}.jpg", cv2.IMREAD_COLOR)
height, width = image.shape[:2]
scale_x = width / 800
scale_y = height / 600
point = (int(image.shape[1] * cX / 800), int(image.shape[0] * cY / 600))
# 원본 이미지에 컨투어 그리기
for contour in sorted_contours:
    contour_rescaled = [
        (int(pt[0][0] * scale_x), int(pt[0][1] * scale_y)) for pt in contour
    ]
contour_rescaled = np.array(contour_rescaled)
contour_rescaled = contour_rescaled.reshape(-1, 1, 2)
mask = np.zeros(image.shape[:2], dtype=np.uint8)
# 모든 컨투어를 마스크에 그리기
cv2.drawContours(mask, [contour_rescaled], -1, (255, 255, 255), thickness=cv2.FILLED)
loss = 1000000000000000000000000000000000000
a = 0.001
eps = 2
hh = 0.1
x, y, ang = cX * scale_x, cY * scale_y, angle


def make_rect(x, y, w, h, ang):
    # 직사각형 만들기
    rect_mask = np.zeros(image.shape[:2], dtype=np.uint8)

    # Calculate the rotated rectangle
    rect = ((x, y), (w, h), ang)

    # Get the 4 corners of the rotated rectangle
    box = cv2.boxPoints(rect)  # Get the corner points (float values)
    box = np.int64(box)  # Convert to integer values

    # Draw the rotated rectangle with white color
    cv2.fillPoly(rect_mask, [box], color=255)
    return rect_mask


def Loss(contour_mask, rectangle_mask):
    both_masks = cv2.bitwise_and(contour_mask, rectangle_mask)
    only_contour = cv2.bitwise_and(
        contour_mask, cv2.bitwise_not(rectangle_mask)
    )  # Only in contour mask
    only_rectangle = cv2.bitwise_and(
        rectangle_mask, cv2.bitwise_not(contour_mask)
    )  # Only in rectangle mask
    both = cv2.countNonZero(both_masks)
    only_contour = cv2.countNonZero(only_contour)
    only_rectangle = cv2.countNonZero(only_rectangle)
    print(both, only_contour, only_rectangle)
    aa = only_contour + only_rectangle * 10 - both
    return aa


area = cv2.contourArea(contour_rescaled)
w = math.sqrt(area)
h = area**0.5
print("ans")
print(x, y, w, h, ang)
x_, y_, w_, h_, ang_ = 0, 0, 0, 0, 0
# rect_mask = make_rect(x, y, w, h, ang)
# rect_mask = cv2.resize(rect_mask, (int(rect_mask.shape[1] * 0.2), int(rect_mask.shape[0] * 0.2)))
# cv2.imshow("rect", rect_mask)
for i in range(100):
    # print(x_, y_, w_, h_, ang_)
    count = count + 1

    rect_mask = make_rect(x, y, w, h, ang)
    aa = Loss(mask, rect_mask)

    rect_x = make_rect(x + eps, y, w, h, ang)

    x_ = (Loss(mask, rect_x) - aa) / eps

    rect_y = make_rect(x, y + eps, w, h, ang)
    y_ = (Loss(mask, rect_y) - aa) / eps

    rect_w = make_rect(x, y, w + eps, h, ang)
    w_ = (Loss(mask, rect_w) - aa) / eps

    rect_h = make_rect(x, y, w, h + eps, ang)
    h_ = (Loss(mask, rect_h) - aa) / eps

    rect_ang = make_rect(x, y, w, h, ang + hh)
    ang_ = (Loss(mask, rect_ang) - aa) / hh
    values = np.array([x_, y_, w_, h_, ang_])
    result = np.linalg.norm(values)
    if result < 100:
        print(count)
        print(x_, y_, w_, h_, ang_)
        print(x, y, w, h, ang)
        print("end")
        # cv2.imshow("real", rect_mask)
        break
    x = x - a * x_
    y = y - a * y_
    w = w - a * w_
    h = h - a * h_
    ang = ang - a * ang_

mask = np.zeros_like(image)
rect = ((x, y), (w, h), ang)
# Get the 4 corners of the rotated rectangle
box = cv2.boxPoints(rect)  # Get the corner points (float values)
box = np.int64(box)  # Convert to integer values
cv2.fillPoly(mask, [box], color=(255, 255, 255))

result = cv2.bitwise_and(image, mask)
mask = cv2.resize(mask, (int(mask.shape[1] * 0.2), int(mask.shape[0] * 0.2)))
cv2.imshow("rect", mask)
print(f"모양{result.shape}")
cv2.circle(result, point, 20, (0, 255, 0), -1)
print(result.shape)
# image = cv2.resize(image, (800, 600))  # 이미지 크기 조정-카메라 설치 후 하이퍼파라미터 조정
top_left, top_right, bottom_left, bottom_right = split_image(result, point, 0)
# result = cv2.resize(result, (int(result.shape[1] * 0.2), int(result.shape[0] * 0.2)))
# cv2.imshow("Contours with Color Analysis", result)
cv2.waitKey(0)
cv2.destroyAllWindows()


# 결과 출력
def where(imgx, imgy, loc):
    if loc == 1:
        imgx += cX * scale_x
    elif loc == 3:
        imgy += cY * scale_y
    elif loc == 2:
        imgx += cX * scale_x
        imgy += cY * scale_y
    img_point = (int(imgx), int(imgy))
    return img_point


template = cv2.imread("./image/marker_ideal.jpg", 0)


def marker_detector(side):
    count = 0
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(template, None)
    # 입력할 부분 설정
    input_image = cv2.cvtColor(side, cv2.COLOR_BGR2GRAY)  # 여기다
    kp2, des2 = orb.detectAndCompute(input_image, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    points1 = []  # 템플릿 이미지에서의 매칭된 특징점 좌표
    points2 = []  # 입력 이미지에서의 매칭된 특징점 좌표

    # 매칭 결과를 거리 기준으로 정렬
    matches = sorted(matches, key=lambda x: x.distance)
    for match in matches:
        # 첫 번째 이미지의 특징점 좌표
        pt1 = kp1[match.queryIdx].pt
        # 두 번째 이미지의 특징점 좌표
        pt2 = kp2[match.trainIdx].pt
        points1.append(pt1)
        points2.append(pt2)
        # 출력

    if points1 and points2:
        avg_x1 = np.mean([pt[0] for pt in points1])
        avg_y1 = np.mean([pt[1] for pt in points1])
        avg_x2 = np.mean([pt[0] for pt in points2])
        avg_y2 = np.mean([pt[1] for pt in points2])

        print(f"Template Image Average (x, y): ({avg_x1:.2f}, {avg_y1:.2f})")
        print(f"Input Image Average (x, y): ({avg_x2:.2f}, {avg_y2:.2f})")
    else:
        print("No matches found.")
    print(len(matches))
    points = [kp2[match.trainIdx].pt for match in matches]
    data = np.array(points)  # 리스트를 numpy 배열로 변환

    db = DBSCAN(eps=70, min_samples=10).fit(
        data
    )  # 라즈베리파이로 찍을 때는 이거 다시 튜닝

    # 클러스터 할당 결과
    labels = db.labels_

    # 결과 출력
    non_noise_labels = labels[labels != -1]
    largest_cluster_label = np.bincount(non_noise_labels).argmax()

    # 해당 클러스터의 점 추출
    largest_cluster_points = data[labels == largest_cluster_label]
    print(f"Number of points in the largest cluster: {len(largest_cluster_points)}")
    kavg = largest_cluster_points.mean(axis=0)

    return (
        kp1,
        kp2,
        matches,
        input_image,
        (avg_x1, avg_y1),
        (avg_x2, avg_y2),
        (kavg[0], kavg[1]),
    )


sides = [top_left, top_right, bottom_right, bottom_left]  # 순서 바꾸기
tem_xy = [(0, 0), (0, 0), (0, 0), (0, 0)]
image_xy = [(0, 0), (0, 0), (0, 0), (0, 0)]
kimage_xy = [(0, 0), (0, 0), (0, 0), (0, 0)]
real_circle = []
for i, side in enumerate(sides):
    kp1, kp2, matches, input_image, tem_xy[i], image_xy[i], kimage_xy[i] = (
        marker_detector(side)
    )
    output_image = cv2.drawMatches(
        template,
        kp1,
        input_image,
        kp2,
        matches[:30],
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    cv2.circle(sides[i], (int(image_xy[i][0]), int(image_xy[i][1])), 5, (0, 255, 0), -1)
    real_circle.append((int(kimage_xy[i][0]), int(kimage_xy[i][1])))
    cv2.circle(
        sides[i], (int(kimage_xy[i][0]), int(kimage_xy[i][1])), 20, (0, 255, 255), -1
    )
cv2.imwrite("output.jpg", output_image)
print(len(matches))
cv2.imshow("Matches", output_image)

new_point = (200, 150)
img_point = where(new_point[0], new_point[1], 3)
for i in range(4):
    real_circle[i] = where(real_circle[i][0], real_circle[i][1], i)
    cv2.circle(image, real_circle[i], 10, (0, 0, 255), -1)
# 시각화(완료한 부분)
print(img_point)
print(bottom_left.shape)
data = np.array(real_circle)  # box

print(f"점 4개 좌표{real_circle}")
ctrx, ctry = np.mean(data[:, 0]), np.mean(data[:, 1])


def calculate_angles(points):
    angles = []
    n = len(points)
    for i in range(n):
        # 현재 점과 다음 점
        p1 = points[i]
        p2 = points[(i + 1) % n]  # 다음 점, 마지막 점일 경우 첫 점과 연결

        # 각도 계산 (atan2 사용)
        angle = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
        angles.append(math.degrees(angle))  # 라디안을 도(degree)로 변환
    return angles


angles = calculate_angles(data)
angles = np.array(angles)
angles[angles > 90] -= 180
angles[angles < -90] += 180
# 1,3번 각도의 평균을 a, 2,4번 각도의 평균을 -90+a라고 할 때 계산하면 아래처럼 나옴
angle = np.mean(angles) + 45
print(f"Translation : ({ctrx},{ctry})")
print(f"rotation : {angle}")

# 시각화

cv2.circle(image, img_point, 5, (0, 255, 0), -1)
cv2.circle(bottom_left, new_point, 300, (0, 255, 0), -1)
image = cv2.resize(image, (int(image.shape[1] * 0.2), int(image.shape[0] * 0.2)))
cv2.imshow("Contours with Color Analysis_final", image)
cv2.imwrite("final_image.jpg", image)
top_left = cv2.resize(
    top_left, (int(top_left.shape[1] * 0.2), int(top_left.shape[0] * 0.2))
)
cv2.imshow("Top Left", top_left)
top_right = cv2.resize(
    top_right, (int(top_right.shape[1] * 0.2), int(top_right.shape[0] * 0.2))
)
cv2.imshow("Top Right", top_right)
bottom_left = cv2.resize(
    bottom_left, (int(bottom_left.shape[1] * 0.2), int(bottom_left.shape[0] * 0.2))
)
cv2.imshow("Bottom Left", bottom_left)
bottom_right = cv2.resize(
    bottom_right, (int(bottom_right.shape[1] * 0.2), int(bottom_right.shape[0] * 0.2))
)
cv2.imshow("Bottom Right", bottom_right)

cv2.waitKey(10000)
cv2.destroyAllWindows()
