import cv2
import math
from sklearn.cluster import DBSCAN
import numpy as np

# filename = "./image/pink/fin_cal_img_20250207_141129.jpg"
filename = "./image/pink/fin_cal_img_20250207_141201"


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


if __name__ == "__main__":
    count = 0
    img = cv2.imread(f"{filename}.jpg", cv2.IMREAD_COLOR)
    size = img.shape[:2]
    print(size)

    # Canny Edge Detection
    img = cv2.blur(img, (3, 3))
    edges = cv2.Canny(img, 240, 240)

    kernel = np.ones((4, 4), np.uint8)
    morph = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    # 컨투어 검출 (Canny후에 블러처리까지 한 결과를 입력으로 사용)

    import matplotlib.pyplot as plt

    # Convert BGR image to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Plot the original image
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(img_rgb)
    plt.axis("off")

    # Plot the edges
    plt.subplot(1, 2, 2)
    plt.title("Edges")
    plt.imshow(edges, cmap="gray")
    plt.axis("off")

    plt.show()

    contours, hierarchy = cv2.findContours(
        morph, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS
    )
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
        sorted_contours = sorted(
            filtered_contours, key=lambda cnt: cv2.contourArea(cnt)
        )

    print("ans")
    print(len(sorted_contours))
    for contour in sorted_contours:
        cv2.drawContours(img, [contour], -1, (0, 255, 0), 2)
    cv2.imshow("Contours with Color Analysis", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
