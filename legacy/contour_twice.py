import cv2
import numpy as np
import matplotlib.pyplot as plt
color=(0,0,0)
count=0
# 이미지 불러오기
filename='02'
img = cv2.imread(f"{filename}.jpg", cv2.IMREAD_COLOR)
image=img
img = cv2.resize(img, (800, 600))  # 이미지 크기 조정-카메라 설치 후 하이퍼파라미터 조정

def remove_overlapping_contours(contours, img_size=(500, 500)):
    # 컨투어 리스트 정렬 (면적 기준 내림차순)
    contours = sorted(contours, key=lambda cnt: cv2.contourArea(cnt), reverse=True)
    remaining_contours = []
    
    # 빈 마스크 초기화
    mask = np.zeros(img_size, dtype=np.uint8)

    for cnt in contours:
        # 현재 컨투어를 그린 마스크 생성
        current_mask = np.zeros(img_size, dtype=np.uint8)
        cv2.drawContours(current_mask, [cnt], -1, 255, -1)

        # 기존 마스크와 AND 연산으로 겹치는 영역 확인
        overlap = cv2.bitwise_and(mask, current_mask)

        if not np.any(overlap):  # 겹치는 영역이 없으면
            remaining_contours.append(cnt)
            # 현재 컨투어를 전체 마스크에 추가
            cv2.drawContours(mask, [cnt], -1, 255, -1)
    return remaining_contours

# Canny Edge Detection
edges = cv2.Canny(img, 240, 240)  # Canny로 엣지 검출

kernel = np.ones((4,4), np.uint8)
morph = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
edges  = cv2.blur(edges, (3, 3))
# 컨투어 검출 (Canny후에 블러처리까지 한 결과를 입력으로 사용)
contours, hierarchy = cv2.findContours(morph, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)
realcontour=[]
for i, cnt in enumerate(contours):
    area = cv2.contourArea(cnt)
    peri = cv2.arcLength(cnt, True)
    if 400000>area > 5000:  # 면적 필터링-카메라 설치 후 하이퍼파리미터 조정
        # 컨투어 내부 마스크 생성
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, -1)  # 내부를 채운 마스크

        mean_val, stddev_val = cv2.meanStdDev(img, mask=mask)  # 평균 및 표준편차 계산

        # 평균 및 표준편차 출력
        print(f"Contour {i}: Mean BGR = {mean_val.flatten()}, StdDev = {stddev_val.flatten()}")

        # 색상이 비슷한지 판별 (임의 기준: 표준편차가 50 미만)-카메라 설치 후 하이퍼파라미터 조정
        if np.all(stddev_val < 50):
            # 컨투어를 이미지에 그리기
            cnt= cv2.approxPolyDP(cnt, 0.005* peri, True)
            M = cv2.moments(cnt, False)
            cX = int(M['m10'] / M['m00'])
            cY = int(M['m01'] / M['m00'])
            color=img[cX,cY]
            print([255,255,255]-color)
            print(cnt.shape)
            realcontour.append(cnt)
            
filtered_contours=remove_overlapping_contours(realcontour)

for i, cnt in enumerate(filtered_contours):
    cv2.drawContours(img, [cnt], -1, (255,0,255), 2)
print(len(realcontour))

#호진 코드

def is_difference_large(values, threshold):
    """
    주어진 리스트에서 최대값과 최소값의 차이가 임계값(threshold)보다 큰지 판단.
    
    Parameters:
        values (list or tuple): [a, b, c]와 같은 숫자 리스트
        threshold (float): 차이가 크다고 판단할 기준값
        
    Returns:
        bool: True (차이가 큼), False (차이가 작음)
    """
    max_value = max(values)  # 최대값
    min_value = min(values)  # 최소값
    difference = max_value - min_value  # 최대값과 최소값의 차이

    return difference < threshold  # 차이가 임계값보다 큰지 확인


threshold = 10  # 차이가 크다고 판단할 기준

# 함수 호출
result = is_difference_large(color, threshold)

if result:
    print("black")
    color=np.array([0,0,255])
else:
    print(f"color{color}")

image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Gaussian Blur 적용
blurred_image = cv2.GaussianBlur(image_gray, (5, 5), 0)

# Harris Corner Detection
harris_corners_blurred = cv2.cornerHarris(blurred_image, blockSize=2, ksize=3, k=0.04)

# 코너를 강조하기 위해 팽창(dilate)
harris_corners_dilated_blurred = cv2.dilate(harris_corners_blurred, None)

# 원본 이미지에 코너 강조 (빨간색으로 표시)
harris_corners_image_blurred = image.copy()
harris_corners_image_blurred[harris_corners_dilated_blurred > 0.01 * harris_corners_dilated_blurred.max()] = [255,255,255]-color

img=harris_corners_image_blurred

#코드 재실행
img = cv2.resize(img, (800, 600))  # 이미지 크기 조정-카메라 설치 후 하이퍼파라미터 조정

# Canny Edge Detection
edges = cv2.Canny(img, 240, 240)  # Canny로 엣지 검출

kernel = np.ones((4,4), np.uint8)
morph = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
edges  = cv2.blur(edges, (3, 3))
# 컨투어 검출 (Canny후에 블러처리까지 한 결과를 입력으로 사용)
contours, hierarchy = cv2.findContours(morph, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)
realcontour=[]
for i, cnt in enumerate(contours):
    area = cv2.contourArea(cnt)
    peri = cv2.arcLength(cnt, True)
    if 400000>area > 5000:  # 면적 필터링-카메라 설치 후 하이퍼파리미터 조정
        # 컨투어 내부 마스크 생성
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, -1)  # 내부를 채운 마스크

        mean_val, stddev_val = cv2.meanStdDev(img, mask=mask)  # 평균 및 표준편차 계산

        # 평균 및 표준편차 출력
        print(f"Contour {i}: Mean BGR = {mean_val.flatten()}, StdDev = {stddev_val.flatten()}")

        # 색상이 비슷한지 판별 (임의 기준: 표준편차가 50 미만)-카메라 설치 후 하이퍼파라미터 조정
        if np.all(stddev_val < 50):
            # 컨투어를 이미지에 그리기
            cnt= cv2.approxPolyDP(cnt, 0.005* peri, True)
            M = cv2.moments(cnt, False)
            cX = int(M['m10'] / M['m00'])
            cY = int(M['m01'] / M['m00'])
            color=img[cX,cY]
            print([255,255,255]-color)
            print(cnt.shape)
            realcontour.append(cnt)
            
filtered_contours=remove_overlapping_contours(realcontour)

for i, cnt in enumerate(filtered_contours):
    cv2.drawContours(img, [cnt], -1, (255,0,255), 2)
print(len(realcontour))
# 결과 출력
cv2.imshow("Contours with Color Analysis_final", img)
cv2.waitKey(5000)  # 5000ms = 5초 동안 대기
