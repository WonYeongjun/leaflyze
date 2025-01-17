import cv2
import numpy as np

count=0
# 이미지 불러오기
filename=3
img = cv2.imread(f"{filename}.jpg", cv2.IMREAD_COLOR)
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
edges = cv2.Canny(img, 190, 200)  # Canny로 엣지 검출
edges  = cv2.blur(edges, (3, 3)) #이미지 블러처리-카메라 설치 후 하이퍼파라미터 조정
# 컨투어 검출 (Canny후에 블러처리까지 한 결과를 입력으로 사용)
contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)
realcontour=[]
for i, cnt in enumerate(contours):
    area = cv2.contourArea(cnt)
    peri = cv2.arcLength(cnt, True)
    if 400000>area > 5000:  # 면적 필터링-카메라 설치 후 하이퍼파리미터 조정
        # 컨투어 내부 마스크 생성
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, -1)  # 내부를 채운 마스크
        cv2.imshow('mask'+str(i),mask)

        mean_val, stddev_val = cv2.meanStdDev(img, mask=mask)  # 평균 및 표준편차 계산

        # 평균 및 표준편차 출력
        print(f"Contour {i}: Mean BGR = {mean_val.flatten()}, StdDev = {stddev_val.flatten()}")

        # 색상이 비슷한지 판별 (임의 기준: 표준편차가 50 미만)-카메라 설치 후 하이퍼파라미터 조정
        if np.all(stddev_val < 50):
            # 컨투어를 이미지에 그리기
            cnt= cv2.approxPolyDP(cnt, 0.005* peri, True)
            print(cnt.shape)
            realcontour.append(cnt)
            
cv2.imshow("Canny Edges", edges)
filtered_contours=remove_overlapping_contours(realcontour)

for i, cnt in enumerate(filtered_contours):
    cv2.drawContours(img, [cnt], -1, (255,0,255), 2)
print(len(realcontour))
# 결과 출력
cv2.imshow("Contours with Color Analysis", img)

import svgwrite

def save_contours_as_svg(contours, svg_filename, img_size):
    # SVG 파일 생성
    dwg = svgwrite.Drawing(svg_filename, size=img_size)
    
    # 각 컨투어를 SVG 경로로 변환하여 추가
    for cnt in contours:
        path_data = []
        for point in cnt:
            x, y = point[0]
            path_data.append((x, y))
        
        # SVG path로 추가
        path_str = "M " + " L ".join(f"{x},{y}" for x, y in path_data) + " Z"
        dwg.add(dwg.path(d=path_str, fill="none", stroke="black", stroke_width=1))
    
    # SVG 저장
    dwg.save()

# 이미지 크기 및 SVG 파일 이름 설정
image_size = (img.shape[1], img.shape[0])  # OpenCV는 (height, width), SVG는 (width, height)
svg_filename = "contours.svg"

# SVG 저장
save_contours_as_svg(filtered_contours, svg_filename, image_size)
print(f"Contours saved to {svg_filename}")

def save_individual_contours_as_svg(contours, img_size):
    for idx, cnt in enumerate(contours):
        # 각 컨투어에 대해 SVG 파일 생성
        svg_filename = f"contour_{idx + 1}.svg"
        dwg = svgwrite.Drawing(svg_filename, size=img_size)
        
        # 컨투어를 SVG path로 변환
        path_data = []
        for point in cnt:
            x, y = point[0]
            path_data.append((x, y))
        
        # SVG path 추가
        path_str = "M " + " L ".join(f"{x},{y}" for x, y in path_data) + " Z"
        dwg.add(dwg.path(d=path_str, fill="none", stroke="black", stroke_width=1))
        
        # SVG 저장
        dwg.save()
        print(f"Saved: {svg_filename}")

save_individual_contours_as_svg(filtered_contours, image_size)
cv2.waitKey(5000)  # 5000ms = 5초 동안 대기