import cv2
import numpy as np
import random
import math
count=0
# 이미지 불러오기
filename='7'
img = cv2.imread(f"{filename}.jpg", cv2.IMREAD_COLOR)
img = cv2.resize(img, (800, 600))  # 이미지 크기 조정-카메라 설치 후 하이퍼파라미터 조정
def make_rect(cx,cy,area,angle):
#직사각형 만들기
    rect_mask = np.zeros(img.shape[:2], dtype=np.uint8)

    # Define rectangle parameters
    center = (random.uniform(cx-img.shape[1]/5,cx+img.shape[1]/5),random.uniform(cy-img.shape[0]/5,cy+img.shape[0]/5))  # Center of the rectangle (x, y)
    size = (random.uniform((area**0.5)/5,(area**0.5)*5),random.uniform((area**0.5)/5,(area**0.5)*5))    # Width and height of the rectangle (width, height)
    angle = random.uniform(-10, 10)  # Rotation angle in degrees (counter-clockwise)

    # Calculate the rotated rectangle
    rect = ((center[0], center[1]), (size[0], size[1]), angle)

    # Get the 4 corners of the rotated rectangle
    box = cv2.boxPoints(rect)  # Get the corner points (float values)
    box = np.int64(box)        # Convert to integer values

    # Draw the rotated rectangle with white color
    cv2.fillPoly(rect_mask, [box], color=255)
    return center,size,angle,rect_mask



#컨투어 유일하게 만들기

def remove_overlapping_contours(contours, img_size=(800, 600)):
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
edges = cv2.Canny(img, 200, 200)  # Canny로 엣지 검출


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
        cv2.imshow('mask'+str(i),mask)

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
            mu20 = M["mu20"]
            mu02 = M["mu02"]
            mu11 = M["mu11"]
            
            angle = 0.5 * math.atan2(2 * mu11, mu20 - mu02)  # 라디안 단위
            angle = angle * (180 / np.pi)  # 각도로 변환
            cv2.circle(img, (cX, cY), radius=5, color=(0, 255, 0), thickness=-1)  # 녹색 점
        
        # 중심점 좌표 텍스트 표시
            text = f"({cX}, {cY})"
            cv2.putText(img, text, (cX + 10, cY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            ans=1000000000000000000000000000000000000
            for i in range(10000):
                area=cv2.contourArea(cnt)
                ctr,si,ang,rect_mask=make_rect(cX,cY,area,angle)
                both_masks = cv2.bitwise_and(mask, rect_mask)
                only_contour = cv2.bitwise_and(mask, cv2.bitwise_not(rect_mask))  # Only in contour mask
                only_rectangle = cv2.bitwise_and(rect_mask, cv2.bitwise_not(mask))  # Only in rectangle mask
                both = cv2.countNonZero(both_masks)
                only_contour = cv2.countNonZero(only_contour)
                only_rectangle = cv2.countNonZero(only_rectangle)
                aa=100*only_contour-both**2
                if ans>aa and only_rectangle<100:
                    ans=aa
                    realctr=ctr
                    realsi=si
                    realang=ang
                    cv2.imshow("real",rect_mask)
            
            print(realctr,realsi,realang)
            realcontour.append(cnt)



filtered_contours=remove_overlapping_contours(realcontour)
rect=(realctr,realsi,realang)
print(rect)
box=cv2.boxPoints(rect)
box=np.int64(box)
cv2.drawContours(img,[box],-1,(0,255,0),2)

for i, cnt in enumerate(filtered_contours):
    cv2.drawContours(img, [cnt], -1, (255,0,255), 2)
print(len(realcontour))
# 결과 출력
cv2.imshow("Contours with Color Analysis", img)
cv2.waitKey(5000)  # 5000ms = 5초 동안 대기