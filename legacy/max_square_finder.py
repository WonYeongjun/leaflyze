import cv2
import numpy as np
import random
import math
count=0
# 이미지 불러오기
filename='green'
img = cv2.imread(f"{filename}.jpg", cv2.IMREAD_COLOR)
img = cv2.resize(img, (800, 600))  # 이미지 크기 조정-카메라 설치 후 하이퍼파라미터 조정


def make_rect(x,y,w,h,ang):
#직사각형 만들기
    rect_mask = np.zeros(img.shape[:2], dtype=np.uint8)

    # Calculate the rotated rectangle
    rect = ((x,y), (w,h), ang)

    # Get the 4 corners of the rotated rectangle
    box = cv2.boxPoints(rect)  # Get the corner points (float values)
    box = np.int64(box)        # Convert to integer values

    # Draw the rotated rectangle with white color
    cv2.fillPoly(rect_mask, [box], color=255)
    return rect_mask

def Loss(contour_mask,rectangle_mask):
    both_masks = cv2.bitwise_and(contour_mask, rectangle_mask)
    only_contour = cv2.bitwise_and(contour_mask, cv2.bitwise_not(rectangle_mask))  # Only in contour mask
    only_rectangle = cv2.bitwise_and(rectangle_mask, cv2.bitwise_not(contour_mask))  # Only in rectangle mask
    both = cv2.countNonZero(both_masks)
    only_contour = cv2.countNonZero(only_contour)
    only_rectangle = cv2.countNonZero(only_rectangle)
    aa=only_contour+only_rectangle*10-both
    return aa


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
        print(f"area : {area}")
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
            loss=1000000000000000000000000000000000000
            a=0.001
            print(cX)
            eps=2
            hh=0.1
            x,y,ang=cX,cY,angle
            w=math.sqrt(area)
            h=area**0.5
            print("ans")
            print(x,y,w,h,ang)
            x_,y_,w_,h_,ang_=0,0,0,0,0
            for i in range(1000):
                count=count+1

                rect_mask=make_rect(x,y,w,h,ang)
                aa=Loss(mask,rect_mask)
                
                rect_x=make_rect(x+eps,y,w,h,ang)

                x_=(Loss(mask,rect_x)-aa)/eps

                rect_y=make_rect(x,y+eps,w,h,ang)
                y_=(Loss(mask,rect_y)-aa)/eps

                rect_w=make_rect(x,y,w+eps,h,ang)
                w_=(Loss(mask,rect_w)-aa)/eps

                rect_h=make_rect(x,y,w,h+eps,ang)
                h_=(Loss(mask,rect_h)-aa)/eps

                rect_ang=make_rect(x,y,w,h,ang+hh)
                ang_=(Loss(mask,rect_ang)-aa)/hh
                values = np.array([x_,y_,w_,h_,ang_])
                result = np.linalg.norm(values)
                if result<100:
                    print(count)
                    print(x_,y_,w_,h_,ang_)
                    print(x,y,w,h,ang)
                    print("end")
                    cv2.imshow("real",rect_mask)
                    break
                x=x-a*x_
                y=y-a*y_
                w=w-a*w_
                h=h-a*h_
                ang=ang-a*ang_
                
            realcontour.append(cnt)


print(x,y,w,h,ang)
filtered_contours=remove_overlapping_contours(realcontour)
rect=((x,y),(w,h),ang)
box=cv2.boxPoints(rect)
box=np.int64(box)
cv2.drawContours(img,[box],-1,(255,0,0),2)
cv2.imshow("box", img)
for i, cnt in enumerate(filtered_contours):
    cv2.drawContours(img, [cnt], -1, (255,0,255), 2)
print(len(realcontour))
# 결과 출력
cv2.imshow("Contours with Color Analysis", img)
cv2.waitKey(5000)  # 5000ms = 5초 동안 대기