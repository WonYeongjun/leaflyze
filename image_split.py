import cv2
import math

filename='box'

#이미지 4분할
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

count=0
img = cv2.imread(f"{filename}.jpg", cv2.IMREAD_COLOR)
size=img.shape[:2]
print(size)
img = cv2.resize(img, (800,600))  # 이미지 크기 조정-카메라 설치 후 하이퍼파라미터 조정

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
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        mean_val, stddev_val = cv2.meanStdDev(img, mask=mask)
        # 색상이 비슷한지 판별 (임의 기준: 표준편차가 50 미만)-카메라 설치 후 하이퍼파라미터 조정
        if np.all(stddev_val < 50):
            cnt= cv2.approxPolyDP(cnt, 0.0001* peri, True) 
            print(cnt.shape)
            realcontour.append(cnt)
            
filtered_contours=remove_overlapping_contours(realcontour)

for i, cnt in enumerate(filtered_contours):
    cv2.drawContours(img, [cnt], -1, (255,0,255), 2)
    M = cv2.moments(cnt, False)
    cX = int(M['m10'] / M['m00'])
    cY = int(M['m01'] / M['m00'])
    
print('ans')
print(len(filtered_contours))

point=(cX,cY)
image = cv2.imread(f"{filename}.jpg", cv2.IMREAD_COLOR)
image = cv2.resize(image, (800, 600))  # 이미지 크기 조정-카메라 설치 후 하이퍼파라미터 조정
top_left, top_right, bottom_left, bottom_right = split_image(image, point,0)
# 결과 출력
new_point=(50,60)
imgx=new_point[0]
imgy=new_point[1]
loc=4
if (loc==2):
    imgx+=cX   
elif (loc==3):
    imgy+=cY
elif (loc==4):
    imgx+=cX
    imgy+=cY
img_point=(int(imgx),int(imgy))

##마커 찾기

#마커 설정
template = cv2.imread('marker_4.png', 0)
orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(template, None)

#입력할 부분 설정
input_image=cv2.cvtColor(top_left, cv2.COLOR_BGR2GRAY) #여기다
kp2, des2 = orb.detectAndCompute(input_image, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)

for match in matches:
    pt1 = kp1[match.queryIdx].pt
    pt2 = kp2[match.trainIdx].pt
    print(f"Match: QueryIdx={match.queryIdx}, TrainIdx={match.trainIdx}")
    print(f"Image 1 (pt1): {pt1}, Image 2 (pt2): {pt2}")

print(len(matches))
output_image = cv2.drawMatches(template, kp1, input_image, kp2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv2.imwrite("output.jpg", output_image)
cv2.imshow('Matches', output_image)
##시각화(완료한 부분)
cv2.circle(img, img_point, 5, (0, 255, 0), -1)
cv2.circle(bottom_right, new_point, 5, (0, 255, 0), -1)
cv2.imshow("Contours with Color Analysis_final", img)
cv2.imshow("Top Left", top_left)
cv2.imshow("Top Right", top_right)
cv2.imshow("Bottom Left", bottom_left)
cv2.imshow("Bottom Right", bottom_right)
cv2.waitKey(0)
cv2.destroyAllWindows()