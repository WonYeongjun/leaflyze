import cv2
import numpy as np

# ArUco 마커 템플릿 이미지 로드
template = cv2.imread('marker_4.png', 0)  # 그레이스케일로 읽기

# ORB 특징점 추출기 생성
orb = cv2.ORB_create()

# 특징점 추출
kp1, des1 = orb.detectAndCompute(template, None)

# 입력 이미지 로드
input_image = cv2.imread('Ddong2.jpg', 0)

# 입력 이미지에서 특징점 추출
kp2, des2 = orb.detectAndCompute(input_image, None)

# BFMatcher로 매칭
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)

# 매칭 결과를 거리 기준으로 정렬
matches = sorted(matches, key=lambda x: x.distance)

# 매칭 결과를 이미지에 그리기
output_image = cv2.drawMatches(template, kp1, input_image, kp2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv2.imwrite("output.jpg", output_image)
output_image=cv2.resize(output_image,(int(output_image.shape[1]*0.2),int(output_image.shape[0]*0.2)))
# 결과 이미지 보여주기
cv2.imshow('Matches', output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
