import cv2
import numpy as np

# ArUco 마커 템플릿 이미지 로드
template = cv2.imread('marker_4.png', 0)  # 그레이스케일로 읽기

# ORB 특징점 추출기 생성
orb = cv2.ORB_create()

# 특징점 추출
kp1, des1 = orb.detectAndCompute(template, None)

# 입력 이미지 로드
input_image = cv2.imread('whywhywifi.jpg', 0)
input_image = cv2.resize(input_image,(40,30))
input_image = cv2.resize(input_image,(400,300))
# 입력 이미지에서 특징점 추출
kp2, des2 = orb.detectAndCompute(input_image, None)

# BFMatcher로 매칭
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)

# 매칭 결과를 거리 기준으로 정렬
matches = sorted(matches, key=lambda x: x.distance)
for match in matches:
    # 첫 번째 이미지의 특징점 좌표
    pt1 = kp1[match.queryIdx].pt
    # 두 번째 이미지의 특징점 좌표
    pt2 = kp2[match.trainIdx].pt
        # 출력
    print(f"Match: QueryIdx={match.queryIdx}, TrainIdx={match.trainIdx}")
    print(f"Image 1 (pt1): {pt1}, Image 2 (pt2): {pt2}")
print(len(matches))
# 매칭 결과를 이미지에 그리기
output_image = cv2.drawMatches(template, kp1, input_image, kp2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv2.imwrite("output.jpg", output_image)
output_image=cv2.resize(output_image,(int(output_image.shape[1]*0.5),int(output_image.shape[0]*0.5)))
# 결과 이미지 보여주기
cv2.imshow('Matches', output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()