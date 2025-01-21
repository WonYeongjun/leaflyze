import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
# ArUco 마커 템플릿 이미지 로드
template = cv2.imread('marker_4.png', 0)  # 그레이스케일로 읽기

# ORB 특징점 추출기 생성
orb = cv2.ORB_create()

# 특징점 추출
kp1, des1 = orb.detectAndCompute(template, None)

# 입력 이미지 로드
input_image = cv2.imread('whywhywifi.jpg', 0)
# 입력 이미지에서 특징점 추출
# _, input_image = cv2.threshold(input_image, 127, 255, cv2.THRESH_BINARY)
kp2, des2 = orb.detectAndCompute(input_image, None)

# BFMatcher로 매칭
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
points = np.array(points)  # 리스트를 numpy 배열로 변환

# K-Means 군집화
k = 3  # 클러스터 개수 (적절히 설정)
kmeans = KMeans(n_clusters=k, random_state=0)
kmeans.fit(points)

# 각 좌표의 클러스터 레이블
labels = kmeans.labels_

# 가장 많은 좌표가 속한 클러스터 찾기
unique_labels, counts = np.unique(labels, return_counts=True)
largest_cluster_label = unique_labels[np.argmax(counts)]

# 가장 큰 클러스터에 속한 좌표들만 추출
largest_cluster_points = points[labels == largest_cluster_label]

# 평균 좌표 계산
if len(largest_cluster_points) > 0:
    avg_x = np.mean(largest_cluster_points[:, 0])
    avg_y = np.mean(largest_cluster_points[:, 1])
    print(f"Largest Cluster Average (x, y): ({avg_x:.2f}, {avg_y:.2f})")
else:
    print("No points in the largest cluster.")

# 매칭 결과를 이미지에 그리기
output_image = cv2.drawMatches(template, kp1, input_image, kp2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv2.imwrite("output.jpg", output_image)
output_image=cv2.resize(output_image,(int(output_image.shape[1]*0.5),int(output_image.shape[0]*0.5)))
# 결과 이미지 보여주기
cv2.imshow('Matches', output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()