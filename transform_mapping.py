# 좌표변환 예시 시각화
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 원본 직사각형 필드 (100x50 크기)
src_pts = np.float32([[0, 0], [100, 0], [100, 50], [0, 50]])

# 왜곡된 사각형 좌표
dst_pts = np.float32([[10, 5], [90, 15], [105, 45], [15, 50]])

# 투시 변환 행렬 계산 (왜곡된 -> 직사각형)
M = cv2.getPerspectiveTransform(dst_pts, src_pts)


def transform_point(x, y):
    """왜곡된 사각형 내의 좌표 (x, y)를 직사각형 필드 내의 대응 좌표로 변환"""
    input_point = np.array([[[x, y]]], dtype=np.float32)
    transformed_point = cv2.perspectiveTransform(input_point, M)
    return transformed_point[0][0]  # 변환된 (x, y) 좌표 반환


# 테스트할 점 (왜곡된 사각형 내부의 점)
test_points = [(50, 25)]

# 변환된 점 계산
transformed_points = [transform_point(x, y) for x, y in test_points]

# 플롯 생성
plt.figure(figsize=(6, 4))
ax = plt.gca()

# 두 개의 사각형을 그림
original_polygon = np.vstack([src_pts, src_pts[0]])  # 원본 직사각형
distorted_polygon = np.vstack([dst_pts, dst_pts[0]])  # 왜곡된 사각형

plt.plot(original_polygon[:, 0], original_polygon[:, 1], "bo-", label="직사각형 필드")
plt.plot(distorted_polygon[:, 0], distorted_polygon[:, 1], "ro-", label="왜곡된 사각형")

# 점과 변환된 점을 표시
for (dx, dy), (sx, sy) in zip(test_points, transformed_points):
    plt.plot(dx, dy, "ro")  # 왜곡된 사각형 내의 점
    plt.plot(sx, sy, "bo")  # 변환된 직사각형 필드 내의 점
    plt.arrow(sx, sy, dx - sx, dy - sy, head_width=2, head_length=3, fc='gray', ec='gray')


# 축 설정
plt.xlim(-10, 120)
plt.ylim(-10, 60)
plt.gca().invert_yaxis()  # 이미지 좌표계처럼 y축 방향 반전

plt.legend()
plt.title("mapping")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)
plt.show()
