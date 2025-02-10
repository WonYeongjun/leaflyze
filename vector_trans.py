import cv2
import numpy as np
import matplotlib.pyplot as plt

# 원본 직사각형 필드 (100x50 크기)
src_pts = np.float32([[0, 0], [100, 0], [100, 50], [0, 50]])

# 왜곡된 사각형 좌표
dst_pts = np.float32([[10, 5], [90, 15], [105, 45], [15, 50]])

# 투시 변환 행렬 계산 (왜곡된 -> 직사각형)
M = cv2.getPerspectiveTransform(dst_pts, src_pts)

# 원래 벡터를 일정한 방향으로 설정 (예: 모든 점에서 동일한 방향의 벡터)
# 각 점에서 벡터 (10, 0)로 설정 (수평 방향)
vectors = np.array([[10, 0], [10, 0], [10, 0], [10, 0]])

# 사각형 내부 점 생성
x_values = np.linspace(0, 100, 10)  # X 좌표: 0에서 100까지 10개 점
y_values = np.linspace(0, 50, 5)   # Y 좌표: 0에서 50까지 5개 점

# 벡터 회전 및 왜곡 후 변형된 벡터 계산
def calculate_deformation_vector(x, y, vx, vy):
    """ 원래 벡터 (vx, vy)가 왜곡된 후 변형된 벡터를 계산 """
    input_point = np.array([[[x, y]]], dtype=np.float32)
    transformed_point = cv2.perspectiveTransform(input_point, M)[0][0]
    
    # 원래 벡터를 변형된 점에서 계산
    transformed_vector = np.array([vx, vy])
    return transformed_point, transformed_vector

# 플롯 생성
plt.figure(figsize=(6, 4))
ax = plt.gca()

# 원래 직사각형 (파란색)
original_polygon = np.vstack([src_pts, src_pts[0]])  # 원본 직사각형
plt.plot(original_polygon[:, 0], original_polygon[:, 1], 'bo-', label="원래 직사각형")

# 왜곡된 사각형 (빨간색)
distorted_polygon = np.vstack([dst_pts, dst_pts[0]])  # 왜곡된 사각형
plt.plot(distorted_polygon[:, 0], distorted_polygon[:, 1], 'ro-', label="왜곡된 사각형")

# 왜곡된 사각형 내부의 여러 점에 대해 벡터 회전 및 찌그러짐 벡터 그리기
for x in x_values:
    for y in y_values:
        # 각 점에서 벡터 회전 및 찌그러짐 벡터 계산
        for (vx, vy) in vectors:
            # 왜곡된 필드에서 벡터 회전 계산
            (sx, sy), _ = calculate_deformation_vector(x, y, vx, vy)
            
            # 변형된 사각형 내의 점에 대해서만 변형된 벡터를 그림 (파란색 화살표)
            plt.plot(x, y, 'bo')  # 원래 직사각형 내의 점
            plt.arrow(sx, sy, vx, vy, head_width=2, head_length=3, fc='blue', ec='blue')  # 변형된 벡터

# 축 설정
plt.xlim(-10, 120)
plt.ylim(-10, 60)
plt.gca().invert_yaxis()  # 이미지 좌표계처럼 y축 방향 반전

plt.legend()
plt.title("원래 벡터와 변형된 필드 시각화")
plt.xlabel("X 좌표")
plt.ylabel("Y 좌표")
plt.grid(True)
plt.show()
