import numpy as np
from scipy.linalg import polar

# 예제 B (2D -> 3D 확장)
B_2D = np.array([[0, 0, 1, 1], [0, 1, 1, 0]])  # 기존 2D 점
C_2D = np.array([[0, 0, 1, 1], [0, 1, 1.5, 0]])  # 변형된 2D 점

# Z 좌표를 0으로 추가하여 3D 변환
B_3D = np.vstack((B_2D, np.zeros((1, B_2D.shape[1]))))  # (3×4 행렬)
C_3D = np.vstack((C_2D, np.zeros((1, C_2D.shape[1]))))  # (3×4 행렬)

# 안정적인 역행렬 계산 (의사역행렬 사용)
A_3D = C_3D @ B_3D.T @ np.linalg.pinv(B_3D @ B_3D.T)
print("변형 구배 A (3D):")
print(A_3D)

# 방법 1: 극분해 (Polar Decomposition) 이용
R_polar_3D, U_polar_3D = polar(A_3D)
print("\n회전 행렬 R (Polar Decomposition, 3D):")
print(R_polar_3D)

# 방법 2: SVD를 이용한 회전 행렬 계산
U_A, S_A, V_A_T = np.linalg.svd(A_3D)
R_svd_3D = U_A @ V_A_T
print("\n회전 행렬 R (SVD, 3D):")
print(R_svd_3D)

# 2D 회전 행렬을 XY 평면에서 추출 (상위 2×2 부분)
R_polar_2D = R_polar_3D[:2, :2]
R_svd_2D = R_svd_3D[:2, :2]

# 회전 각도 계산 (라디안 & 도 단위 변환)
theta_polar = np.arctan2(R_polar_2D[1, 0], R_polar_2D[0, 0])  # radian
theta_svd = np.arctan2(R_svd_2D[1, 0], R_svd_2D[0, 0])  # radian

print("\n회전 각도 (Polar Decomposition, 3D): {:.4f} 라디안 ({:.2f} 도)".format(theta_polar, np.degrees(theta_polar)))
print("회전 각도 (SVD, 3D): {:.4f} 라디안 ({:.2f} 도)".format(theta_svd, np.degrees(theta_svd)))