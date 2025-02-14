import numpy as np
from scipy.linalg import polar

# 예제 B (2x4 행렬)와 C (2x4 행렬)
B = np.array([[0,0,1,1], [0,1,1,0]])
C = np.array([[0,0,1,1], [0,1,1.5,0]])

# 변형 구배 A 계산
A = C @ B.T @ np.linalg.inv(B @ B.T)
print("변형 구배 A:")
print(A)

# 방법 1: 극분해(Polar Decomposition) 이용
R_polar, U_polar = polar(A)
print("\n회전 행렬 R (Polar Decomposition):")
print(R_polar)

# 방법 2: SVD를 이용한 회전 행렬 계산
U_A, S_A, V_A_T = np.linalg.svd(A)
R_svd = U_A @ V_A_T
print("\n회전 행렬 R (SVD):")
print(R_svd)

# 회전 각도 계산 (라디안 & 도 단위 변환)
theta_polar = np.arctan2(R_polar[1, 0], R_polar[0, 0])  # radian
theta_svd = np.arctan2(R_svd[1, 0], R_svd[0, 0])  # radian

print("\n회전 각도 (Polar Decomposition): {:.4f} 라디안 ({:.2f} 도)".format(theta_polar, np.degrees(theta_polar)))
print("회전 각도 (SVD): {:.4f} 라디안 ({:.2f} 도)".format(theta_svd, np.degrees(theta_svd)))
