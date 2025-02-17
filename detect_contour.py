import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

# --- 1. 이미지 로드 및 에지 검출 ---
img = cv2.imread("./image/pink/fin_cal_img_20250207_141129.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150)

# 에지 좌표 추출 (np.nonzero는 (행, 열) 순서이므로, (x, y) 순서로 변환)
ys, xs = np.nonzero(edges)
points_np = np.column_stack((xs, ys))  # (x, y) 좌표
print(f"Total edge points: {points_np.shape[0]}")

# --- 2. PyTorch 텐서로 변환 및 GPU 전송 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
points = torch.tensor(points_np, dtype=torch.float32, device=device)

# --- 3. 이미지 슬라이딩 윈도우 파라미터 ---
window_size = (50, 50)  # 윈도우 크기 (높이, 너비)
step_size = 25  # 윈도우 이동 간격

detected_lines = []  # 검출된 선 정보를 저장할 리스트

# 이미지의 높이, 너비
img_height, img_width = gray.shape

# --- 4. 슬라이딩 윈도우로 이미지를 나누어 각 영역에서 RANSAC 적용 ---
for y in range(0, img_height - window_size[0], step_size):
    for x in range(0, img_width - window_size[1], step_size):
        # 각 윈도우 영역의 (y, x) 좌표 범위
        window = edges[y : y + window_size[0], x : x + window_size[1]]

        # 윈도우 내 에지 점들의 좌표 추출
        ys, xs = np.nonzero(window)
        points_window = np.column_stack(
            (xs + x, ys + y)
        )  # (x, y) 좌표 (전체 이미지 좌표로 보정)

        if len(points_window) < 2:
            continue  # 점이 2개 미만이면 건너뛰기

        # --- 5. RANSAC 적용 (윈도우 영역 내에서) ---
        points = torch.tensor(points_window, dtype=torch.float32, device=device)

        # RANSAC 파라미터 설정
        num_iterations = 1000  # 각 라인 검출 시도 횟수
        distance_threshold = 2.0  # inlier 판정을 위한 거리 임계값 (픽셀)
        min_inliers = 30  # 최소 inlier 수

        N = points.shape[0]

        # num_iterations 번의 랜덤 샘플링: 각 반복마다 두 점 선택
        idx = torch.randint(0, N, (num_iterations, 2), device=device)
        # 두 점이 동일한 경우 재샘플링
        same = idx[:, 0] == idx[:, 1]
        while same.any():
            idx[same] = torch.randint(0, N, (same.sum(), 2), device=device)
            same = idx[:, 0] == idx[:, 1]

        # 각 반복에 대해 두 점 추출
        p1 = points[idx[:, 0]]  # shape: (num_iterations, 2)
        p2 = points[idx[:, 1]]  # shape: (num_iterations, 2)

        # 두 점으로부터 선의 파라미터 (a, b, c)를 계산
        a = p2[:, 1] - p1[:, 1]
        b = p1[:, 0] - p2[:, 0]
        c = p2[:, 0] * p1[:, 1] - p1[:, 0] * p2[:, 1]

        # 정규화: a^2 + b^2 = 1 (이후 거리는 |a*x + b*y + c|로 계산)
        norm = torch.sqrt(a**2 + b**2) + 1e-8
        a_norm = a / norm
        b_norm = b / norm
        c_norm = c / norm

        # --- 6. 모든 남은 점들에 대해 각 선 모델과의 거리 계산 ---
        X = points[:, 0].unsqueeze(0)  # shape: (1, N)
        Y = points[:, 1].unsqueeze(0)  # shape: (1, N)
        a_batch = a_norm.unsqueeze(1)  # shape: (num_iterations, 1)
        b_batch = b_norm.unsqueeze(1)
        c_batch = c_norm.unsqueeze(1)

        distances = torch.abs(
            a_batch * X + b_batch * Y + c_batch
        )  # shape: (num_iterations, N)
        inlier_mask = distances < distance_threshold  # shape: (num_iterations, N)
        inlier_counts = inlier_mask.sum(dim=1)  # 각 모델의 inlier 수

        # 가장 많은 inlier를 가진 모델 선택
        best_iter = torch.argmax(inlier_counts)
        best_count = inlier_counts[best_iter].item()

        if best_count < min_inliers:
            continue  # 충분한 inlier가 없으면 종료

        # 최적 모델의 선 파라미터 추출
        best_a = a_norm[best_iter].item()
        best_b = b_norm[best_iter].item()
        best_c = c_norm[best_iter].item()

        # 해당 모델의 inlier 점들 선택
        best_inliers = inlier_mask[best_iter]
        inlier_points = points[best_inliers]

        # 검출된 선분을 저장
        if inlier_points.shape[0] > 1:
            detected_lines.append((inlier_points.cpu().numpy(), (x, y)))

# --- 7. 결과 시각화: 각 윈도우에서 검출된 선분 그리기 ---
output = img.copy()

for inlier_points, (x, y) in detected_lines:
    # inlier_points: (N, 2) 배열, 각 행이 (x, y)
    for i in range(1, len(inlier_points)):
        pt1 = tuple(
            np.round(inlier_points[i - 1]).astype(int) + (x, y)
        )  # 윈도우 좌표 보정
        pt2 = tuple(np.round(inlier_points[i]).astype(int) + (x, y))  # 윈도우 좌표 보정
        cv2.line(output, pt1, pt2, (0, 0, 255), 2)

plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()
