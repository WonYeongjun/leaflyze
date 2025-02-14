import matplotlib.pyplot as plt
import numpy as np

# 직선 방정식 그리기
def plot_line(ax, x1, y1, x2, y2, label):
    x_vals = np.array([x1, x2])
    y_vals = np.array([y1, y2])
    ax.plot(x_vals, y_vals, label=label)

# 기울기 계산 (세로방향 기울기)
def calculate_vertical_slope(x1, y1, x2, y2):
    """ 세로 방향 기준 기울기 계산 """
    if y2 - y1 == 0:  # 수평선인 경우
        return None  # 기울기는 무한대
    return (x2 - x1) / (y2 - y1)

def calculate_distance_from_vertical_line(x_p, y_p, x1, y1, x2, y2):
    """ 점 (x_p, y_p)에서 세로방향 직선 (x1, y1) ~ (x2, y2)까지의 수직 거리 계산 """
    m = calculate_vertical_slope(x1, y1, x2, y2)
    if m is None:  # 수평선일 경우
        return abs(y_p - y1)
    b = y1 - m * x1
    return abs(m * x_p - y_p + b) / (m**2 + 1)**0.5

def calculate_interpolated_vertical_slope(x_p, y_p, left_line, right_line):
    """ 왼쪽변과 오른쪽변의 기울기와 점에서 각 변까지의 거리로 내분된 기울기 계산 """
    # 왼쪽변과 오른쪽변의 기울기 구하기
    m_left = calculate_vertical_slope(*left_line)
    m_right = calculate_vertical_slope(*right_line)
    
    # 점에서 왼쪽변과 오른쪽변까지의 거리 구하기
    d_left = calculate_distance_from_vertical_line(x_p, y_p, *left_line)
    d_right = calculate_distance_from_vertical_line(x_p, y_p, *right_line)
    
    # 내분된 기울기 계산
    if (d_left + d_right) != 0:
        interpolated_slope = (m_left * d_right + m_right * d_left) / (d_left + d_right)
        return interpolated_slope
    else:
        return None

# 새로운 사각형 좌표
rectangle_points = [(2, 2), (6, 1), (5, 6), (1, 5)]

# 왼쪽변과 오른쪽변의 좌표
left_line = (2, 2, 1, 5)  # 왼쪽변: (x1, y1) -> (x2, y2)
right_line = (6, 1, 5, 6)  # 오른쪽변: (x1, y1) -> (x2, y2)

# 점 (x_p, y_p)
point_inside = (1, 4)

# 내분된 기울기 계산
interpolated_slope = calculate_interpolated_vertical_slope(point_inside[0], point_inside[1], left_line, right_line)

# 내분된 기울기를 이용해 직선 방정식 구하기 (세로 방향 기준)
# 직선의 방정식은 x = my + b 이므로, b를 구해야 합니다.
m_interpolated = interpolated_slope
b_interpolated = point_inside[0] - m_interpolated * point_inside[1]

# 그래프 그리기
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(0, 7)
ax.set_ylim(0, 7)

# 사각형 그리기
plot_line(ax, *rectangle_points[0], *rectangle_points[1], label="윗변")
plot_line(ax, *rectangle_points[1], *rectangle_points[2], label="오른쪽변")
plot_line(ax, *rectangle_points[2], *rectangle_points[3], label="아랫변")
plot_line(ax, *rectangle_points[3], *rectangle_points[0], label="왼쪽변")

# 점 그리기
ax.plot(point_inside[0], point_inside[1], 'ro', label="점 (3, 4)")

# 내분된 기울기를 이용한 직선 그리기 (x = my + b)
y_vals = np.array([0, 7])  # y 값 범위 설정 (0부터 7까지)
x_vals = m_interpolated * y_vals + b_interpolated  # 직선 방정식으로 x 값 계산
ax.plot(x_vals, y_vals, 'g-', label=f"기울기: {m_interpolated:.2f}")

# 내분된 기울기를 텍스트로 추가
ax.text(0.5, 6.2, f"내분된 기울기: {m_interpolated:.2f}", fontsize=12, color="blue", ha="center")

# 라벨 추가
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.legend()

plt.grid(True)
plt.show()
