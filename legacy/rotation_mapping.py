import matplotlib.pyplot as plt
import numpy as np

# 직선 방정식 그리기
def plot_line(ax, x1, y1, x2, y2, label):
    x_vals = np.array([x1, x2])
    y_vals = np.array([y1, y2])
    ax.plot(x_vals, y_vals, label=label)

# 기울기 계산 (기존)
def calculate_slope(x1, y1, x2, y2):
    """ 두 점 사이의 기울기 계산 """
    if x2 - x1 == 0:  # 수직선인 경우
        return None  # 기울기는 무한대
    return (y2 - y1) / (x2 - x1)

def calculate_distance_from_line(x_p, y_p, x1, y1, x2, y2):
    """ 점 (x_p, y_p)에서 직선 (x1, y1) ~ (x2, y2)까지의 수직 거리 계산 """
    m = calculate_slope(x1, y1, x2, y2)
    if m is None:  # 수직선일 경우
        return abs(x_p - x1)
    b = y1 - m * x1
    return abs(m * x_p - y_p + b) / (m**2 + 1)**0.5

def calculate_interpolated_slope(x_p, y_p, top_line, bottom_line):
    """ 윗변과 아랫변의 기울기와 점에서 각 변까지의 거리로 내분된 기울기 계산 """
    m_top = calculate_slope(*top_line)
    m_bottom = calculate_slope(*bottom_line)
    
    d_top = calculate_distance_from_line(x_p, y_p, *top_line)
    d_bottom = calculate_distance_from_line(x_p, y_p, *bottom_line)
    
    if (d_top + d_bottom) != 0:
        interpolated_slope = (m_top * d_bottom + m_bottom * d_top) / (d_top + d_bottom)
        return interpolated_slope
    else:
        return None

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
    m_left = calculate_vertical_slope(*left_line)
    m_right = calculate_vertical_slope(*right_line)
    
    d_left = calculate_distance_from_vertical_line(x_p, y_p, *left_line)
    d_right = calculate_distance_from_vertical_line(x_p, y_p, *right_line)
    
    if (d_left + d_right) != 0:
        interpolated_slope = (m_left * d_right + m_right * d_left) / (d_left + d_right)
        return interpolated_slope
    else:
        return None

# 새로운 사각형 좌표
rectangle_points = [(2, 2), (6, 1), (5, 6), (1, 5)]

# 윗변과 아랫변의 좌표
top_line = (2, 2, 6, 1)  # 윗변: (x1, y1) -> (x2, y2)
bottom_line = (1, 5, 5, 6)  # 아랫변: (x1, y1) -> (x2, y2)

# 왼쪽변과 오른쪽변의 좌표
left_line = (2, 2, 1, 5)  # 왼쪽변: (x1, y1) -> (x2, y2)
right_line = (6, 1, 5, 6)  # 오른쪽변: (x1, y1) -> (x2, y2)

# 점 (x_p, y_p)
point_inside = (3, 5)

# 내분된 기울기 계산
interpolated_slope_top_bottom = calculate_interpolated_slope(point_inside[0], point_inside[1], top_line, bottom_line)
interpolated_slope_left_right = calculate_interpolated_vertical_slope(point_inside[0], point_inside[1], left_line, right_line)

# 내분된 기울기를 이용해 직선 방정식 구하기
m_interpolated_top_bottom = interpolated_slope_top_bottom
b_interpolated_top_bottom = point_inside[1] - m_interpolated_top_bottom * point_inside[0]

m_interpolated_left_right = interpolated_slope_left_right
b_interpolated_left_right = point_inside[0] - m_interpolated_left_right * point_inside[1]

# 두 기울기 합 계산
sum_of_slopes = (m_interpolated_top_bottom - m_interpolated_left_right)/2

# 합한 기울기 직선 방정식 구하기
# 직선 방정식: y = m_sum * x + b_sum
m_sum = sum_of_slopes
b_sum = point_inside[1] - m_sum * point_inside[0]

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
ax.plot(point_inside[0], point_inside[1], 'ro', label="점 (3, 5)")

# 내분된 기울기를 이용한 직선 그리기 (y = mx + b)
x_vals = np.array([0, 7])  # x 값 범위 설정 (0부터 7까지)
y_vals_top_bottom = m_interpolated_top_bottom * x_vals + b_interpolated_top_bottom  # 직선 방정식으로 y 값 계산
ax.plot(x_vals, y_vals_top_bottom, 'g-', label=f"기울기 (윗변/아랫변): {m_interpolated_top_bottom:.2f}")

# 내분된 기울기를 이용한 직선 그리기 (x = my + b)
y_vals_left_right = np.array([0, 7])  # y 값 범위 설정 (0부터 7까지)
x_vals_left_right = m_interpolated_left_right * y_vals_left_right + b_interpolated_left_right  # 직선 방정식으로 x 값 계산
ax.plot(x_vals_left_right, y_vals_left_right, 'b-', label=f"기울기 (왼쪽변/오른쪽변): {m_interpolated_left_right:.2f}")

# 합한 기울기를 이용한 직선 그리기 (y = mx + b)
y_vals_sum = m_sum * x_vals + b_sum
ax.plot(x_vals, y_vals_sum, 'm--', label=f"합한 기울기: {m_sum:.2f}")

# 내분된 기울기 합을 텍스트로 추가
ax.text(0.5, 4.5, f"기울기 합: {sum_of_slopes:.2f}", fontsize=12, color="purple", ha="center")

# 내분된 기울기를 텍스트로 추가
ax.text(0.5, 6.2, f"윗변-아랫변 기울기: {m_interpolated_top_bottom:.2f}", fontsize=12, color="blue", ha="center")
ax.text(0.5, 5.8, f"왼쪽변-오른쪽변 기울기: {m_interpolated_left_right:.2f}", fontsize=12, color="red", ha="center")

# 라벨 추가
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.legend()

plt.grid(True)
plt.show()
