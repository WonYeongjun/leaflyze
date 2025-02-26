import math


def merge_collinear_segments(segments):
    groups = {}

    # 1. 각 선분에 대해 방향 벡터와 직선 고유의 키(방향, offset)를 구함
    for x1, y1, x2, y2 in segments:
        dx = x2 - x1
        dy = y2 - y1
        norm = math.hypot(dx, dy)
        if norm < 1e-8:  # 길이가 0인 경우는 무시
            continue
        dx /= norm
        dy /= norm
        # 방향의 표준화를 위해, dx가 음수이거나 dx가 0인데 dy가 음수면 반전
        if dx < 0 or (abs(dx) < 1e-8 and dy < 0):
            dx, dy = -dx, -dy
        # 선분이 속한 직선을 표현하는 상수: 모든 점 (x,y)에 대해 -dy*x + dx*y = c가 일정함
        c = -dy * x1 + dx * y1
        key = (round(dx, 6), round(dy, 6), round(c, 6))
        groups.setdefault(key, []).append((x1, y1, x2, y2))

    merged_segments = []

    # 2. 같은 직선 그룹별로 선분 병합
    for key, group in groups.items():
        dx, dy, c = key
        # 그룹 내의 기준점 p0 (여기서는 첫 번째 선분의 시작점을 사용)
        p0 = group[0][:2]
        intervals = []
        # 각 선분의 두 끝점을 p0 기준으로 투영하여 구간 [t_min, t_max]로 나타냄
        for x1, y1, x2, y2 in group:
            t1 = (x1 - p0[0]) * dx + (y1 - p0[1]) * dy
            t2 = (x2 - p0[0]) * dx + (y2 - p0[1]) * dy
            intervals.append((min(t1, t2), max(t1, t2)))

        # 투영된 구간을 t값 기준으로 정렬 후 병합
        intervals.sort(key=lambda iv: iv[0])
        merged_intervals = []
        current_start, current_end = intervals[0]
        for start, end in intervals[1:]:
            if start <= current_end:  # 구간이 겹치면 확장
                current_end = max(current_end, end)
            else:
                merged_intervals.append((current_start, current_end))
                current_start, current_end = start, end
        merged_intervals.append((current_start, current_end))

        # 3. 병합된 t 구간을 다시 좌표로 변환
        for t_start, t_end in merged_intervals:
            start_point = (p0[0] + t_start * dx, p0[1] + t_start * dy)
            end_point = (p0[0] + t_end * dx, p0[1] + t_end * dy)
            merged_segments.append(
                [start_point[0], start_point[1], end_point[0], end_point[1]]
            )

    return merged_segments


# 예시 사용
segments = [
    [0, 0, 1, 1],
    [1, 1, 2, 2],
    [2, 2, 3, 3],
    [0, 1, 1, 2],  # 약간 다른 직선에 속한 예시
]

result = merge_collinear_segments(segments)
print(result)
