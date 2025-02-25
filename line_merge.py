import numpy as np
from scipy.spatial import KDTree
from collections import defaultdict


def calculate_angle(x1, y1, x2, y2):
    """두 점을 연결하는 선의 각도(기울기) 계산 (라디안 -> 도)"""
    return np.degrees(np.arctan2(y2 - y1, x2 - x1))


def create_kd_tree(points):
    """KDTree 생성"""
    return KDTree(points)


def find_connected_segments(segments, distance_threshold=30, angle_threshold=10.0):
    """선분 간 거리와 각도를 고려하여 연결된 선분 찾기"""
    points = []
    segment_info = {}

    for i, (x1, y1, x2, y2) in enumerate(segments):
        angle = calculate_angle(x1, y1, x2, y2)
        points.append((x1, y1))
        points.append((x2, y2))
        segment_info[(x1, y1)] = (i, angle)
        segment_info[(x2, y2)] = (i, angle)

    kd_tree = create_kd_tree(points)
    adj_list = defaultdict(list)

    for i, (x1, y1, x2, y2) in enumerate(segments):
        angle_i = calculate_angle(x1, y1, x2, y2)

        # 🔥 시점과 가까운 점 찾기
        distances, indices = kd_tree.query([x1, y1], k=5)
        for idx in indices:
            if idx >= len(points):
                continue
            x_near, y_near = points[idx]
            if (x_near, y_near) in segment_info:
                neighbor, angle_j = segment_info[(x_near, y_near)]
                angle_new = calculate_angle(x1, y1, x_near, y_near)
                r = np.linalg.norm([x1 - x_near, y1 - y_near])
                # ✅ 거리 & 각도 차이 조건 추가
                if (
                    neighbor != i
                    and r < distance_threshold
                    and abs(angle_i - angle_new) < 3 * angle_threshold
                ):
                    # print(angle_i, angle_j, angle_new, abs(angle_i - angle_j))
                    if abs(angle_i - angle_j) < angle_threshold:
                        adj_list[i].append(neighbor)

        # 🔥 종점과 가까운 점 찾기
        distances, indices = kd_tree.query([x2, y2], k=5)
        for idx in indices:
            if idx >= len(points):
                continue
            x_near, y_near = points[idx]
            if (x_near, y_near) in segment_info:
                neighbor, angle_j = segment_info[(x_near, y_near)]
                angle_new = calculate_angle(x2, y2, x_near, y_near)
                r = np.linalg.norm([x2 - x_near, y2 - y_near])
                if (
                    neighbor != i
                    and r < distance_threshold
                    and abs(angle_i - angle_new) < 3 * angle_threshold
                ):
                    # print(angle_i, angle_j, angle_new, abs(angle_i - angle_j))
                    if abs(angle_i - angle_j) < angle_threshold:
                        adj_list[i].append(neighbor)

    return adj_list


def dfs(idx, adj_list, visited, group):
    if visited[idx]:  # 🔥 이미 방문했으면 바로 리턴
        return
    """DFS로 연결된 선분들을 그룹화"""
    visited[idx] = True
    group.append(idx)

    for neighbor in adj_list[idx]:
        if 0 <= neighbor < len(visited) and not visited[neighbor]:  # 유효성 체크
            dfs(neighbor, adj_list, visited, group)


def merge_segments(segments, adj_list):
    """연결된 선분 그룹을 찾아 병합"""
    visited = [False] * len(segments)
    result = []

    for i in range(len(segments)):
        if not visited[i]:
            group = []
            dfs(i, adj_list, visited, group)
            if len(group) == 1:
                continue
            merged_segment = merge_group(segments, group)
            result.append(merged_segment)

    return result


def merge_group(segments, group):
    """그룹화된 선분을 하나의 긴 선분으로 병합"""
    group_points = []
    for idx in group:
        x1, y1, x2, y2 = segments[idx]
        group_points.append((x1, y1))
        group_points.append((x2, y2))

    # x 또는 y 기준으로 정렬
    group_points = sorted(group_points, key=lambda p: (p[0], p[1]))

    return [
        group_points[0][0],
        group_points[0][1],
        group_points[-1][0],
        group_points[-1][1],
    ]


def merge_all_segments(segments):
    """전체 선분을 병합"""
    adj_list = find_connected_segments(segments)
    merged_segments = merge_segments(segments, adj_list)
    return merged_segments


if __name__ == "__main__":
    # 테스트 데이터
    segments = [
        [0, 0, 1, 1],
        [1, 1, 2, 2],
        [2, 2, 3, 3],
        [4, 4, 5, 5],  # 이 선분은 독립적
        [10, 10, 11, 11],
    ]

    merged_segments = merge_all_segments(segments)
    print(merged_segments)
