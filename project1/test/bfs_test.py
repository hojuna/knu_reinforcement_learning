import numpy as np
import pickle
from collections import deque
from knu_rl_env.grid_adventure import GridAdventureAgent, make_grid_adventure, evaluate

def bfs_shortest_path(grid, start, goal):
    """
    BFS를 사용하여 최단 경로를 찾는 함수

    :param grid: 2D 리스트, 각 셀은 문자열로 표현 ('W', 'L', 등)
    :param start: 시작 위치 (tuple), 예: (0, 0)
    :param goal: 목표 위치 (tuple), 예: (25, 25)
    :return: 최단 경로의 리스트, 예: [(0,0), (0,1), ..., (25,25)]
    """
    rows, cols = len(grid), len(grid[0])
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 상, 하, 좌, 우
    visited = [[False for _ in range(cols)] for _ in range(rows)]
    parent = [[None for _ in range(cols)] for _ in range(rows)]
    
    queue = deque()
    queue.append(start)
    visited[start[0]][start[1]] = True

    while queue:
        current = queue.popleft()
        if current == goal:
            break
        for d in directions:
            new_row, new_col = current[0] + d[0], current[1] + d[1]
            if (0 <= new_row < rows and 0 <= new_col < cols and
                not visited[new_row][new_col] and
                grid[new_row][new_col] not in ['W', 'L']):  # 벽이나 용암은 통과 불가
                queue.append((new_row, new_col))
                visited[new_row][new_col] = True
                parent[new_row][new_col] = current

    # 경로 재구성
    path = []
    current = goal
    while current != start:
        path.append(current)
        current = parent[current[0]][current[1]]
        if current is None:
            print("경로를 찾을 수 없습니다.")
            return []
    path.append(start)
    path.reverse()
    return path

def mark_shortest_path(grid, path):
    """
    최단 경로를 기반으로 2D 배열을 생성하여 경로 셀을 1로, 나머지를 0으로 표시

    :param grid: 2D 리스트, 각 셀은 문자열로 표현
    :param path: 최단 경로의 리스트, 예: [(0,0), (0,1), ..., (25,25)]
    :return: 2D numpy 배열
    """
    rows, cols = len(grid), len(grid[0])
    path_array = np.zeros((rows, cols), dtype=int)
    for position in path:
        path_array[position[0]][position[1]] = 1
    return path_array

def print_grid(grid_array):
    """
    2D 배열을 이쁘게 출력하는 함수

    :param grid_array: 2D numpy 배열
    """
    # NumPy의 set_printoptions을 사용하여 출력 옵션 설정
    np.set_printoptions(edgeitems=30, linewidth=1000, formatter={'int': '{:2d}'.format})
    print(grid_array)



env = make_grid_adventure(show_screen=False)
grid, _ = env.reset()
path=bfs_shortest_path(grid, (1, 1), (24, 24))
path_array=mark_shortest_path(grid, path)
print_grid(path_array)
