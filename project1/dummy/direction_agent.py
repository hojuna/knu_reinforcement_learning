import numpy as np
import random
import os
from knu_rl_env.grid_adventure import GridAdventureAgent, make_grid_adventure, evaluate
from collections import deque

# 상태 매핑 (문자 -> 숫자)
state_mapping = {
    'E': 0, 'W': 1, 'L': 2, 'G': 3, 'AL': 4, 'AR': 5, 'AU': 6, 'AD': 7, 
    'DBL': 8, 'DBC': 9, 'DBO': 10, 'DGL': 11, 'DGC': 12, 'DGO': 13, 'DRL': 14,
    'DRC': 15, 'DRO': 16, 'KB': 17, 'KG': 18, 'KR': 19
}

direction_mapping = {
    'AL': 0,  # 좌측
    'AR': 1,  # 우측
    'AU': 2,  # 위
    'AD': 3   # 아래
}

def discretize_state(grid, position):
    if isinstance(grid, dict):
        grid = grid['grid']
    x, y = position
    # 상태 인덱스 계산 시 그리드의 크기를 고려
    return x * len(grid[0]) + y

def find_agent_position_and_direction(grid):
    if isinstance(grid, dict):
        grid = grid['grid']
    for x in range(len(grid)):
        for y in range(len(grid[x])):
            if grid[x][y] in direction_mapping:
                return (x, y), direction_mapping[grid[x][y]]
    return None, None

def get_front_position(position, direction):
    x, y = position
    if direction == 0:
        return x, y - 1
    elif direction == 1:
        return x, y + 1
    elif direction == 2:
        return x - 1, y
    elif direction == 3:
        return x + 1, y

def find_goal_position(grid, goal_type):
    for x in range(len(grid)):
        for y in range(len(grid[x])):
            if grid[x][y] == goal_type:
                return (x, y)
    return None

def bfs_shortest_path(grid, start, goal):
    queue = deque([(start, 0)])
    visited = set()
    visited.add(start)

    while queue:
        (x, y), distance = queue.popleft()

        if (x, y) == goal:
            return distance

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < len(grid) and 0 <= ny < len(grid[0]) and (nx, ny) not in visited:
                if grid[nx][ny] not in ['W', 'L']:  # 벽이나 용암이 아닌 경우에만 이동 가능
                    queue.append(((nx, ny), distance + 1))
                    visited.add((nx, ny))

    return float('inf')  # 목표에 도달할 수 없는 경우

class GridAdventureRLAgent(GridAdventureAgent):
    possible_actions = [
        GridAdventureAgent.ACTION_LEFT,
        GridAdventureAgent.ACTION_RIGHT,
        GridAdventureAgent.ACTION_FORWARD,
        GridAdventureAgent.ACTION_PICKUP,
        GridAdventureAgent.ACTION_DROP,  # 드랍 추가
        GridAdventureAgent.ACTION_UNLOCK
    ]

    def __init__(self, q_table_path='direction_q_table.npy'):
        super().__init__()

        # 실제 그리드의 크기를 확인하고 설정
        grid_width = 26  # 실제 그리드의 가로 크기로 수정
        grid_height = 26  # 실제 그리드의 세로 크기로 수정
        state_space_size = grid_width * grid_height  # 위치만 고려한 상태 공간 크기
        action_space_size = len(self.possible_actions)

        self.q_table_path = q_table_path
        if os.path.exists(self.q_table_path):
            self.q_table = np.load(self.q_table_path)
        else:
            self.q_table = np.random.uniform(low=-1, high=1, size=(state_space_size, action_space_size))
        
        self.epsilon = 0.5
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.alpha = 0.1
        self.gamma = 0.9
        self.carrying = set()
        self.visit_count = {}
        self.initial_distance_to_goal = None

    def act(self, observation):
        grid = observation
        if isinstance(grid, dict):
            grid = grid['grid']

        position, direction = find_agent_position_and_direction(grid)

        if position is None:
            raise ValueError("Error: Agent position not found in the grid")

        state_idx = discretize_state(grid, position)

        if random.uniform(0, 1) < self.epsilon:
            action = random.choice(self.possible_actions)
        else:
            q_values = self.q_table[state_idx]
            # print(f"State Index: {state_idx}, Q-Values: {q_values}")

            if q_values.ndim == 1 and q_values.size > 0:
                best_action_index = np.argmax(q_values)
                action = self.possible_actions[best_action_index]
            else:
                action = random.choice(self.possible_actions)

        if action == GridAdventureAgent.ACTION_LEFT:
            direction = (direction - 1) % 4
        elif action == GridAdventureAgent.ACTION_RIGHT:
            direction = (direction + 1) % 4

        if grid[position[0]][position[1]] in ['KB', 'KG', 'KR']:
            self.carrying.add(grid[position[0]][position[1]])

        # 드랍 행동 처리
        if action == GridAdventureAgent.ACTION_DROP:
            if self.carrying:
                self.carrying.clear()  # 모든 열쇠를 드랍

        return action

    def update_q_table(self, grid, position, action, reward, next_grid, next_position):
        state_idx = discretize_state(grid, position)
        next_state_idx = discretize_state(next_grid, next_position)

        # 상태 인덱스가 Q-테이블의 크기를 초과하지 않도록 확인
        if state_idx >= self.q_table.shape[0] or next_state_idx >= self.q_table.shape[0]:
            print(f"Warning: State index out of bounds. State index: {state_idx}, Next state index: {next_state_idx}")
            return

        action_index = self.possible_actions.index(action)
        next_max = np.max(self.q_table[next_state_idx])

        # Q-테이블 업데이트
        self.q_table[state_idx][action_index] = (1 - self.alpha) * self.q_table[state_idx][action_index] + self.alpha * (reward + self.gamma * next_max)

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_q_table(self):
        np.save(self.q_table_path, self.q_table)

    def calculate_reward(self, grid, position, direction, terminated, truncated, previous_state_action, current_state_action):
        if isinstance(grid, dict):
            grid = grid['grid']

        x, y = position
        state = grid[x][y]
        _, action = current_state_action

        reward = 0

        # 단계적 목표 설정
        goals = ['KB', 'DBO', 'KG', 'DGO', 'KR', 'DRO', 'G']
        current_goal_index = 0

        if terminated:
            if state == 'G':
                reward = 100
            else:
                reward = 0
        else:
            reward = 0.1

        # 현재 목표에 도달했는지 확인
        if state == goals[current_goal_index]:
            reward += 20  # 목표 달성 시 보상
            current_goal_index += 1  # 다음 목표로 이동

        if action == GridAdventureAgent.ACTION_FORWARD:
            reward += 1

            # 포워드 행동을 선택했을 때 앞에 벽이나 용암이 있는지 확인
            front_position = get_front_position(position, direction)
            if (0 <= front_position[0] < len(grid) and 0 <= front_position[1] < len(grid[0])):
                front_state = grid[front_position[0]][front_position[1]]
                if front_state in ['W', 'L']:  # 벽이나 용암에 부딪히면 마이너스 리드
                    reward -= 5

        # 열쇠가 없는 곳에서 열쇠를 집는 행동에 대한 마이너스 보상
        if action == GridAdventureAgent.ACTION_PICKUP and state not in ['KB', 'KG', 'KR']:
            reward -= 10

        # 문이 없는 곳에서 문을 여는 행동에 대한 마이너스 보상
        if action == GridAdventureAgent.ACTION_UNLOCK and state not in ['DBO', 'DGO', 'DRO']:
            reward -= 10

        # 잘못된 열쇠로 문을 여는 행동에 대한 마이너스 보상
        if action == GridAdventureAgent.ACTION_UNLOCK:
            if (state == 'DBO' and 'KB' not in self.carrying) or \
               (state == 'DGO' and 'KG' not in self.carrying) or \
               (state == 'DRO' and 'KR' not in self.carrying):
                reward -= 10

        # 열쇠를 버리는 행동에 대한 마이너스 보상
        if action == GridAdventureAgent.ACTION_DROP:
            reward -= 5

        if previous_state_action == current_state_action:
            reward -= 1

        # 같은 위치에서 반복적으로 행동을 취하는 경우 마이너스 리워드 추가
        if previous_state_action and previous_state_action[0] == current_state_action[0]:
            reward -= 0.5

        # 목표에 가까워질 때 추가적인 보상
        goal_position = find_goal_position(grid, goals[current_goal_index])
        if goal_position:
            current_distance_to_goal = bfs_shortest_path(grid, position, goal_position)

            # 처음 목표까지의 거리를 저장
            if self.initial_distance_to_goal is None:
                self.initial_distance_to_goal = current_distance_to_goal

            # 목표에 도달했을 때의 보상을 크게 설정
            if current_distance_to_goal == 0:
                reward += 50  # 목표에 도달했을 때 큰 보상
                self.initial_distance_to_goal = None  # 목표에 도달했으므로 초기 거리 초기화
            else:
                # 목표에 가까워질수록 보상을 증가
                distance_improvement = self.initial_distance_to_goal - current_distance_to_goal
                max_possible_improvement = self.initial_distance_to_goal
                improvement_ratio = distance_improvement / max_possible_improvement if max_possible_improvement > 0 else 0
                reward += improvement_ratio * 15  # 비율적으로 (0, 10) 범위의 보상

        return reward



def train():
    def evn_step(action):
        if isinstance(action, list):
            # action이 리스트일 때의 로직
            for i in action:
                temp= env.step(i)
            print("List action:", action)
            return temp
        
        elif isinstance(action, int):
            # action이 정수일 때의 로직
            print("Integer action:", action)
            return env.step(i)
        
        else:
            raise ValueError("Unsupported action type")
    

    env = make_grid_adventure(
            show_screen=True
            # show_screen=False
        )
    
    agent = GridAdventureRLAgent()
    
    num_episodes = 1000
    max_steps = 5000

    for episode in range(num_episodes):
        grid, _ = env.reset()
        position, direction = find_agent_position_and_direction(grid)
        if position is None:
            print("Error: Agent position not found in the grid")
            break

        total_reward = 0
        previous_state_action = None
        goals_achieved = 0

        for step in range(max_steps):
            observation = {'grid': grid}
            action = agent.act(observation)

            next_grid, _, terminated, truncated, info = env.step(action)
            next_position, next_direction = find_agent_position_and_direction(next_grid)

            current_state_action = (discretize_state(grid, position), action)
            reward = agent.calculate_reward(grid, position, direction, terminated, truncated, previous_state_action, current_state_action)

            agent.update_q_table(grid, position, action, reward, next_grid, next_position)
            
            grid = next_grid
            position = next_position
            direction = next_direction
            previous_state_action = current_state_action
            total_reward += reward

            if terminated or truncated:
                break

            # 목표 달성 여부 확인
            if reward >= 20:  # 목표 달성 시 보상 기준
                goals_achieved += 1

        agent.decay_epsilon()

        # 에피소드가 끝날 때마다 Q-테이블 저장
        agent.save_q_table()

        print(f"Episode {episode + 1}: Total Reward: {total_reward}, Goals Achieved: {goals_achieved}, Epsilon: {agent.epsilon:.4f}")

    return agent

if __name__ == '__main__':
    trained_agent = train()
    evaluate(trained_agent)
