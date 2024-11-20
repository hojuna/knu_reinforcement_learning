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

def discretize_state(grid, position):
    if isinstance(grid, dict):
        grid = grid['grid']
    x, y = position
    state_symbol = grid[x][y]
    return state_mapping.get(state_symbol, 0)

def find_agent_position(grid):
    if isinstance(grid, dict):
        grid = grid['grid']
    for x in range(len(grid)):
        for y in range(len(grid[x])):
            if grid[x][y] in ['AR', 'AL', 'AU', 'AD']:
                return (x, y)
    return None

def get_front_position(position, direction):
    x, y = position
    if direction == 0:
        return x, y + 1
    elif direction == 1:
        return x + 1, y
    elif direction == 2:
        return x, y - 1
    elif direction == 3:
        return x - 1, y

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
    def __init__(self, q_table_path='q_table.npy'):
        super().__init__()
        
        # 상태 공간과 행동 공간의 크기 설정
        state_space_size = len(state_mapping)
        action_space_size = 5

        self.q_table_path = q_table_path
        if os.path.exists(self.q_table_path):
            self.q_table = np.load(self.q_table_path)
        else:
            self.q_table = np.random.uniform(low=-1, high=1, size=(state_space_size, action_space_size))
        
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.alpha = 0.1
        self.gamma = 0.9
        self.carrying = set()
        self.agent_dir = 0
        self.visit_count = {}
        self.initial_distance_to_goal = None

    def act(self, observation):
        grid = observation
        if isinstance(grid, dict):
            grid = grid['grid']

        position = find_agent_position(grid)

        if position is None:
            raise ValueError("Error: Agent position not found in the grid")

        state_idx = discretize_state(grid, position)

        possible_actions = [
            GridAdventureAgent.ACTION_LEFT,
            GridAdventureAgent.ACTION_RIGHT,
            GridAdventureAgent.ACTION_FORWARD,
            GridAdventureAgent.ACTION_PICKUP,
            # GridAdventureAgent.ACTION_DROP,
            GridAdventureAgent.ACTION_UNLOCK
        ]

        if random.uniform(0, 1) < self.epsilon:
            action = random.choice(possible_actions)
        else:
            q_values = self.q_table[state_idx]
            print(f"State Index: {state_idx}, Q-Values: {q_values}")

            if q_values.ndim == 1 and q_values.size > 0:
                best_action_index = np.argmax(q_values)
                action = possible_actions[best_action_index]
            else:
                action = random.choice(possible_actions)

        if action == GridAdventureAgent.ACTION_LEFT:
            self.agent_dir = (self.agent_dir - 1) % 4
        elif action == GridAdventureAgent.ACTION_RIGHT:
            self.agent_dir = (self.agent_dir + 1) % 4

        if grid[position[0]][position[1]] in ['KB', 'KG', 'KR']:
            self.carrying.add(grid[position[0]][position[1]])

        return action

    def update_q_table(self, grid, position, action, reward, next_grid, next_position):
        state_idx = discretize_state(grid, position)
        next_state_idx = discretize_state(next_grid, next_position)

        # 방문 횟수에 따른 패널티 제거
        # if state_idx not in self.visit_count:
        #     self.visit_count[state_idx] = 1
        # else:
        #     self.visit_count[state_idx] += 1

        # visit_penalty = self.visit_count[state_idx] * -1
        # adjusted_reward = reward + visit_penalty

        old_value = self.q_table[state_idx][action]
        next_max = np.max(self.q_table[next_state_idx])
        new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
        self.q_table[state_idx][action] = new_value

        print(f"State: {state_idx}, Action: {action}, Reward: {reward}, Old Q-value: {old_value}, New Q-value: {new_value}")

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_q_table(self):
        np.save(self.q_table_path, self.q_table)

    def calculate_reward(self, grid, position, terminated, truncated, previous_state_action, current_state_action):
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
            front_position = get_front_position(position, self.agent_dir)
            if (0 <= front_position[0] < len(grid) and 0 <= front_position[1] < len(grid[0])):
                front_state = grid[front_position[0]][front_position[1]]
                if front_state in ['W', 'L']:  # 벽이나 용암에 부딪히면 마이너스 리워드
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
                reward += improvement_ratio * 10  # 비율적으로 (0, 10) 범위의 보상
                # print(f"목표에 가까워짐: {improvement_ratio * 10}")

        # print(f"Position: {position}, State: {state}, Action: {action}, Reward: {reward}")

        return reward

def train():
    env = make_grid_adventure(show_screen=True)
    
    agent = GridAdventureRLAgent()
    
    num_episodes = 1000
    max_steps = 5000

    for episode in range(num_episodes):
        grid, _ = env.reset()
        position = find_agent_position(grid)
        if position is None:
            print("Error: Agent position not found in the grid")
            break

        total_reward = 0
        previous_state_action = None

        for step in range(max_steps):
            observation = {'grid': grid}
            action = agent.act(observation)

            next_grid, _, terminated, truncated, info = env.step(action)
            next_position = find_agent_position(next_grid)

            current_state_action = (discretize_state(grid, position), action)
            reward = agent.calculate_reward(grid, position, terminated, truncated, previous_state_action, current_state_action)

            agent.update_q_table(grid, position, action, reward, next_grid, next_position)
            
            grid = next_grid
            position = next_position
            previous_state_action = current_state_action
            total_reward += reward

            if terminated or truncated:
                break

        agent.decay_epsilon()

        # 에피소드가 끝날 때마다 Q-테이블 저장
        agent.save_q_table()

        print(f"Episode {episode + 1}: Total Reward: {total_reward}")

    return agent

if __name__ == '__main__':
    trained_agent = train()
    evaluate(trained_agent)
