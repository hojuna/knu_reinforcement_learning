import numpy as np
from collections import deque
from knu_rl_env.grid_adventure import GridAdventureAgent, make_grid_adventure, evaluate

class GridAdventureRLAgent(GridAdventureAgent):
    def __init__(self):
        # 가능한 행동들 정의 (기본 행동)
        self.ACTION_LEFT = GridAdventureAgent.ACTION_LEFT
        self.ACTION_RIGHT = GridAdventureAgent.ACTION_RIGHT
        self.ACTION_FORWARD = GridAdventureAgent.ACTION_FORWARD
        self.ACTION_PICKUP = GridAdventureAgent.ACTION_PICKUP
        self.ACTION_DROP = GridAdventureAgent.ACTION_DROP
        self.ACTION_UNLOCK = GridAdventureAgent.ACTION_UNLOCK

        # 복합 행동 정의
        self.knu_agent_left = [self.ACTION_LEFT, self.ACTION_FORWARD]
        self.knu_agent_right = [self.ACTION_RIGHT, self.ACTION_FORWARD]
        self.knu_agent_forward = [self.ACTION_FORWARD]
        self.knu_agent_pickup = [self.ACTION_PICKUP]
        self.knu_agent_unlock = [
            self.ACTION_UNLOCK,
            self.ACTION_LEFT,
            self.ACTION_LEFT,
            self.ACTION_DROP,
            self.ACTION_LEFT,
            self.ACTION_LEFT,
            self.ACTION_FORWARD
        ]
        self.knu_agent_actions = [
            self.knu_agent_left,
            self.knu_agent_right,
            self.knu_agent_forward,
            self.knu_agent_pickup,
            self.knu_agent_unlock
        ]
        # 상태 공간 크기 정의
        self.state_size = (26, 26)
        # 복합 행동 공간 크기 정의
        self.action_size = len(self.knu_agent_actions)
        # 학습률(alpha)
        self.alpha = 0.1
        # 할인율(gamma)
        self.gamma = 0.99
        # 탐색률(epsilon)
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        # 정책 테이블 초기화 (Q-테이블)
        self.q_table = np.zeros(self.state_size + (self.action_size,))
        # BFS 테이블 초기화
        self.bfs_table_flag = False
        self.bfs_table = np.zeros(self.state_size)
        # 목표 리스트 (예: 열쇠와 문의 위치)
        self.goal_list = []
    
    def find_agent_position(self, observation):
        # 관찰에서 에이전트의 위치를 찾아 반환
        for i in range(self.state_size[0]):
            for j in range(self.state_size[1]):
                if observation[i][j] in ['AL', 'AR', 'AU', 'AD']:
                    return (i, j)
        return None  # 에이전트 위치를 찾지 못한 경우

    def get_next_position(self, position, action_sequence):
        # 현재 위치에서 주어진 행동 시퀀스를 수행했을 때의 최종 위치를 반환
        # 이 함수는 실제 환경의 동작을 시뮬레이션해야 하지만,
        # 여기서는 간단하게 현재 위치를 반환하도록 합니다.
        return position  # 실제 구현 필요

    def find_current_goal(self):
        # 현재 목표를 반환하는 함수 (예: 가장 가까운 열쇠나 문)
        # 간단히 목표를 (25, 25)로 설정
        return (25, 25)

    def update_goal_flag(self, goal):
        # 목표를 달성했을 때 호출되는 함수
        # 목표 리스트에서 해당 목표를 제거하거나 플래그를 업데이트함
        pass

    def custom_reward(self, current_position, action_index):
        # BFS 테이블에 기반한 이동 보상 계산
        action_sequence = self.knu_agent_actions[action_index]
        next_position = self.get_next_position(current_position, action_sequence)
        goal = self.find_current_goal()

        # 이동 보상은 BFS 테이블 값으로 설정 (값이 높을수록 목표에 가까움)
        move_reward = self.bfs_table[next_position[0], next_position[1]]

        # 목표 보상 초기화
        goal_reward = 0

        # 목표에 도달했는지 확인
        if next_position == goal:
            goal_reward = 100  # 목표 보상 (예시로 100 설정)
            self.update_goal_flag(goal)
            self.bfs_table_flag = True  # 목표 달성 시 플래그 변경

        # 총 보상 계산
        total_reward = move_reward + goal_reward
        return total_reward

    def act(self, observation):
        position = self.find_agent_position(observation)
        if position is None:
            return self.knu_agent_forward  # 기본적으로 앞으로 가기
        if np.random.rand() < self.epsilon:
            # 무작위 행동 선택 (탐험)
            action_index = np.random.choice(self.action_size)
        else:
            # 정책에 따른 최적 행동 선택 (활용)
            state_actions = self.q_table[position[0], position[1], :]
            action_index = np.argmax(state_actions)
        action_sequence = self.knu_agent_actions[action_index]
        return action_sequence

    def update_q_table(self, state, action_index, reward, next_state, done):
        x, y = state
        next_x, next_y = next_state
        current_q = self.q_table[x, y, action_index]
        if done:
            target = reward
        else:
            next_max_q = np.max(self.q_table[next_x, next_y, :])
            target = reward + self.gamma * next_max_q
        # Q-러닝 업데이트 공식
        self.q_table[x, y, action_index] += self.alpha * (target - current_q)

    def bfs_shortest_path(self, grid, start, goal):
        # 그리드의 크기
        rows, cols = len(grid), len(grid[0])
        
        # 방향 벡터 (상, 하, 좌, 우)
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        # 방문 여부를 기록하는 배열
        visited = [[False for _ in range(cols)] for _ in range(rows)]
        
        # BFS를 위한 큐 초기화
        queue = deque([(start, 0)])  # (현재 위치, 현재까지의 거리)
        visited[start[0]][start[1]] = True
        
        while queue:
            (current, distance) = queue.popleft()
            
            # 목표에 도달하면 거리 반환
            if current == goal:
                return distance
            
            # 현재 위치에서 가능한 모든 방향으로 이동
            for direction in directions:
                new_row, new_col = current[0] + direction[0], current[1] + direction[1]
                
                # 그리드 범위 내에 있고, 방문하지 않았으며, 이동 가능한 경우
                if (0 <= new_row < rows and 0 <= new_col < cols and
                    not visited[new_row][new_col] and grid[new_row][new_col] != 'W'):
                    visited[new_row][new_col] = True
                    queue.append(((new_row, new_col), distance + 1))
        
        # 목표에 도달할 수 없는 경우
        return float('inf')

    def update_bfs_table(self, grid, goal):
        # BFS 테이블 업데이트 로직
        for i in range(26):
            for j in range(26):
                if grid[i][j] not in ['W', 'L']:  # 벽이나 용암이 아닌 경우에만 업데이트
                    distance = self.bfs_shortest_path(grid, (i, j), goal)
                    if distance != float('inf'):
                        self.bfs_table[i, j] = 1 / distance
                    else:
                        self.bfs_table[i, j] = 0  # 도달 불가능한 경우 0으로 설정
        return self.bfs_table

def train():
    env = make_grid_adventure(show_screen=True)
    agent = GridAdventureRLAgent()
    num_episodes = 1000  # 학습 에피소드 수

    for episode in range(num_episodes):
        observation,_ = env.reset()
        state = agent.find_agent_position(observation)
        done = False
        total_reward = 0

        if not agent.bfs_table_flag:
            # BFS 테이블 업데이트
            grid = observation  # 환경의 그리드를 얻어옴
            goal = agent.find_current_goal()
            agent.update_bfs_table(grid, goal)
            agent.bfs_table_flag = True

        while not done:
            action_sequence = agent.act(observation)
            for action in action_sequence:
                observation_next, env_reward, terminated, truncated, info = env.step(action)
                next_state = agent.find_agent_position(observation_next)

                # 위치 정보를 찾을 수 없는 경우 처리
                if state is None or next_state is None:
                    done = True
                    break

                # 커스텀 보상 함수 적용
                action_index = agent.knu_agent_actions.index(action_sequence)
                reward = agent.custom_reward(state, action_index)
                agent.update_q_table(state, action_index, reward, next_state, done)
                total_reward += reward
                observation = observation_next
                state = next_state

                if done:
                    break

            if done:
                # 탐색률 감소
                if agent.epsilon > agent.epsilon_min:
                    agent.epsilon *= agent.epsilon_decay
                print(f"Episode: {episode+1}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.4f}")
                break

    # 학습된 에이전트 반환
    return agent

if __name__ == '__main__':
    trained_agent = train()
    evaluate(trained_agent)
