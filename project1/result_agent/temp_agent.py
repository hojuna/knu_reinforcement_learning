import numpy as np
import pickle
from collections import deque
from knu_rl_env.grid_adventure import GridAdventureAgent, make_grid_adventure, evaluate

class GridAdventureRLAgent(GridAdventureAgent):
    def __init__(self, load_q_table=False, q_table_filename='result_agent/knu_q_table_v2_1.pkl',epsilon=0.1):
        # 가능한 행동들 정의 (기본 행동)
        self.ACTION_LEFT = GridAdventureAgent.ACTION_LEFT
        self.ACTION_RIGHT = GridAdventureAgent.ACTION_RIGHT
        self.ACTION_FORWARD = GridAdventureAgent.ACTION_FORWARD
        self.ACTION_PICKUP = GridAdventureAgent.ACTION_PICKUP
        self.ACTION_DROP = GridAdventureAgent.ACTION_DROP
        self.ACTION_UNLOCK = GridAdventureAgent.ACTION_UNLOCK

        self.knu_agent_actions = [self.ACTION_LEFT, self.ACTION_RIGHT, self.ACTION_FORWARD, self.ACTION_PICKUP, self.ACTION_UNLOCK]

        # 상태 공간 크기 정의
        self.state_size = (26, 26)
        # 행동 공간 크기 정의
        self.action_size = len(self.knu_agent_actions)

        # 방향 공간 크기 정의
        self.direction_size = 4
        self.direction_mapping = {'AU': 0, 'AR': 1, 'AD': 2, 'AL': 3} 

        # 현재 방향 초기화 (초기 설정 필요)
        self.current_direction = None

        # 학습률(alpha)
        self.alpha = 0.2
        # 할인율(gamma)
        self.gamma = 0.95
        # 탐색률(epsilon)
        self.epsilon = epsilon
        self.epsilon_min = 0.1
        self.epsilon_decay = (self.epsilon - self.epsilon_min) / 10000  # 선형 감소

        # 정책 테이블 초기화 (Q-테이블)
        self.q_table_filename = q_table_filename
        if load_q_table:
            try:
                with open(self.q_table_filename, 'rb') as f:
                    self.q_table = pickle.load(f)
                print(f"Q-table loaded from {self.q_table_filename}")
            except FileNotFoundError:
                print(f"No Q-table file found. Starting with a new Q-table.")
                self.q_table = np.zeros(self.state_size + (self.direction_size, self.action_size))
        else:
            self.q_table = np.zeros(self.state_size + (self.direction_size, self.action_size))
        # BFS 테이블 초기화
        self.bfs_table_flag = False
        self.bfs_table = np.zeros(self.state_size)
        # 목표 리스트
        self.goals = ['KB', 'DBC', 'KG', 'DGC', 'KR', 'DRC', 'G']
        self.current_goal_index = 0
        # 방향 매핑
        self.directions = ['U', 'R', 'D', 'L']  # 시계 방향 순서
        # 보상 값 설정
        self.goal_reward_value = 1000  # 목표 달성 보상
        self.lava_penalty_value = -50  # 용암에 대한 큰 음의 보상
        self.time_penalty = 0  # 시간 페널티
        self.move_reward_scale = 100  # 이동 보상 스케일링 계수
        self.observation = None
        self.reward_queue = deque(maxlen=len(self.goals))

    def find_agent_position(self, observation):
        """에이전트의 위치와 방향을 찾는 함수"""
        for i in range(self.state_size[0]):
            for j in range(self.state_size[1]):
                cell = observation[i][j]
                if cell in ['AL', 'AR', 'AU', 'AD']:
                    direction = self.direction_mapping[cell]
                    return (i, j), direction
        return None, None

    def find_goal_position(self, goal):
        for i in range(self.state_size[0]):
            for j in range(self.state_size[1]):
                if self.observation[i][j] == goal:
                    return (i, j)
        return None

    def update_direction(self, current_direction, action):
        """현재 방향에서 좌우 회전했을 때의 방향을 반환 하고 에이전트의 방향을 업데이트"""
        idx = self.directions.index(current_direction)
        
        if action == self.ACTION_LEFT:
            idx = (idx - 1) % 4
        elif action == self.ACTION_RIGHT:
            idx = (idx + 1) % 4

        self.current_direction = self.directions[idx]
        return self.directions[idx]

    def is_value_in_grid(self, value, grid):
        for row in grid:
            if value in row:
                return True
        return False

    def custom_reward(self, current_position, action_index, next_position, grid, terminated, cnt):
        """커스텀 보상 함수"""
        goal_reward = 0
        time_penalty = self.time_penalty

        cnt_pos=cnt

        # 이동 보상 계산 (모든 단계에서 적용)
        move_reward = self.bfs_table[next_position[0], next_position[1]] * self.move_reward_scale

        # 이동 보상을 받은 위치의 값을 감소시켜 반복적인 보상 누적 방지
        self.bfs_table[next_position[0], next_position[1]] *= 0.95  # 5% 감소로 수정

        # 현재 목표 가져오기 및 목표 달성 확인
        if self.knu_agent_actions[action_index] == self.ACTION_PICKUP:
            # 열쇠 목표 달성 확인
            keys = ["KB", "KG", "KR"]
            for k in keys:
                if not self.is_value_in_grid(k, grid):
                    if k not in self.reward_queue:
                        self.reward_queue.append(k)
                        goal_reward += self.goal_reward_value

        if self.knu_agent_actions[action_index] == self.ACTION_UNLOCK:
            # 문 목표 달성 확인
            opened_doors = ["DBC", "DGC", "DRC"]
            for d in opened_doors:
                if self.is_value_in_grid(d, grid):
                    if d not in self.reward_queue:
                        self.reward_queue.append(d)
                        goal_reward += self.goal_reward_value

        if self.knu_agent_actions[action_index] == self.ACTION_FORWARD:
            cell = grid[next_position[0]][next_position[1]]
            if cell.endswith('O'):  # 문이 열린 상태
                if cell not in self.reward_queue:
                    self.reward_queue.append(cell)
                    goal_reward += 100  # 추가 보상

        if terminated and next_position == (24, 24):
            if 'G' not in self.reward_queue:
                goal_reward += 1000
                self.reward_queue.append('G')
            print(f"목표 달성: {next_position}")       

        # 용암에 빠진 경우
        if terminated and grid[next_position[0]][next_position[1]] == 'L':
            print(f"용암 보상: {next_position}")
            terminal_penalty = self.lava_penalty_value  # 큰 음의 보상
        else:
            terminal_penalty = 0

        # 벽에 부딪혔을 때 페널티 부여
        if current_position == next_position:
            cnt_pos += 1
            wall_penalty = -1 * cnt_pos
        else:
            wall_penalty = 0
            cnt_pos = 0

        if goal_reward > 0:
            print(f"목표 보상: {goal_reward}, goals:{self.reward_queue}")

        total_reward = goal_reward + move_reward + time_penalty + terminal_penalty + wall_penalty
        return total_reward, cnt_pos

    def act(self, observation):
        """현재 상태에서 행동을 선택하는 함수"""
        position, direction = self.find_agent_position(observation)
        if position is None:
            # 기본 행동으로 앞으로 이동
            return self.ACTION_FORWARD, self.knu_agent_actions.index(self.ACTION_FORWARD)
        
        if np.random.rand() < self.epsilon:
            # 무작위 행동 선택
            action_index = np.random.choice(self.action_size)
        else:
            # 최적 행동 선택
            state_actions = self.q_table[position[0], position[1], direction, :]
            action_index = np.argmax(state_actions)
            
            # Q-값이 모두 동일한 경우 무작위 선택
            if np.all(state_actions == state_actions[0]):
                action_index = np.random.choice(self.action_size)
        print(action_index,position,direction)
        
        action = self.knu_agent_actions[action_index]
        # print(f"행동: {action}, 행동 인덱스: {action_index}")
        return action_index

    def update_q_table(self, state, action_index, direction_index, reward, next_state, next_direction, done):
        """Q-테이블 업데이트"""
        x, y = state
        current_q = self.q_table[x, y, direction_index, action_index]
        if done:
            target = reward  # 종료 상태에서는 다음 상태의 Q-값을 고려하지 않음
        else:
            next_x, next_y = next_state
            next_max_q = np.max(self.q_table[next_x, next_y, next_direction, :])  # 다음 상태의 방향을 고려
            target = reward + self.gamma * next_max_q
        self.q_table[x, y, direction_index, action_index] += self.alpha * (target - current_q)

    def bfs_shortest_path(self, grid, start, goal):
        """BFS를 사용하여 최단 경로의 거리를 계산 (문은 모두 열려 있다고 가정)"""
        rows, cols = len(grid), len(grid[0])
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        visited = [[False for _ in range(cols)] for _ in range(rows)]
        queue = deque([(start, 0)])
        visited[start[0]][start[1]] = True

        while queue:
            (current, distance) = queue.popleft()

            if current == goal:
                return distance
            for d in directions:
                new_row, new_col = current[0] + d[0], current[1] + d[1]
                if (0 <= new_row < rows and 0 <= new_col < cols and not visited[new_row][new_col]):
                    cell = grid[new_row][new_col]
                    if cell != 'W' and cell != 'L':  # 벽과 용암은 통과 불가능
                        visited[new_row][new_col] = True
                        queue.append(((new_row, new_col), distance + 1))
        return float('inf')

    def update_bfs_table(self, grid):
        """BFS 테이블 업데이트"""
        goal_position = (24, 24)
        for i in range(26):
            for j in range(26):
                if grid[i][j] == 'W' or grid[i][j] == 'L':
                    self.bfs_table[i, j] = 0
                else:
                    distance = self.bfs_shortest_path(grid, (i, j), goal_position)
                    if distance != float('inf') and distance != 0:
                        self.bfs_table[i, j] = 1 / distance  # 스케일링 값 10 적용
                    else:
                        self.bfs_table[i, j] = 0

        # BFS 테이블 정규화 제거
        max_value = np.max(self.bfs_table)
        if max_value > 0:
            self.bfs_table = self.bfs_table / max_value

        self.bfs_table_flag = True

    def save_q_table(self):
        """Q-테이블 저장"""
        with open(self.q_table_filename, 'wb') as f:
            pickle.dump(self.q_table, f)
        print(f"Q-table saved to {self.q_table_filename}")

    def train(self, num_episodes=10000):
        # 환경 생성 시 max_steps=2704로 설정
        env = make_grid_adventure(show_screen=False)
        episode_rewards = []
        episode = 0

        while episode < num_episodes:
            self.reward_queue.clear()
            episode += 1
            observation, _ = env.reset()
            (state, direction) = self.find_agent_position(observation)
            done = False
            total_reward = 0
            self.current_goal_index = 0  # 에피소드 시작 시 목표 인덱스 초기화
            grid = observation  # 현재 그리드 초기화

            self.observation = observation

            self.update_bfs_table(grid)  # BFS 테이블 업데이트

            step_count = 0  # 스텝 수 초기화
            cnt_pos = 0
            while not done:
                action_index = self.act(observation)
                action = self.knu_agent_actions[action_index]
                current_position = state
                current_direction = direction

                # 행동 수행
                observation_next, _, terminated, truncated, _ = env.step(action)

                done = terminated or truncated
                (next_state, next_direction) = self.find_agent_position(observation_next)
                if next_state is None:
                    done = True
                    break

                state = next_state
                direction = next_direction
                observation = observation_next
                grid = observation  # 그리드 업데이트

                step_count += 1  # 스텝 수 증가


                # 커스텀 보상 계산
                custom_reward , cnt_pos = self.custom_reward(
                    current_position, action_index, state, grid, done, cnt_pos
                )

                # Q-테이블 업데이트
                self.update_q_table(
                    current_position, 
                    action_index, 
                    current_direction, 
                    custom_reward, 
                    state, 
                    next_direction,  # 다음 방향 정보 전달
                    done
                )
                total_reward += custom_reward

                # 탐색률 감소 (선형 감소)
                if self.epsilon > self.epsilon_min:
                    self.epsilon -= self.epsilon_decay

                if done:
                    print(f"Episode: {episode}, Total Reward: {total_reward:.2f}, Epsilon: {self.epsilon:.4f}, Goals Achieved: {self.reward_queue}, Steps: {step_count}, Current Position: {current_position}")
                    break

            episode_rewards.append(total_reward)

            # 에피소드마다 Q-테이블 저장
            self.save_q_table()

        return self, episode_rewards

if __name__ == '__main__':
    # 학습 재개를 원하면 load_q_table=True로 설정
    print(np.__version__)
    agent = GridAdventureRLAgent(load_q_table=True, q_table_filename='result_agent/knu_q_table_v2_1.pkl',epsilon=0)
    trained_agent, episode_rewards = agent.train(1)
    np.set_printoptions(precision=2, suppress=True, linewidth=300)
    print(*trained_agent.bfs_table, sep='\n')
    # 학습된 에이전트를 로드하여 평가 실행
    # evaluate(trained_agent)
