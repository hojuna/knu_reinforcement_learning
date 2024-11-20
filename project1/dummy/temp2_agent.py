import numpy as np
import pickle
import copy
from collections import deque
from knu_rl_env.grid_adventure import GridAdventureAgent, make_grid_adventure, evaluate

class GridAdventureRLAgent(GridAdventureAgent):
    def __init__(self, load_q_table=False, q_table_filename='q_table_improved.pkl'):
        # 가능한 행동들 정의 (기본 행동)
        self.ACTION_LEFT = GridAdventureAgent.ACTION_LEFT
        self.ACTION_RIGHT = GridAdventureAgent.ACTION_RIGHT
        self.ACTION_FORWARD = GridAdventureAgent.ACTION_FORWARD
        self.ACTION_PICKUP = GridAdventureAgent.ACTION_PICKUP
        self.ACTION_DROP = GridAdventureAgent.ACTION_DROP
        self.ACTION_UNLOCK = GridAdventureAgent.ACTION_UNLOCK

        # 복합 행동 정의 (복잡성 감소)
        self.knu_agent_left = [self.ACTION_LEFT, self.ACTION_FORWARD]
        self.knu_agent_right = [self.ACTION_RIGHT, self.ACTION_FORWARD]
        self.knu_agent_forward = [self.ACTION_FORWARD]
        self.knu_agent_pickup = [self.ACTION_PICKUP]
        self.knu_agent_unlock = [self.ACTION_UNLOCK, self.ACTION_FORWARD]
        self.knu_agent_drop = [self.ACTION_DROP]
        self.knu_agent_actions = [
            self.knu_agent_left,
            self.knu_agent_right,
            self.knu_agent_forward,
            self.knu_agent_pickup,
            self.knu_agent_unlock,
            self.knu_agent_drop
        ]

        # 상태 공간 크기 정의
        self.state_size = (26, 26)
        # 행동 공간 크기 정의
        self.action_size = len(self.knu_agent_actions)
        # 학습률(alpha)
        self.alpha = 0.1
        # 할인율(gamma)
        self.gamma = 0.99
        # 탐색률(epsilon)
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = (self.epsilon - self.epsilon_min) / 500  # 선형 감소
        # 정책 테이블 초기화 (Q-테이블)
        self.q_table_filename = q_table_filename
        if load_q_table:
            with open(self.q_table_filename, 'rb') as f:
                self.q_table = pickle.load(f)
            print(f"Q-table loaded from {self.q_table_filename}")
        else:
            self.q_table = np.zeros(self.state_size + (self.action_size,))
        # BFS 테이블 초기화
        self.bfs_table_flag = False
        self.bfs_table = np.zeros(self.state_size)
        # 목표 리스트
        self.goals = ['KB', 'DBO', 'KG', 'DGO', 'KR', 'DRO', 'G']
        self.current_goal_index = 0
        # 방향 매핑
        self.directions = ['U', 'R', 'D', 'L']  # 시계 방향 순서
        # 에피소드별 목표 달성 수 초기화
        self.goals_achieved = 0
        # 보상 값 설정
        self.goal_reward_value = 100
        self.lava_penalty_value = -100  # 용암에 대한 큰 음의 보상
        self.time_penalty = -0.1  # 시간 페널티 감소
        self.move_reward_scale = 10  # move_reward 스케일링 계수 추가

    def find_agent_position(self, observation):
        """에이전트의 위치와 방향을 찾는 함수"""
        for i in range(self.state_size[0]):
            for j in range(self.state_size[1]):
                cell = observation[i][j]
                if cell in ['AL', 'AR', 'AU', 'AD']:
                    direction = cell[-1]  # 'L', 'R', 'U', 'D'
                    return (i, j), direction
        return None, None

    def get_next_position(self, current_position, current_direction, action_sequence, grid):
        """현재 위치와 방향에서 주어진 행동들을 수행했을 때의 예상 위치와 방향을 반환"""
        x, y = current_position
        direction = current_direction
        grid = copy.deepcopy(grid)  # 깊은 복사로 그리드 복사

        for action in action_sequence:
            if action == self.ACTION_LEFT:
                direction = self.rotate_direction(direction, 'LEFT')
            elif action == self.ACTION_RIGHT:
                direction = self.rotate_direction(direction, 'RIGHT')
            elif action == self.ACTION_FORWARD:
                dx, dy = self.direction_to_delta(direction)
                new_x, new_y = x + dx, y + dy
                if 0 <= new_x < self.state_size[0] and 0 <= new_y < self.state_size[1]:
                    cell = grid[new_x][new_y]
                    if cell not in ['W', 'DBL', 'DBC', 'DGL', 'DGC', 'DRL', 'DRC']:
                        x, y = new_x, new_y
            elif action == self.ACTION_UNLOCK:
                dx, dy = self.direction_to_delta(direction)
                door_x, door_y = x + dx, y + dy
                if 0 <= door_x < self.state_size[0] and 0 <= door_y < self.state_size[1]:
                    door_cell = grid[door_x][door_y]
                    if door_cell in ['DBL', 'DBC', 'DGL', 'DGC', 'DRL', 'DRC']:
                        if door_cell.startswith('DB'):
                            grid[door_x][door_y] = 'DBO'
                        elif door_cell.startswith('DG'):
                            grid[door_x][door_y] = 'DGO'
                        elif door_cell.startswith('DR'):
                            grid[door_x][door_y] = 'DRO'
            # 다른 행동들은 위치 변화 없음
        return (x, y), direction

    def rotate_direction(self, current_direction, turn):
        """현재 방향에서 좌우 회전했을 때의 방향을 반환"""
        idx = self.directions.index(current_direction)
        if turn == 'LEFT':
            idx = (idx - 1) % 4
        elif turn == 'RIGHT':
            idx = (idx + 1) % 4
        return self.directions[idx]

    def direction_to_delta(self, direction):
        """방향을 움직임의 변화량으로 변환"""
        if direction == 'U':
            return (-1, 0)
        elif direction == 'D':
            return (1, 0)
        elif direction == 'L':
            return (0, -1)
        elif direction == 'R':
            return (0, 1)

    def find_current_goal(self, grid):
        """현재 목표의 위치를 찾는 함수"""
        if self.current_goal_index >= len(self.goals):
            return None  # 모든 목표 달성
        current_goal = self.goals[self.current_goal_index]
        for i in range(self.state_size[0]):
            for j in range(self.state_size[1]):
                if grid[i][j] == current_goal:
                    return (i, j)
        return None

    def update_goal_flag(self):
        """목표를 달성했을 때 호출되는 함수"""
        self.current_goal_index += 1
        self.bfs_table_flag = False  # 새로운 목표를 위해 BFS 테이블 업데이트 필요
        self.goals_achieved += 1  # 목표 달성 수 증가

    def custom_reward(self, current_position, action_index, next_position, grid, terminated):
        """커스텀 보상 함수"""
        move_reward = self.bfs_table[next_position[0], next_position[1]] * self.move_reward_scale
        goal_reward = 0
        time_penalty = self.time_penalty

        # 에피소드가 종료되었는지 확인 (용암에 빠졌거나 목표를 달성한 경우)
        if terminated:
            if grid[next_position[0]][next_position[1]] == 'L':
                # 용암에 빠진 경우
                terminal_penalty = self.lava_penalty_value  # 큰 음의 보상
            else:
                # 목표를 달성한 경우
                goal_reward = self.goal_reward_value
                terminal_penalty = 0
                self.update_goal_flag()
        else:
            terminal_penalty = 0

        # 벽에 부딪혔을 때 페널티 부여
        if current_position == next_position:
            wall_penalty = -10
        else:
            wall_penalty = 0

        total_reward = move_reward + goal_reward + time_penalty + terminal_penalty + wall_penalty
        return total_reward

    def act(self, observation):
        """현재 상태에서 행동을 선택하는 함수"""
        position, direction = self.find_agent_position(observation)
        if position is None:
            return self.knu_agent_forward, self.knu_agent_actions.index(self.knu_agent_forward)
        if np.random.rand() < self.epsilon:
            # 무작위 행동 선택
            action_index = np.random.choice(self.action_size)
        else:
            # 최적 행동 선택
            state_actions = self.q_table[position[0], position[1], :]
            action_index = np.argmax(state_actions)
        action_sequence = self.knu_agent_actions[action_index]
        return action_sequence, action_index

    def update_q_table(self, state, action_index, reward, next_state, done):
        """Q-테이블 업데이트"""
        x, y = state
        current_q = self.q_table[x, y, action_index]
        if done:
            target = reward  # 종료 상태에서는 다음 상태의 Q-값을 고려하지 않음
        else:
            next_x, next_y = next_state
            next_max_q = np.max(self.q_table[next_x, next_y, :])
            target = reward + self.gamma * next_max_q
        self.q_table[x, y, action_index] += self.alpha * (target - current_q)

    def bfs_shortest_path(self, grid, start, goal):
        """BFS를 사용하여 최단 경로의 거리를 계산"""
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
                if (0 <= new_row < rows and 0 <= new_col < cols and
                    not visited[new_row][new_col]):
                    cell = grid[new_row][new_col]
                    if cell not in ['W', 'DBL', 'DBC', 'DGL', 'DGC', 'DRL', 'DRC']:
                        visited[new_row][new_col] = True
                        queue.append(((new_row, new_col), distance + 1))
        return float('inf')

    def update_bfs_table(self, grid):
        """BFS 테이블 업데이트"""
        goal = self.find_current_goal(grid)
        if goal is None:
            return
        for i in range(26):
            for j in range(26):
                if grid[i][j] != 'W':
                    distance = self.bfs_shortest_path(grid, (i, j), goal)
                    if distance != float('inf') and distance != 0:
                        self.bfs_table[i, j] = 1 / distance
                    else:
                        self.bfs_table[i, j] = 0
                else:
                    self.bfs_table[i, j] = 0

        # bfs_table 값 정규화
        max_value = np.max(self.bfs_table)
        if max_value > 0:
            self.bfs_table = self.bfs_table / max_value

        self.bfs_table_flag = True

    def save_q_table(self, episode=None):
        """Q-테이블 저장 (에피소드 번호별로 저장 가능)"""
        if episode is not None:
            filename = f'q_table_improved_episode_{episode+1}.pkl'
        else:
            filename = self.q_table_filename
        with open(filename, 'wb') as f:
            pickle.dump(self.q_table, f)
        print(f"Q-table saved to {filename}")

    def train(self, load_q_table=False, q_table_filename='q_table_improved.pkl'):
        env = make_grid_adventure(show_screen=True)
        agent = GridAdventureRLAgent(load_q_table=load_q_table, q_table_filename=q_table_filename)
        num_episodes = 10000  # 학습 에피소드 수
        episode_rewards = []
        episode_goals = []
        episode = 0

        while True:
            episode += 1
            observation, _ = env.reset()
            (state, direction) = agent.find_agent_position(observation)
            done = False
            total_reward = 0
            agent.current_goal_index = 0
            agent.bfs_table_flag = False
            agent.goals_achieved = 0
            grid = observation  # 현재 그리드 초기화

            while not done:
                if not agent.bfs_table_flag:
                    agent.update_bfs_table(grid)

                action_sequence, action_index = agent.act(observation)
                current_position = state
                current_direction = direction

                # 시뮬레이션을 위한 그리드 복사
                simulated_grid = copy.deepcopy(grid)

                # 예상되는 다음 위치와 방향을 계산 (시뮬레이션)
                simulated_next_position, simulated_next_direction = agent.get_next_position(
                    current_position, current_direction, action_sequence, simulated_grid
                )

                # 환경에서 실제로 행동 수행
                for action in action_sequence:
                    observation_next, env_reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    (next_state, next_direction) = agent.find_agent_position(observation_next)
                    if next_state is None:
                        done = True
                        break

                    state = next_state
                    direction = next_direction
                    observation = observation_next
                    grid = observation  # 그리드 업데이트

                    if done:
                        break

                # 커스텀 보상 계산 (terminated 플래그 전달)
                custom_reward = agent.custom_reward(current_position, action_index, simulated_next_position, grid, terminated)

                # Q-테이블 업데이트
                agent.update_q_table(current_position, action_index, custom_reward, simulated_next_position, done)
                total_reward += custom_reward

                # 탐색률 감소 (선형 감소)
                if agent.epsilon > agent.epsilon_min:
                    agent.epsilon -= agent.epsilon_decay

                if done:
                    print(f"Episode: {episode}, Total Reward: {total_reward:.2f}, Goals Achieved: {agent.goals_achieved}, Epsilon: {agent.epsilon:.4f}")
                    break

            episode_rewards.append(total_reward)
            episode_goals.append(agent.goals_achieved)

            # 에피소드마다 Q-테이블과 보상, 목표 달성 수 저장
            agent.save_q_table(episode=episode)
            with open('episode_rewards_improved.pkl', 'wb') as f:
                pickle.dump(episode_rewards, f)
            with open('episode_goals_improved.pkl', 'wb') as f:
                pickle.dump(episode_goals, f)

            # 원하는 에피소드 수만큼 학습 후 종료 (여기서는 10000 에피소드)
            if episode >= num_episodes:
                break

        return agent, episode_rewards, episode_goals

if __name__ == '__main__':
    trained_agent, episode_rewards, episode_goals = GridAdventureRLAgent().train(load_q_table=False, q_table_filename='q_table_improved.pkl')
    # 학습된 에이전트를 로드하여 평가 실행
    agent_for_evaluation = GridAdventureRLAgent(load_q_table=True, q_table_filename='q_table_improved.pkl')
    evaluate(agent_for_evaluation)
