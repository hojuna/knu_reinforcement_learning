import numpy as np
import pickle  # Q-테이블 저장 및 로드를 위한 모듈
from collections import deque
from knu_rl_env.grid_adventure import GridAdventureAgent, make_grid_adventure, evaluate

class GridAdventureRLAgent(GridAdventureAgent):
    def __init__(self, load_q_table=False, q_table_filename='q_table.pkl'):
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
        self.q_table_filename = q_table_filename
        if load_q_table:
            # 저장된 Q-테이블 로드
            with open(self.q_table_filename, 'rb') as f:
                self.q_table = pickle.load(f)
            print(f"Q-table loaded from {self.q_table_filename}")
        else:
            self.q_table = np.zeros(self.state_size + (self.action_size,))
        # BFS 테이블 초기화
        self.bfs_table_flag = False
        self.bfs_table = np.zeros(self.state_size)
        # 목표 리스트 및 현재 목표 인덱스 초기화
        self.goals = ['KB', 'DBO', 'KG', 'DGO', 'KR', 'DRO', 'G']
        self.current_goal_index = 0
        # 방향 매핑
        self.directions = ['U', 'R', 'D', 'L']  # 시계 방향 순서
        # 에피소드별 목표 달성 수 초기화
        self.goals_achieved = 0

    def find_agent_position(self, observation):
        for i in range(self.state_size[0]):
            for j in range(self.state_size[1]):
                cell = observation[i][j]
                if cell in ['AL', 'AR', 'AU', 'AD']:
                    # 에이전트의 방향도 함께 반환
                    direction = cell[-1]  # 'L', 'R', 'U', 'D'
                    return (i, j), direction
        return None, None  # 에이전트 위치를 찾지 못한 경우

    def get_next_position(self, current_position, current_direction, action_sequence, grid):
        x, y = current_position
        direction = current_direction

        for action in action_sequence:
            if action == self.ACTION_LEFT:
                # 왼쪽으로 90도 회전
                direction = self.rotate_direction(direction, 'LEFT')
            elif action == self.ACTION_RIGHT:
                # 오른쪽으로 90도 회전
                direction = self.rotate_direction(direction, 'RIGHT')
            elif action == self.ACTION_FORWARD:
                # 현재 방향으로 한 칸 전진
                dx, dy = self.direction_to_delta(direction)
                new_x, new_y = x + dx, y + dy
                # 그리드 범위 내에 있고
                if 0 <= new_x < self.state_size[0] and 0 <= new_y < self.state_size[1]:
                    x, y = new_x, new_y
                # 이동 불가능한 경우 위치 변화 없음
            # PICKUP, DROP, UNLOCK 등의 행동은 위치 변화 없음
            else:
                pass  # 위치 변화 없음
        return (x, y), direction

    def rotate_direction(self, current_direction, turn):
        # 현재 방향을 기준으로 회전 후의 방향 반환
        idx = self.directions.index(current_direction)
        if turn == 'LEFT':
            idx = (idx - 1) % 4
        elif turn == 'RIGHT':
            idx = (idx + 1) % 4
        return self.directions[idx]

    def direction_to_delta(self, direction):
        # 방향을 움직임의 변화량으로 변환
        if direction == 'U':
            return (-1, 0)
        elif direction == 'D':
            return (1, 0)
        elif direction == 'L':
            return (0, -1)
        elif direction == 'R':
            return (0, 1)

    def find_current_goal(self, observation):
        if self.current_goal_index >= len(self.goals):
            return None  # 모든 목표 달성
        current_goal = self.goals[self.current_goal_index]
        # 현재 목표의 위치 찾기
        for i in range(self.state_size[0]):
            for j in range(self.state_size[1]):
                if observation[i][j] == current_goal:
                    return (i, j)  # 목표 위치 반환
        return None  # 목표를 찾지 못한 경우

    def update_goal_flag(self):
        self.current_goal_index += 1
        self.bfs_table_flag = False  # 새로운 목표를 위해 BFS 테이블 업데이트 필요
        self.goals_achieved += 1  # 목표 달성 수 증가

    def custom_reward(self, current_position, action_index, next_position, grid):
        # BFS 테이블에 기반한 이동 보상 계산
        move_reward = self.bfs_table[next_position[0], next_position[1]]
        # 목표 보상 및 용암 페널티 초기화
        goal_reward = 0
        lava_penalty = 0

        # 용암 지형인지 확인하여 페널티 적용
        if grid[next_position[0]][next_position[1]] == 'L':
            lava_penalty = -10  # 용암 페널티

        # 목표에 도달했는지 확인
        goal = self.find_current_goal(grid)
        if goal and next_position == goal:
            goal_reward = 100  # 목표 보상
            self.update_goal_flag()
            self.bfs_table_flag = False  # 새로운 목표를 위해 BFS 테이블 업데이트 필요

        # 총 보상 계산
        total_reward = move_reward + goal_reward + lava_penalty
        return total_reward

    def act(self, observation):
        position, direction = self.find_agent_position(observation)
        if position is None:
            return self.knu_agent_forward, self.knu_agent_actions.index(self.knu_agent_forward)  # 기본적으로 앞으로 가기
        if np.random.rand() < self.epsilon:
            # 무작위 행동 선택 (탐험)
            action_index = np.random.choice(self.action_size)
        else:
            # 정책에 따른 최적 행동 선택 (활용)
            state_actions = self.q_table[position[0], position[1], :]
            action_index = np.argmax(state_actions)
        action_sequence = self.knu_agent_actions[action_index]
        return action_sequence, action_index

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
            for d in directions:
                new_row, new_col = current[0] + d[0], current[1] + d[1]

                # 그리드 범위 내에 있고, 방문하지 않았으며, 이동 가능한 경우
                if (0 <= new_row < rows and 0 <= new_col < cols and
                    not visited[new_row][new_col] and grid[new_row][new_col] != 'W'):
                    visited[new_row][new_col] = True
                    queue.append(((new_row, new_col), distance + 1))

        # 목표에 도달할 수 없는 경우
        return float('inf')

    def update_bfs_table(self, grid):
        goal = self.find_current_goal(grid)
        if goal is None:
            return  # 업데이트할 목표가 없음
        # BFS 테이블 업데이트 로직
        for i in range(26):
            for j in range(26):
                if grid[i][j] != 'W':  # 벽이 아닌 경우에만 업데이트
                    distance = self.bfs_shortest_path(grid, (i, j), goal)
                    if distance != float('inf') and distance != 0:
                        self.bfs_table[i, j] = 1 / distance
                    else:
                        self.bfs_table[i, j] = 0  # 도달 불가능한 경우 0으로 설정
                else:
                    self.bfs_table[i, j] = 0  # 벽인 경우 0으로 설정
        self.bfs_table_flag = True

    def save_q_table(self):
        with open(self.q_table_filename, 'wb') as f:
            pickle.dump(self.q_table, f)
        print(f"Q-table saved to {self.q_table_filename}")

def train(load_q_table=False, q_table_filename='q_table.pkl'):
    env = make_grid_adventure(show_screen=True)
    agent = GridAdventureRLAgent(load_q_table=load_q_table, q_table_filename=q_table_filename)
    num_episodes = 1000  # 학습 에피소드 수
    episode_rewards = []
    episode_goals = []  # 에피소드별 목표 달성 수를 저장하기 위한 리스트

    for episode in range(num_episodes):
        observation, info = env.reset()
        (state, direction) = agent.find_agent_position(observation)
        done = False
        total_reward = 0
        agent.current_goal_index = 0  # 에피소드 시작 시 목표 인덱스 초기화
        agent.bfs_table_flag = False
        agent.goals_achieved = 0  # 에피소드별 목표 달성 수 초기화

        while not done:
            if not agent.bfs_table_flag:
                # BFS 테이블 업데이트
                grid = observation  # 환경의 그리드를 얻어옴
                agent.update_bfs_table(grid)
                # agent.bfs_table_flag = True  # 이미 메서드에서 설정됨

            action_sequence, action_index = agent.act(observation)
            current_position = state
            current_direction = direction
            next_state = state
            next_direction = direction

            # 예상되는 다음 위치와 방향을 계산 (시뮬레이션)
            simulated_next_position, simulated_next_direction = agent.get_next_position(
                current_position, current_direction, action_sequence, observation
            )

            # 커스텀 보상 계산
            custom_reward = agent.custom_reward(current_position, action_index, simulated_next_position, observation)

            for action in action_sequence:
                observation_next, env_reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                # 에이전트의 위치와 방향 업데이트
                (next_state, next_direction) = agent.find_agent_position(observation_next)
                if next_state is None:
                    # 에이전트 위치를 찾지 못한 경우 종료
                    done = True
                    break

                # 상태와 방향 업데이트
                state = next_state
                direction = next_direction
                observation = observation_next

                if done:
                    break

            # Q-테이블 업데이트 (복합 행동 단위로)
            agent.update_q_table(current_position, action_index, custom_reward, simulated_next_position, done)
            total_reward += custom_reward

            # 탐색률 감소
            if agent.epsilon > agent.epsilon_min:
                agent.epsilon *= agent.epsilon_decay

            if done:
                print(f"Episode: {episode+1}, Total Reward: {total_reward:.2f}, Goals Achieved: {agent.goals_achieved}, Epsilon: {agent.epsilon:.4f}")
                break

        episode_rewards.append(total_reward)
        episode_goals.append(agent.goals_achieved)  # 에피소드별 목표 달성 수 저장

    # 학습된 에이전트 저장
    agent.save_q_table()

    # 에피소드별 총 보상과 목표 달성 수를 파일로 저장
    with open('episode_rewards.pkl', 'wb') as f:
        pickle.dump(episode_rewards, f)
    with open('episode_goals.pkl', 'wb') as f:
        pickle.dump(episode_goals, f)

    # 학습된 에이전트 반환
    return agent, episode_rewards, episode_goals

if __name__ == '__main__':
    trained_agent, episode_rewards, episode_goals = train(load_q_table=False, q_table_filename='q_table.pkl')
    # 에피소드별 총 보상과 목표 달성 수를 시각화할 수 있습니다.
    # 예를 들어, matplotlib를 사용하여 그래프를 그릴 수 있습니다.
    # 학습된 에이전트를 로드하여 평가 실행
    agent_for_evaluation = GridAdventureRLAgent(load_q_table=True, q_table_filename='q_table.pkl')
    evaluate(agent_for_evaluation)
