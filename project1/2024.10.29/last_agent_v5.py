import numpy as np
import pickle
from collections import deque
from knu_rl_env.grid_adventure import GridAdventureAgent, make_grid_adventure, evaluate

class GridAdventureRLAgent(GridAdventureAgent):
    def __init__(self, load_q_table=False, q_table_filename='q_table.pkl', epsilon=0.1):
        super().__init__()
        # 행동 정의
        self.ACTION_LEFT = GridAdventureAgent.ACTION_LEFT
        self.ACTION_RIGHT = GridAdventureAgent.ACTION_RIGHT
        self.ACTION_FORWARD = GridAdventureAgent.ACTION_FORWARD
        self.ACTION_PICKUP = GridAdventureAgent.ACTION_PICKUP
        self.ACTION_DROP = GridAdventureAgent.ACTION_DROP
        self.ACTION_UNLOCK = GridAdventureAgent.ACTION_UNLOCK

        self.knu_agent_actions = [
            self.ACTION_LEFT,
            self.ACTION_RIGHT,
            self.ACTION_FORWARD,
            self.ACTION_PICKUP,
            self.ACTION_DROP,
            self.ACTION_UNLOCK
        ]

        self.action_size = len(self.knu_agent_actions)

        # 상태 인코딩
        self.state_size = (26, 26, 4, 4, 3)  # 위치, 방향, 키, 문 상태
        self.q_table_filename = q_table_filename
        if load_q_table:
            try:
                with open(self.q_table_filename, 'rb') as f:
                    self.q_table = pickle.load(f)
                print(f"Q-table loaded from {self.q_table_filename}")
            except FileNotFoundError:
                print(f"No Q-table found. Initializing a new Q-table.")
                self.q_table = np.zeros(self.state_size + (self.action_size,))
        else:
            self.q_table = np.zeros(self.state_size + (self.action_size,))

        # 하이퍼파라미터
        self.alpha = 0.1  # 학습률
        self.gamma = 0.95  # 할인율
        self.epsilon = epsilon  # 탐색률
        self.epsilon_min = 0.05
        self.epsilon_decay = (self.epsilon - self.epsilon_min) / 20000  # 20,000 에피소드 동안 선형 감소

        # 추가 변수
        self.key_flag = False
        self.current_key = None
        self.dropped_keys = set()  # 열쇠를 버린 위치를 저장하는 집합

        # 보상 값
        self.goal_reward = 1000
        self.lava_penalty = -1000
        self.time_penalty = -1
        self.unnecessary_drop_penalty = -500

    def find_agent_position(self, grid):
        """플레이어의 위치와 방향을 찾는 함수"""
        for i in range(26):
            for j in range(26):
                cell = grid[i][j]
                if cell in ['AL', 'AR', 'AU', 'AD']:
                    direction = {'AL': 3, 'AR': 1, 'AU': 0, 'AD': 2}[cell]
                    return (i, j), direction
        return None, None

    def find_faced_cell(self, position, direction, grid):
        """에이전트가 바라보고 있는 셀의 내용을 반환하는 함수"""
        direction_mapping = {
            0: (-1, 0),  # 위
            1: (0, 1),   # 오른쪽
            2: (1, 0),   # 아래
            3: (0, -1)   # 왼쪽
        }
        delta = direction_mapping.get(direction, (0, 0))
        faced_position = (position[0] + delta[0], position[1] + delta[1])
        if 0 <= faced_position[0] < 26 and 0 <= faced_position[1] < 26:
            return grid[faced_position[0]][faced_position[1]]
        return None

    def get_key_index(self, flag, key_type=None):
        """키 상태를 인덱스로 변환하는 함수"""
        if not flag or key_type is None:
            return 0
        key_mapping = {'KB': 1, 'KG': 2, 'KR': 3}
        return key_mapping.get(key_type, 0)

    def get_door_state_index(self, grid):
        """문의 상태를 인덱스로 변환하는 함수"""
        # 예시: 특정 위치의 문 상태를 인덱스로 반환
        # 실제 문 위치와 상태는 환경에 따라 조정 필요
        # 여기서는 예시로 일부 문 상태를 가져오는 코드
        # 'DBO', 'DGO', 'DRO' 등
        b_state, g_state, r_state = 0, 0, 0
        for i in range(26):
            for j in range(26):
                cell = grid[i][j]
                if cell.startswith('DB'):
                    b_state = self.door_suffix_to_index(cell[-1])
                elif cell.startswith('DG'):
                    g_state = self.door_suffix_to_index(cell[-1])
                elif cell.startswith('DR'):
                    r_state = self.door_suffix_to_index(cell[-1])
        return (b_state, g_state, r_state)

    def door_suffix_to_index(self, suffix):
        """문 상태 접미사를 인덱스로 변환하는 함수"""
        mapping = {'L': 0, 'O': 1, 'C': 2}
        return mapping.get(suffix, 0)

    def encode_state(self, position, direction, has_key, door_state_index):
        """상태를 하나의 고유한 숫자로 인코딩하는 함수"""
        x, y = position
        b, g, r = door_state_index
        return (
            (x * 26 + y) * (4 * 4 * 27) +
            direction * (4 * 27) +
            has_key * 27 +
            b * 9 +
            g * 3 +
            r * 1
        )

    def act(self, observation):
        """현재 상태에서 행동을 선택하는 함수"""
        position, direction = self.find_agent_position(observation)
        if position is None or direction is None:
            return self.ACTION_LEFT  # 기본 행동 선택

        has_key = self.get_key_index(self.key_flag, self.current_key)
        door_state_index = self.get_door_state_index(observation)

        # 상태 인코딩
        state_encoded = self.encode_state(position, direction, has_key, door_state_index)

        # 행동 마스킹: 열쇠를 버린 위치에서는 PICKUP 행동을 비활성화
        mask = np.ones(self.action_size, dtype=bool)
        if position in self.dropped_keys:
            mask[self.ACTION_PICKUP] = False  # PICKUP 행동을 비활성화

        if np.random.rand() < self.epsilon:
            # 무작위 행동 선택 (마스킹된 행동 제외)
            available_actions = np.where(mask)[0]
            action_index = np.random.choice(available_actions)
        else:
            # Q-테이블에서 최적 행동 선택 (마스킹된 행동 제외)
            state_actions = self.q_table[position[0], position[1], direction, has_key, door_state_index[0], door_state_index[1], door_state_index[2], :]
            state_actions = np.where(mask, state_actions, -np.inf)  # 마스킹된 행동의 Q-값을 매우 낮게 설정
            action_index = np.argmax(state_actions)
            if np.all(state_actions == state_actions[0]):
                action_index = np.random.choice(np.where(mask)[0])

        action = self.knu_agent_actions[action_index]

        # 키 소유 상태 업데이트
        if action == self.ACTION_PICKUP and not self.key_flag:
            faced_cell = self.find_faced_cell(position, direction, observation)
            if faced_cell in ["KB", "KG", "KR"] and (position not in self.dropped_keys):
                self.key_flag = True
                self.current_key = faced_cell

        # 열쇠를 버리는 행동 처리
        if action == self.ACTION_DROP and self.key_flag:
            DROP_ZONES = ['D']  # 열쇠를 버릴 수 있는 셀 타입 정의
            faced_cell = self.find_faced_cell(position, direction, observation)
            if faced_cell in DROP_ZONES:
                self.key_flag = False
                self.current_key = None
                self.dropped_keys.add(position)  # 열쇠를 버린 위치 기록
            else:
                # 열쇠를 버리지 않도록 다른 행동 선택 (예: 앞으로 이동)
                action_index = self.ACTION_FORWARD
                action = self.knu_agent_actions[action_index]

        return action_index

    def custom_reward(self, current_position, action_index, next_position, grid, terminated, cnt, step_count, has_key, current_direction):
        """커스텀 보상 함수"""
        # 기본 이동 페널티
        total_reward = self.time_penalty
        cnt_pos = cnt

        # 목표 지점 도달
        if grid[next_position[0]][next_position[1]] == 'G':
            total_reward += self.goal_reward
            print("목표 지점에 도달했습니다!")
            return total_reward, cnt_pos

        # 용암에 빠졌을 때
        if grid[next_position[0]][next_position[1]] == 'L':
            total_reward += self.lava_penalty
            print("용암에 빠졌습니다!")
            return total_reward, cnt_pos

        # 불필요하게 열쇠를 버리는 행동에 페널티 부여
        if self.knu_agent_actions[action_index] == self.ACTION_DROP and self.key_flag:
            faced_cell = self.find_faced_cell(current_position, current_direction, grid)
            if faced_cell != 'D':
                total_reward += self.unnecessary_drop_penalty
                print("불필요하게 열쇠를 버렸습니다. 페널티 적용.")
        
        return total_reward, cnt_pos

    def update_q_table(self, state, action_index, direction_index, reward, next_state, next_direction, done, has_key, has_key_next, door_state_index, next_door_state_index):
        """Q-테이블 업데이트"""
        x, y = state
        b, g, r = door_state_index
        try:
            current_q = self.q_table[x, y, direction_index, has_key, b, g, r, action_index]
        except IndexError:
            print(f"Invalid indices: x={x}, y={y}, direction={direction_index}, has_key={has_key}, doors=({b},{g},{r}), action={action_index}")
            return

        if done:
            target = reward
        else:
            next_x, next_y = next_state
            next_b, next_g, next_r = next_door_state_index
            try:
                next_max_q = np.max(self.q_table[next_x, next_y, next_direction, has_key_next, next_b, next_g, next_r, :])
            except IndexError:
                next_max_q = 0
            target = reward + self.gamma * next_max_q

        # 열쇠를 버리는 행동에 페널티 추가
        if self.knu_agent_actions[action_index] == self.ACTION_DROP and self.key_flag:
            target += self.unnecessary_drop_penalty  # 페널티를 이미 보상에 반영했으므로 중복 적용하지 않음

        # Q-테이블 업데이트
        self.q_table[x, y, direction_index, has_key, b, g, r, action_index] += self.alpha * (target - current_q)

    def train_agent(self, num_episodes=10000, show=False):
        """에이전트 훈련 함수"""
        env = make_grid_adventure(show_screen=show)
        episode_rewards = []
        episode = 0

        while episode < num_episodes:
            self.dropped_keys.clear()  # 에피소드 시작 시 버린 열쇠 위치 초기화
            self.key_flag = False
            self.current_key = None
            episode += 1
            observation, _ = env.reset()
            state, direction = self.find_agent_position(observation)
            if state is None or direction is None:
                continue
            done = False
            total_reward = 0
            step_count = 0
            cnt_pos = 0

            while not done:
                has_key = self.get_key_index(self.key_flag, self.current_key)
                door_state_index = self.get_door_state_index(observation)

                action_index = self.act(observation)
                action = self.knu_agent_actions[action_index]

                # 행동 수행
                observation_next, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                next_state, next_direction = self.find_agent_position(observation_next)
                if next_state is None:
                    done = True
                    break

                grid = observation_next
                next_door_state_index = self.get_door_state_index(observation_next)

                step_count += 1

                # 커스텀 보상 계산
                reward, cnt_pos = self.custom_reward(
                    state, action_index, next_state, grid, done, cnt_pos, step_count, has_key, direction
                )

                # Q-테이블 업데이트
                self.update_q_table(
                    state,
                    action_index,
                    direction,
                    reward,
                    next_state,
                    next_direction,
                    done,
                    has_key,
                    self.get_key_index(self.key_flag, self.current_key),
                    door_state_index,
                    next_door_state_index
                )
                total_reward += reward

                if done:
                    print(f"Episode: {episode}, Total Reward: {total_reward:.2f}, Epsilon: {self.epsilon:.4f}, Steps: {step_count}")
                    break

                state = next_state
                direction = next_direction
                observation = observation_next

            # 탐색률 감소 (선형 감소)
            if self.epsilon > self.epsilon_min:
                self.epsilon -= self.epsilon_decay

            episode_rewards.append(total_reward)

            # 에피소드마다 Q-테이블 저장 (예: 100 에피소드마다)
            if episode % 100 == 0:
                self.save_q_table()

        # 학습 종료 시 Q-테이블 저장
        self.save_q_table()

        return self, episode_rewards

    def save_q_table(self):
        """Q-테이블 저장 함수"""
        with open(self.q_table_filename, 'wb') as f:
            pickle.dump(self.q_table, f)
        print(f"Q-table saved to {self.q_table_filename}")

    def load_q_table(self):
        """Q-테이블 로드 함수"""
        try:
            with open(self.q_table_filename, 'rb') as f:
                self.q_table = pickle.load(f)
            print(f"Q-table loaded from {self.q_table_filename}")
        except FileNotFoundError:
            print(f"No Q-table found at {self.q_table_filename}. Starting with a new Q-table.")
            self.q_table = np.zeros(self.state_size + (self.action_size,))

def train():
    """에이전트를 훈련시키는 함수"""
    agent = GridAdventureRLAgent(load_q_table=False, q_table_filename='q_table.pkl', epsilon=0.5)
    trained_agent, episode_rewards = agent.train_agent(num_episodes=10000, show=False)
    return trained_agent

if __name__ == '__main__':
    # 에이전트 훈련
    agent = train()

    # 학습된 에이전트 평가
    evaluate(agent)
