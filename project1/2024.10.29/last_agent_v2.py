
import numpy as np
import pickle
from collections import deque
from knu_rl_env.grid_adventure import GridAdventureAgent, make_grid_adventure, evaluate

class GridAdventureRLAgent(GridAdventureAgent):
    def __init__(self, load_q_table=False, q_table_filename='',epsilon=0.1):
        # 가능한 행동들 정의 (기본 행동)
        self.ACTION_LEFT = GridAdventureAgent.ACTION_LEFT
        self.ACTION_RIGHT = GridAdventureAgent.ACTION_RIGHT
        self.ACTION_FORWARD = GridAdventureAgent.ACTION_FORWARD
        self.ACTION_PICKUP = GridAdventureAgent.ACTION_PICKUP
        self.ACTION_DROP = GridAdventureAgent.ACTION_DROP
        self.ACTION_UNLOCK = GridAdventureAgent.ACTION_UNLOCK

        self.knu_agent_actions = [self.ACTION_LEFT, self.ACTION_RIGHT, self.ACTION_FORWARD, self.ACTION_PICKUP, self.ACTION_DROP,self.ACTION_UNLOCK]

        # 상태 공간 크기 정의
        self.state_size = (26, 26)
        # 행동 공간 크기 정의
        self.action_size = len(self.knu_agent_actions)
        self.key_flag = False
        self.key_size = 4
        self.current_key = None

        # 방향 공간 크기 정의
        self.direction_size = 4
        self.direction_mapping = {'AU': 0, 'AR': 1, 'AD': 2, 'AL': 3}

        # 문 상태 크기 정의 (잠김 : 0, 열림 : 1, 닫힘 : 2)
        self.door_state_size = (3, 3, 3)  # 초록문, 빨간문, 파랑문 각각 3가지 상태
        self.door_state=(0,0,0)

        # 현재 방향 초기화 (초기 설정 필요)
        self.current_direction = None

        # 학습률(alpha)
        self.alpha = 0.1
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
                self.q_table = np.zeros(self.state_size + (self.direction_size, self.key_size) + self.door_state_size + (self.action_size,))
        else:
            self.q_table = np.zeros(self.state_size + (self.direction_size, self.key_size) + self.door_state_size + (self.action_size,))

        # BFS 테이블 초기화
        self.bfs_table_flag = False
        self.bfs_table = np.zeros(self.state_size)
        # 목표 리스트
        self.goals = ['KB', 'DBC', 'KG', 'DGC', 'KR', 'DRC', 'G']
        self.current_goal_index = 0
        # 방향 매핑
        self.directions = ['U', 'R', 'D', 'L']  # 시계 방향 순서
        # 보상 값 설정
        self.goal_reward_value = 2000  # 목표 달성 보상
        self.lava_penalty_value = -20  # 용암에 대한 큰 음의 보상
        self.time_penalty = 0  # 시간 페널티
        self.move_reward_scale = 10  # 이동 보상 스케일링 계수
        self.observation = None
        self.reward_queue = deque()

        self.door_positions = {}
        self.list_item = [("KB","DBO"), ("KG","DGO"), ("KR","DRO")]

    def get_key_index(self, flag, key_type=None):
        """
        키 소유 여부 및 키 종류에 따라 인덱스를 반환하는 함수.
        :param flag: 키 소유 여부 (True/False)
        :param key_type: 현재 가지고 있는 키의 종류 ('KB', 'KG', 'KR') 또는 None
        :return: 키 상태 인덱스 (0: 없음, 1: 파란 키, 2: 초록 키, 3: 빨간 키)
        """
        if not flag or key_type is None:
            return 0
        key_mapping = {'KB': 1, 'KG': 2, 'KR': 3}
        return key_mapping.get(key_type, 0)
    
    def get_door_state_index(self, grid):
        """
        각 문의 상태(잠김, 열림, 닫힘)를 인덱스로 변환하는 함수.
        :param door_states: 각 문의 상태를 나타내는 리스트 (예: [0, 1, 2])
        :return: 문 상태 인덱스 튜플
        """

        suffix_mapping = {'C': 2, 'L': 0, 'O': 1}
        # suffix_mapping.get(suffix, -1)  # 해당되지 않는 경우 -1 반환

        #GP
        dg_state= grid[1][16]

        #B
        db_state=grid[1][12]

        #R
        dr_state=grid[4][24]


        result= (suffix_mapping.get(db_state[-1], 0),suffix_mapping.get(dg_state[-1], 0),suffix_mapping.get(dr_state[-1], 0))
        return result
    

    def find_agent_position(self, observation):
        """에이전트의 위치와 방향을 찾는 함수"""
        for i in range(self.state_size[0]):
            for j in range(self.state_size[1]):
                cell = observation[i][j]
                if cell in ['AL', 'AR', 'AU', 'AD']:
                    direction = self.direction_mapping[cell]
                    return (i, j), direction
        return None, None

    def find_faced_cell(self, current_position, direction,grid):
        direction_mapping = {
            0: (-1, 0),
            1: (0, 1),
            2: (1, 0),
            3: (0, -1)
        }
        next_position = (current_position[0] + direction_mapping[direction][0], current_position[1] + direction_mapping[direction][1])

        return grid[next_position[0]][next_position[1]]


    def is_value_in_grid(self, value, grid):
        for row in grid:
            if value in row:
                return True
        return False

    def custom_reward(self, current_position, action_index, next_position, grid, terminated, cnt,step_count):
        """커스텀 보상 함수"""
        goal_reward = 0
        time_penalty = self.time_penalty

        cnt_pos=cnt

        # 이동 보상 계산 (모든 단계에서 적용)
        move_reward = self.bfs_table[next_position[0], next_position[1]] * self.move_reward_scale

        # 이동 보상을 받은 위치의 값을 감소시켜 반복적인 보상 누적 방지
        self.bfs_table[next_position[0], next_position[1]] *= 0.9  # 5% 감소로 수정


        if self.knu_agent_actions[action_index] == self.ACTION_PICKUP:
            # 열쇠 목표 달성 확인
            keys = ["KB", "KG", "KR"]
            for k in keys:
                if not self.is_value_in_grid(k, grid):
                    if k not in self.reward_queue:
                        self.reward_queue.append(k)
                        if k == "KB" :
                            goal_reward+=self.goal_reward_value

                        elif k == "KG" and "KB" in self.reward_queue:
                            goal_reward+=self.goal_reward_value+self.goal_reward_value

                        elif k == "KR" and "KG" in self.reward_queue and "KB" in self.reward_queue:
                            goal_reward+=self.goal_reward_value+self.goal_reward_value+self.goal_reward_value


        if self.knu_agent_actions[action_index] == self.ACTION_UNLOCK:
            # 문 목표 달성 확인
            opened_doors = ["DBO", "DGO", "DRO"]
            for d in opened_doors:
                if self.is_value_in_grid(d, grid):
                    if d not in self.reward_queue:
                        self.reward_queue.append(d)
                        goal_reward += self.goal_reward_value

                        if d == "DGO" and "DBO" in self.reward_queue:
                            goal_reward+=self.goal_reward_value

                        elif d == "DRO" and "DGO" in self.reward_queue and "DBO" in self.reward_queue:
                            goal_reward+=self.goal_reward_value+self.goal_reward_value

        # 문열고 이동했하면 보상
        if self.knu_agent_actions[action_index] == self.ACTION_FORWARD:
            if next_position in self.door_positions:
                if self.door_positions[next_position] not in self.reward_queue:
                        self.reward_queue.append(str(self.door_positions[next_position]))
                        goal_reward += 500  # 추가 보상


        if self.knu_agent_actions[action_index] == self.ACTION_DROP:
                if current_position == (1,19):
                    if self.is_value_in_grid("KG", grid):
                            if "DGL" in self.reward_queue and not "KGD" in self.reward_queue:
                                goal_reward += 500
                                self.reward_queue.append("KGD")

                elif current_position == (1,13):
                    if self.is_value_in_grid("KB", grid):
                        if "DBL" in self.reward_queue and not "KBD" in self.reward_queue:
                            self.reward_queue.append("KBD")
                            goal_reward += 500
        # drop_penalty=0

        # # 문열고 이동 안했는데 버리면 패널티
        # # 일단 버리는거 보상은,,
        # if self.knu_agent_actions[action_index] == self.ACTION_DROP:
        #     if key_flag==1 and key_flag_next==0 :
        #         if not "DBO" in self.reward_queue or not "DGO" in self.reward_queue :
        #             drop_penalty =-100

        if terminated and next_position == (24, 24):
            if 'G' not in self.reward_queue:
                goal_reward += 15000 * (1 -step_count/2704)
                self.reward_queue.append('G')
            print(f"목표 달성: {next_position}")

        # 용암에 빠진 경우
        if terminated and self.observation[next_position[0]][next_position[1]] == 'L':
            terminal_penalty = self.lava_penalty_value  # 큰 음의 보상
        else:
            terminal_penalty = 0

        # 벽에 부딪혔을 때 페널티 부여
        if current_position == next_position:
            cnt_pos += 1
            wall_penalty = -1 *cnt_pos
            # wall_penalty = -0.1 *cnt_pos
        else:
            wall_penalty = 0
            cnt_pos = 0

        if goal_reward > 0:
            print(f"목표 보상: {goal_reward}, goals:{self.reward_queue}")

        temp_reward = goal_reward + move_reward

        total_reward = temp_reward + time_penalty + terminal_penalty + wall_penalty
        # print(f"temp_reward:{temp_reward}, time_penalty:{time_penalty}, terminal_penalty:{terminal_penalty}, wall_penalty:{wall_penalty}")
        return total_reward, cnt_pos

    def act(self, observation):
        """현재 상태에서 행동을 선택하는 함수"""
        position, direction = self.find_agent_position(observation)
        has_key = self.get_key_index(self.key_flag, self.current_key)
        door_state_index = self.get_door_state_index(observation)


        if np.random.rand() < self.epsilon:
            # 무작위 행동 선택
            action_index = np.random.choice(self.action_size)
        else:
            # 최적 행동 선택
            state_actions = self.q_table[position[0], position[1], direction, has_key,door_state_index[0],door_state_index[1],door_state_index[2], :]
            action_index = np.argmax(state_actions)
            # print(state_actions)
            # Q-값이 모두 동일한 경우 무작위 선택
            if np.all(state_actions == state_actions[0]):
                action_index = np.random.choice(self.action_size)
        # print(action_index,position,direction)

        action = self.knu_agent_actions[action_index]
        # print(f"행동: {action}, 행동 인덱스: {action_index}")
        # 키 소유 상태 업데이트
        if action == self.ACTION_PICKUP and not self.key_flag:
            faced_cell = self.find_faced_cell(position, direction, observation)
            if faced_cell in ["KB", "KG", "KR"]:
                self.key_flag = True
                self.current_key = faced_cell  # 현재 들고 있는 키 종류 설정

        if action == self.ACTION_DROP and self.key_flag:
            faced_cell = self.find_faced_cell(position, direction, observation)
            if faced_cell == "W":
                self.key_flag = False
                self.current_key = None  # 키를 버리면 현재 키 종류 초기화

        # print(f"행동: {action}, 행동 인덱스: {action_index},key_flag:{self.key_flag}")
        return action_index

    def update_q_table(self, state, action_index, direction_index, reward, next_state, next_direction, done, has_key, has_key_next,door_state_index, next_door_state_index):
        """Q-테이블 업데이트"""
        x, y = state
        b,g,r=door_state_index
        try:
            current_q = self.q_table[x, y, direction_index, has_key, b,g,r, action_index]    
        except IndexError: 
            print([x, y, direction_index, has_key, door_state_index, action_index])

        if done:
            target = reward  # 종료 상태에서는 다음 상태의 Q-값을 고려하지 않음
        else:
            next_x, next_y = next_state
            next_b,next_g,next_r=next_door_state_index
            next_max_q = np.max(self.q_table[next_x, next_y, next_direction, has_key_next,next_b,next_g,next_r, :])  # 다음 상태의 방향을 고려
            target = reward + self.gamma * next_max_q
        self.q_table[x, y, direction_index, has_key,b,g,r, action_index] += self.alpha * (target - current_q)

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

    def state_find_door(self, grid):
        door_positions = {}
        for i in range(26):
            for j in range(26):
                if grid[i][j] in ["DBL", "DGL", "DRL"]:
                    door_positions[(i, j)]=grid[i][j]
        return door_positions

    def bfs_find_path(self, grid, start, goal):
        """BFS를 사용하여 최단 경로를 찾는 함수 (벽과 용암은 통과 불가능)"""
        rows, cols = len(grid), len(grid[0])
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 상, 하, 좌, 우
        visited = [[False for _ in range(cols)] for _ in range(rows)]
        parent = [[None for _ in range(cols)] for _ in range(rows)]
        queue = deque([start])
        visited[start[0]][start[1]] = True

        while queue:
            current = queue.popleft()

            if current == goal:
                break

            for d in directions:
                new_row, new_col = current[0] + d[0], current[1] + d[1]
                if (0 <= new_row < rows and 0 <= new_col < cols and not visited[new_row][new_col]):
                    cell = grid[new_row][new_col]
                    if cell != 'W' and cell != 'L':  # 벽과 용암은 통과 불가능
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

    def generate_and_print_combined_table(self, grid, current_position):
        """
        BFS 테이블과 최단 경로 이진 테이블을 합산하여 출력하는 메소드

        :param grid: 2D 리스트, 그리드 상태
        :param current_position: tuple (x, y), 에이전트의 현재 위치
        :param goal: tuple (x, y), 목표 위치
        :return: None
        """
        goal = (24, 24)

        # 1. BFS 테이블 업데이트
        self.update_bfs_table(grid)

        # 2. 최단 경로 찾기
        path = self.bfs_find_path(grid, current_position, goal)
        if not path:
            print("최단 경로를 찾을 수 없습니다.")
            return

        # 3. 이진 테이블 생성 (최단 경로 셀을 1로)
        binary_path = np.zeros_like(self.bfs_table, dtype=int)
        for pos in path:
            binary_path[pos[0], pos[1]] = 1
        kg_position = self.find_goal_position("KG")

        path2 = self.bfs_find_path(grid, current_position, kg_position)
        if not path:
            print("최단 경로를 찾을 수 없습니다.")
            return

        # 3. 이진 테이블 생성 (최단 경로 셀을 1로)
        binary_path2 = np.zeros_like(self.bfs_table, dtype=int)
        for pos in path2:
            binary_path2[pos[0], pos[1]] = 1

        # 4. 테이블 합산
        combined_table = self.bfs_table *10 + binary_path + (binary_path2 *0.2)
        # 4. 테이블 합산

        combined_table = combined_table *0.1
        # 4. 테이블 합산


        return combined_table

    def train(self, num_episodes=10000):
        # 환경 생성 시 max_steps=2704로 설정
        env = make_grid_adventure(show_screen=False)
        episode_rewards = []
        episode = 0


        while episode < num_episodes:
            self.reward_queue.clear()

            self.key_flag = False
            self.current_key = None
            episode += 1
            observation, _ = env.reset()
            (state, direction) = self.find_agent_position(observation)
            done = False
            total_reward = 0
            self.current_goal_index = 0  # 에피소드 시작 시 목표 인덱스 초기화
            grid = observation  # 현재 그리드 초기화

            self.observation = observation

            self.bfs_table = self.generate_and_print_combined_table(grid,state)  # BFS 테이블 업데이트

            self.door_positions = self.state_find_door(grid)

            step_count = 0  # 스텝 수 초기화
            cnt_pos = 0
            while not done:

                has_key = self.get_key_index(self.key_flag, self.current_key)
                door_state_index=self.get_door_state_index(observation)

                action_index = self.act(observation)
                action = self.knu_agent_actions[action_index]
                current_position = state
                current_direction = direction
                # print(f"action {action}, action _index {action_index}")
                has_key_next = self.get_key_index(self.key_flag, self.current_key)
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
                next_door_state_index=self.get_door_state_index(observation_next)

                step_count += 1  # 스텝 수 증가


                # 커스텀 보상 계산
                custom_reward , cnt_pos = self.custom_reward(
                    current_position, action_index, state, grid, done, cnt_pos ,step_count

                )

                # Q-테이블 업데이트
                self.update_q_table(
                    current_position,
                    action_index,
                    current_direction,
                    custom_reward,
                    state,
                    next_direction,  # 다음 방향 정보 전달
                    done,
                    has_key,
                    has_key_next,
                    door_state_index,
                    next_door_state_index
                )
                total_reward += custom_reward



                if done:
                    print(f"Episode: {episode}, Total Reward: {total_reward:.2f}, Epsilon: {self.epsilon:.4f}, Goals Achieved: {self.reward_queue}, Steps: {step_count}, Current Position: {current_position}")
                    break
                            # 탐색률 감소 (선형 감소)
            if self.epsilon > self.epsilon_min:
                self.epsilon -= self.epsilon_decay
            episode_rewards.append(total_reward)

            # 에피소드마다 Q-테이블 저장
            self.save_q_table()

        return self, episode_rewards
    def find_goal_position(self, goal):
        for i in range(self.state_size[0]):
            for j in range(self.state_size[1]):
                if self.observation[i][j] == goal:
                    return (i, j)
        return None
if __name__ == '__main__':
    # 학습 재개를 원하면 load_q_table=True로 설정
    print(np.__version__)
    agent = GridAdventureRLAgent(load_q_table=True, q_table_filename='2024.10.29/last_q-table_v2.pkl',epsilon=0.3)
    trained_agent, episode_rewards = agent.train(num_episodes=10000)

    # 학습된 에이전트를 로드하여 평가 실행
    evaluate(trained_agent)
