# A2C 에이전트 정의
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from collections import deque
import random
from knu_rl_env.grid_survivor import GridSurvivorAgent, make_grid_survivor
import wandb
from A2CNetwork import A2CNetwork


class A2CAgent(): # GridSurvivorAgent 상속 제거
    def __init__(self, state_size, save_dir=f"/home/comoz/main_project/knu_reinforcement_learning/project2/A2C/A2C_v9/save_model"):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")  # 디바이스 확인용 출력
        
        # 네트워크와 옵티마이저를 GPU로 이동
        self.network = A2CNetwork(state_size, 128, 3).to(self.device)
        self.optimizer = optim.RMSprop(
            self.network.parameters(),
            lr=0.00025,
            alpha=0.95,
            eps=1e-5,
            momentum=0.0,
            centered=True
        )
        
        # numpy 배열들은 CPU에 유지
        self.visit_table = np.ones((35,35))
        self.isolation_table = np.zeros((35,35))
        
        # 경험 저장용 리스트들
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.raw_states = []
        
        # 하이퍼라미터
        self.gamma = 0.99
        self.entropy_coef = 0.01
        self.value_coef = 0.5
        
        # 스케줄러도 조정
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=800,          # 1000 → 800 (더 빠른 주기)
            eta_min=5e-5        # 1e-5 → 5e-5 (최소값 살짝 높임)
        )
        
        # 모델 저장 경로
        self.save_dir = save_dir
    
    def visit_table_reset(self):
        self.visit_table = np.ones((35,35))
        self.isolation_table = np.zeros((35,35))

    def preprocess_state(self,state):
        # 위치와 방향 분리
        cell_types = {'E':0, 'W':1, 'B':2, 'H':3, 'K':4, 'A':5}  # 6차원
        directions = {'L':0, 'R':1, 'U':2, 'D':3}  # 추가 방향 정보
        grid = state['grid']
        N = len(grid)
        
        # 기본 그리드 정보 (6차원)
        grid_tensor = torch.zeros(N, N, 6, device=self.device)  # 6개 셀 타입
        # 방향 정보 (4차원)
        direction_tensor = torch.zeros(4, device=self.device)    # 4개 방향
        
        agent_pos = None
        
        for i in range(N):
            for j in range(N):
                cell = grid[i][j]
                if cell.startswith('A'):  # 에이전트 셀
                    grid_tensor[i, j, 5] = 1  # A 표시
                    direction_tensor[directions[cell[1]]] = 1  # 방향 시
                    agent_pos = (i, j)
                else:
                    grid_tensor[i, j, cell_types[cell]] = 1
        
        # 1D로 변환
        grid_flat = grid_tensor.flatten()
        
        # 최종 상태 벡터 구성
        state_vector = torch.cat([
            grid_flat,
            direction_tensor,
            torch.tensor([state['hit_points'] / 100.0], device=self.device)
        ])
        
        return state_vector

    def find_forward_obj(self, grid):
        """
        에이전트의 방향에 따라 앞으로 있는 객체를 탐색하고, 우선순위에 따라 가장 높은 우선순위의 객체를 반환합니다.
        
        Parameters:
        - grid (2D numpy array): 현재 그리드 상태. 각 셀은 'W', 'B', 'H', 'K', 'AU', 'AD', 'AL', 'AR' 등으로 표시됩니다.
        
        Returns:
        - target (str or None): 우선순위가 가장 높은 객체의 기호 또는 객체가 없으면 None.
        """
        # 객체 우선순위 정의 (낮은 인덱스일수록 높은 우선순위)
        priority = ['K', 'H','B','W','E' ]
        
        # 에이전트의 위치와 방향 추출
        position, direction = self.extract_agent_info(grid)
        
        if position is None or direction is None:
            return None
        
        x, y = position
        grid_height, grid_width = grid.shape
        
        # 방향별 탐색할 상대적 셀 좌표 정의
        direction_offsets = {
            'UP': [(-2, 0), (-1, -1), (-1, 1),(-1,0)],
            'DOWN': [(2, 0), (1, -1), (1, 1),(1,0)],
            'RIGHT': [(0, 2), (-1, 1), (1, 1),(0,1)],
            'LEFT': [(0, -2), (-1, -1), (1, -1),(0,-1)]
        }
        
        # 현재 방향에 따른 탐색할 셀 좌표 가져오기
        offsets = direction_offsets.get(direction.upper())
        if offsets is None:
            return None  # 유효하지 않은 방향일 경우
        
        found_symbols = []
        
        for dx, dy in offsets:
            cx, cy = x + dx, y + dy
            # 그리드 범위 내인지 확인
            if 0 <= cx < grid_height and 0 <= cy < grid_width:
                cell_symbol = grid[cx][cy]
                if cell_symbol in priority:
                    found_symbols.append(cell_symbol)
        
        if not found_symbols:
            return None  # 탐색된 객체가 없을 경우
        
        # 우선순위에 따라 가장 높은 우선순위의 객체 선택
        for symbol in priority:
            if symbol in found_symbols:
                return symbol
        
        return None

    def find_forward_obj_H(self, grid):
        """
        에이전트의 방향에 따라 앞으로 있는 객체를 탐색하고, 우선순위에 따라 가장 높은 우선순위의 객체를 반환합니다.
        
        Parameters:
        - grid (2D numpy array): 현재 그리드 상태. 각 셀은 'W', 'B', 'H', 'K', 'AU', 'AD', 'AL', 'AR' 등으로 표시됩니다.
        
        Returns:
        - target (str or None): 우선순위가 가장 높은 객체의 기호 또는 객체가 없으면 None.
        """
        # 객체 우선순위 정의 (낮은 인덱스일수록 높은 우선순위)
        priority = ['H','B','W','E' ]
        
        # 에이전트의 위치와 방향 추출
        position, direction = self.extract_agent_info(grid)
        
        if position is None or direction is None:
            return None
        
        x, y = position
        grid_height, grid_width = grid.shape
        
        # 방향별 탐색할 상대적 셀 좌표 정의
        direction_offsets = {
            'UP': [(-2, 0), (-1, -1), (-1, 1),(-1,0)],
            'DOWN': [(2, 0), (1, -1), (1, 1),(1,0)],
            'RIGHT': [(0, 2), (-1, 1), (1, 1),(0,1)],
            'LEFT': [(0, -2), (-1, -1), (1, -1),(0,-1)]
        }
        
        # 현재 방향에 따른 탐색할 셀 좌표 가져오기
        offsets = direction_offsets.get(direction.upper())
        if offsets is None:
            return None  # 유효하지 않은 방향일 경우
        
        found_symbols = []
        
        for dx, dy in offsets:
            cx, cy = x + dx, y + dy
            # 그리드 범위 내인지 확인
            if 0 <= cx < grid_height and 0 <= cy < grid_width:
                cell_symbol = grid[cx][cy]
                if cell_symbol in priority:
                    found_symbols.append(cell_symbol)
        
        if not found_symbols:
            return None  # 탐색된 객체가 없을 경우
        
        # 우선순위에 따라 가장 높은 우선순위의 객체 선택
        for symbol in priority:
            if symbol in found_symbols:
                return symbol
        
        return None

    def extract_agent_info(self,grid):
        direction_symbols = {'AU': 'UP', 'AD': 'DOWN', 'AL': 'LEFT', 'AR': 'RIGHT'}
        positions = np.argwhere(np.isin(grid, list(direction_symbols.keys())))

        if len(positions) == 0:
            return (None, None)

        x, y = positions[0]
        symbol = grid[x, y]
        direction = direction_symbols.get(symbol, 'DOWN')

        return (x, y), direction


    def get_valid_actions(self, state):
        """유효한 행동 마스크 생성 - GPU 지원"""
        grid = state['grid']
        valid_actions = torch.ones(3, device=self.device)  # [LEFT, RIGHT, FORWARD]
        
        # 에이전트 위치와 방향 파악
        position, direction = self.extract_agent_info(grid)
        if position is None or direction is None:
            return valid_actions
        
        x, y = position
        # FORWARD 액션이 유효한지 확인
        next_pos = self._get_next_position(position, direction)

        if not self._is_valid_move(next_pos, grid,state['hit_points'],position):
            valid_actions[2] = 0  # FORWARD 불가능

        return valid_actions
    
    def _get_next_position(self, position, direction):
        """주어진 방향으로 한 칸 앞의 위치 반환"""
        x, y = position
        if direction == 'UP':
            return (x-1, y)
        elif direction == 'DOWN':
            return (x+1, y)
        elif direction == 'LEFT':
            return (x, y-1)
        elif direction == 'RIGHT':
            return (x, y+1)
        return position
    
    def _is_valid_move(self, next_pos, grid,hit_points,position):

        self.visit_table[position] -=1
        """해당 위치로 이동이 가능한지 확인"""
        x, y = next_pos
        # 격자 범위 체크
        if not (0 <= x < len(grid) and 0 <= y < len(grid[0])):
            return False

        if (self.isolation_table[position] > 5 and grid[x][y] == 'E') or (self.visit_table[position] < -30 and grid[x][y] == 'E'):
            #죽을 위험을 감수
            self.isolation_table[position] =0
            self.visit_table[position] = 0
            return True
        
        # 벽 체크
        if self.find_forward_obj(grid) == 'K' or (hit_points <= 20 and self.find_forward_obj_H(grid) == 'H'):
            if grid[x][y] == 'W' and self.visit_table[position] < -5:
                self.visit_table[position] = 0
                self.isolation_table[position] += 1
                return True
            return False

        if grid[x][y] == 'W':
            return False
        
        return True


    def _is_valid_wall(self,position,direction,grid):
        next_pos = self._get_next_position(position, direction)
        if grid[next_pos[0]][next_pos[1]] == 'W':
            return True
        return False
    
    def calculate_reward(self, state, next_state, done, step,action):
        total_reward = 0

        # base_reward = -0.1
        base_reward=0

        # survival_reward = 0.01 * (step / 100)
        survival_reward=0

        previous_bees = np.sum(state['grid'] == 'B')
        current_bees = np.sum(next_state['grid'] == 'B')
        rescued_bees = previous_bees - current_bees

        bee_reward = rescued_bees * 100


        hornet_penalty=0

        health_reward = 0

        



        early_termination = 0

        if done:
            if current_bees != 0:  # 벌이 남았는데 죽은 경우
                early_termination -= 200  # 벌 구출 보상(100)의 2배
            
            if next_state['hit_points'] <= 0:  # 체력이 0 이하로 떨어져 죽은 경우
                early_termination -= 150  # 추가 패널티
            
            if self.visit_table.min() < -100:  # 같은 자리를 너무 많이 방문해서 죽은 경우
                early_termination -= 100  # 탐색 실패에 대한 패널티

        # 체력 관련 보상/패널티 추가
        health_diff = next_state['hit_points'] - state['hit_points']
        if health_diff < 0:  # 체력이 감소했을 때
            health_reward = health_diff * 2  # 체력 감소에 대한 페널티 강화
        
        position, direction = self.extract_agent_info(state['grid'])
        next_position, next_direction = self.extract_agent_info(next_state['grid'])
        
        # collision_penalty = -0.1 if position == next_position and self else 0
        collision_penalty = 0
        
        # if position == next_position:
        #     self.visit_table[position]-=1
        #     collision_penalty=self.visit_table[position]/100

        first_visit_reward=0
        # self.visit_table[position]-=1

        # 벌과의 거리에 따른 보상 계산
        bee_distance_reward = 0
        position, _ = self.extract_agent_info(next_state['grid'])
        
        if 'B' in next_state['grid']:
            bee_positions = np.argwhere(next_state['grid'] == 'B')
            if len(bee_positions) > 0:
                # 실제 경로 거리 계산
                path_distances = []
                for bee_pos in bee_positions:
                    dist = self.get_path_distance(next_state['grid'], position, bee_pos)
                    if dist != float('inf'):  # 도달 가능한 경우만 고려
                        path_distances.append(dist)
                
                if path_distances:  # 도달 가능한 벌이 있는 경우
                    # 가장 가까운 3마리의 벌에 대한 보상 계산
                    closest_distances = sorted(path_distances)[:3]
                    
                    for dist in closest_distances:
                        if dist <= 1:  # 바로 옆에 있는 벌
                            bee_distance_reward += 1.5
                        elif dist <= 3:  # 가까운 거리의 벌
                            bee_distance_reward += 0.5 / (dist + 1)
                        else:  # 먼 거리의 벌
                            bee_distance_reward += 0.1 / (dist + 1)
                
                    # 이전 상태와의 거리 화 계산
                    if 'B' in state['grid']:
                        prev_position, _ = self.extract_agent_info(state['grid'])
                        prev_bee_positions = np.argwhere(state['grid'] == 'B')
                        
                        prev_path_distances = []
                        for bee_pos in prev_bee_positions:
                            dist = self.get_path_distance(state['grid'], prev_position, bee_pos)
                            if dist != float('inf'):
                                prev_path_distances.append(dist)
                        
                        if prev_path_distances:
                            prev_min_dist = min(prev_path_distances)
                            curr_min_dist = min(path_distances)
                            
                            # 실제 경로 거리가 줄어들었다면 추가 보상
                            if curr_min_dist < prev_min_dist:
                                bee_distance_reward += 0.25

        # 벌의 수에 따른 가중치 적용
        num_bees = len(np.argwhere(next_state['grid'] == 'B'))

        if num_bees > 0:
            # 벌이 적게 남을수록 각 벌에 대한 보상을 증가
            bee_distance_reward *= (1 + (50 - num_bees) / 50)


        pos=self._get_next_position(position,direction)

        forward_reward=0
        if (state['grid'][pos] == 'E' and
            action==2):
            
            if self.isolation_table[position] > 10 :
                forward_reward += 10.0
            else:
                forward_reward -= 1

        if (self.visit_table[position] < -5 and 
            action == 2 and 
            self._is_valid_wall(position, direction, state['grid'])):  
                forward_reward += 5
        hornet_distance_penalty = 0
        # if 'H' in next_state['grid']:
        #     min_distance = self.min_distance(next_state, next_position, 'H')
        #     if min_distance is not None:
            
        #         hornet_distance_penalty -= 1 / (min_distance + 1e-6)

        killerbee_distance_penalty = 0
        # if 'K' in next_state['grid']:
        #     min_distance = self.min_distance(next_state, next_position, 'K')
        #     if min_distance is not None:
        #         killerbee_distance_penalty -= 1 / (min_distance + 1e-6)


        # if 0 == current_bees and done:
        #     total_reward += 500 / (step + 1e-6)

        total_reward = (base_reward + survival_reward + bee_reward + hornet_penalty + 
                    health_reward + early_termination + collision_penalty + 
                     bee_distance_reward + hornet_distance_penalty +
                    killerbee_distance_penalty+first_visit_reward)

        return total_reward
        

    def get_path_distance(self, grid, start, goal):
        """BFS를 사용하여 실제 이동 가능한 최단 경로 거리를 계산"""
        if tuple(start) == tuple(goal):
            return 0
        
        rows, cols = grid.shape
        queue = [(start, 0)]  # (위치, 거리)
        visited = {tuple(start)}
        
        # 이동 가능한 방향 (상하좌우)
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        while queue:
            (x, y), dist = queue.pop(0)
            
            # 현재 위치가 목표지점이면 거리 반환
            if (x, y) == tuple(goal):
                return dist
            
            # 4방향 탐색
            for dx, dy in directions:
                next_x, next_y = x + dx, y + dy
                
                # 그리드 범위 체크
                if 0 <= next_x < rows and 0 <= next_y < cols:
                    next_pos = (next_x, next_y)
                    # 방문하지 않았고 벽이 아닌 경우
                    if next_pos not in visited and grid[next_x, next_y] != '#':
                        visited.add(next_pos)
                        queue.append((next_pos, dist + 1))
        
        # 경로가 없는 경우
        return float('inf')

    def act(self, state):

        #평가용 act로 내부적으로 visit_table을 업데이트 해줌

        position, direction = self.extract_agent_info(state['grid'])
        self.visit_table[position]-=1

        """행동 선택"""
        state_tensor = self.preprocess_state(state)
        valid_actions = self.get_valid_actions(state)
        
        with torch.no_grad():
            action_probs, value = self.network(state_tensor)

                        # 불가능한 행동의 확률을 0으로 만들고 재정규화
            masked_probs = action_probs * valid_actions
            if masked_probs.sum() > 0:  # 적어도 하나의 유효한 행동이 있는 경우
                masked_probs = masked_probs / masked_probs.sum()
            else:
                # 모든 행동이 불가능한 경우 (발생하면 안되지만 안전장치)
                masked_probs = action_probs
            
            # 행동 샘플링
            dist = Categorical(masked_probs)
            action = dist.sample()
            
            next_position = self._get_next_position(position,direction)
            if self.isolation_table[next_position] > 10 and state['grid'][next_position] == 'E':
                return 2

            elif self.visit_table[position] < -5 and self._is_valid_wall(position,direction,state['grid']):
                self.visit_table[position]=0
                self.isolation_table[position]+=1
                return 2
        
        return action.item()

    def visit_table_update(self,state,next_state):
        position, _ = self.extract_agent_info(state['grid'])
        next_position, _ = self.extract_agent_info(next_state['grid'])

        collision_penalty=0
        if position == next_position:   
            self.visit_table[position]-=1
            collision_penalty=self.visit_table[position]/10

        return collision_penalty
    
    def train_act(self, state):
        """행동 선택"""
        state_tensor = self.preprocess_state(state)
        valid_actions = self.get_valid_actions(state)
        
        with torch.no_grad():
            action_probs, value = self.network(state_tensor)

                        # 불가능한 행동의 확률을 0으로 만들고 재정규화
            masked_probs = action_probs * valid_actions
            if masked_probs.sum() > 0:  # 적어도 하나의 유효한 행동이 있는 경우
                masked_probs = masked_probs / masked_probs.sum()
            else:
                # 모든 행동이 불가능한 경우 (발생하면 안되지만 안전장치)
                masked_probs = action_probs
            
            # 행동 샘플링
            dist = Categorical(masked_probs)
            action = dist.sample()
        
        
        # 경험 저장
        self.states.append(state_tensor)
        self.raw_states.append(state)
        self.actions.append(action)
        self.values.append(value)
        
        return action.item()
    
    def update(self, reward, done):
        """경험 업데이트"""
        self.rewards.append(reward)
        self.dones.append(done)
        
        if done:
            loss = self.learn()
            self.reset_episode()
            return loss
        return 0
    
    def learn(self):
        """A2C 학습"""
        # 텐서들을 GPU로 이동
        returns = torch.tensor(self.rewards, device=self.device)
        returns = torch.clamp(returns, -100, 100)
        
        states = torch.stack(self.states)  # 이미 GPU에 있음
        actions = torch.stack(self.actions).to(self.device)
        values = torch.stack(self.values)  # 이미 GPU에 있음
        
        # 유효 행동 마스크를 GPU로 이동
        valid_actions_batch = torch.stack([self.get_valid_actions(s) for s in self.raw_states])
        
        # 네트워크 연산 (이미 GPU에서 수행됨)
        action_probs, state_values = self.network(states)
        
        # GPU에서 연산 수행
        masked_probs = action_probs * valid_actions_batch
        masked_probs = masked_probs / (masked_probs.sum(dim=1, keepdim=True) + 1e-8)
        
        advantages = returns - values.squeeze()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        advantages = torch.clamp(advantages, -10, 10)
        
        dist = Categorical(masked_probs)
        log_probs = dist.log_prob(actions)
        log_probs = torch.clamp(log_probs, -10, 2)
        policy_loss = -(log_probs * advantages.detach()).mean()
        
        value_loss = F.smooth_l1_loss(values.squeeze(), returns)
        
        entropy = dist.entropy()
        entropy = torch.clamp(entropy, -10, 2)
        entropy_loss = -self.entropy_coef * entropy.mean()
        
        total_loss = policy_loss + self.value_coef * value_loss + entropy_loss
        
        if total_loss.item() > 1000:
            return total_loss.item()
        
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
        self.optimizer.step()
        self.scheduler.step()
        
        return total_loss.item()
    
    def reset_episode(self):
        """에피소드 버퍼 초기화"""
        self.states.clear()
        self.raw_states.clear()  # 원본 상태 버퍼도 초기화
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()
    
    def save(self, episode):
        """모델 저장 - GPU/CPU 호환성 추가"""
        torch.save({
            'episode': episode,
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, f"{self.save_dir}/a2c_checkpoint_{episode}.pth",
        _use_new_zipfile_serialization=False)  # 이전 버전과의 호환성을 위해
    
    def load(self, path):
        """모델 로드 - GPU/CPU 호환성 추가"""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # optimizer의 state를 GPU로 이동
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)
        return checkpoint['episode']
