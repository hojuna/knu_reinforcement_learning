import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import math
from collections import namedtuple
import random

from knu_rl_env.grid_survivor import make_grid_survivor, GridSurvivorAgent  # 환경 임포트
from DQN_network import DuelingDQN
from memory import PrioritizedReplayMemory


class DuelingDQNAgent(GridSurvivorAgent):
    def __init__(self, input_channels, grid_height, grid_width, num_actions, device, config):
        self.device = device
        self.policy_net = DuelingDQN(input_channels, grid_height, grid_width, num_actions).to(device)
        self.target_net = DuelingDQN(input_channels, grid_height, grid_width, num_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.num_actions = num_actions
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config["learning_rate"])
        self.memory = PrioritizedReplayMemory(config["memory_capacity"], alpha=config["alpha"])
        self.batch_size = config["batch_size"]
        self.gamma = config["gamma"]
        self.target_update = config["target_update"]
        self.steps_done = 0
        self.beta_start = config["beta_start"]
        self.beta_frames = config["beta_frames"]
        self.visit_table = np.zeros((grid_height, grid_width), dtype=np.int32)
        

    def reset_visit_table(self):
        self.visit_table = np.zeros((35,35), dtype=np.int32)

    def beta_by_frame(self, frame_idx):
        return min(1.0, self.beta_start + frame_idx * (1.0 - self.beta_start) / self.beta_frames)

    def load_model(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['model_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_state_dict'])
        print(f"모델이 {checkpoint_path}에서 로드되었습니다. (에피소드: {checkpoint['episode']})")


    # 7. 체크포인트 저장 함수 정의
    def save_checkpoint(self,filepath):
        checkpoint = {
            'model_state_dict': agent.policy_net.state_dict(),
            'target_state_dict': agent.target_net.state_dict(),  # 타겟 네트워크 추가
        }
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved at episode {episode} to {filepath}")


    def select_action(self, state):
        grid, hit_points, agent_direction, agent_position = state
        grid_tensor = torch.from_numpy(np.array(grid, dtype=np.float32)).unsqueeze(0).to(self.device)
        hit_points_tensor = torch.from_numpy(np.array(hit_points, dtype=np.float32)).unsqueeze(0).to(self.device)
        agent_direction_tensor = torch.from_numpy(np.array(agent_direction, dtype=np.float32)).unsqueeze(0).to(self.device)
        agent_position_tensor = torch.from_numpy(np.array(agent_position, dtype=np.float32)).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            q_values = self.policy_net(grid_tensor, hit_points_tensor, agent_direction_tensor, agent_position_tensor)
            action = q_values.argmax(dim=1).item()
        return action

    def act(self, state, epsilon):
        if random.random() < epsilon:
            return random.randint(0, self.num_actions - 1)
        else:
            return self.select_action(state)

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return None  # 충분한 경험이 없음

        beta = self.beta_by_frame(self.steps_done)
        transitions, idxs, is_weights = self.memory.sample(self.batch_size, beta=beta)
        batch = self.memory.Transition(*zip(*transitions))

        # 상태 및 다음 상태 분리
        state_batch = batch.state
        action_batch = torch.tensor(batch.action, dtype=torch.int64).to(self.device)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float32).to(self.device)
        next_state_batch = batch.next_state
        done_batch = torch.tensor(batch.done, dtype=torch.float32).to(self.device)
        is_weights = torch.tensor(is_weights, dtype=torch.float32).to(self.device)

        # 상태 텐서 변환
        grid_batch = torch.from_numpy(np.array([s[0] for s in state_batch], dtype=np.float32)).to(self.device)
        hit_points_batch = torch.from_numpy(np.array([s[1] for s in state_batch], dtype=np.float32)).to(self.device)
        agent_direction_batch = torch.from_numpy(np.array([s[2] for s in state_batch], dtype=np.float32)).to(self.device)
        agent_position_batch = torch.from_numpy(np.array([s[3] for s in state_batch], dtype=np.float32)).to(self.device)

        # 다음 상태 텐서 변환
        next_grid_batch = torch.from_numpy(np.array([s[0] for s in next_state_batch], dtype=np.float32)).to(self.device)
        next_hit_points_batch = torch.from_numpy(np.array([s[1] for s in next_state_batch], dtype=np.float32)).to(self.device)
        next_agent_direction_batch = torch.from_numpy(np.array([s[2] for s in next_state_batch], dtype=np.float32)).to(self.device)
        next_agent_position_batch = torch.from_numpy(np.array([s[3] for s in next_state_batch], dtype=np.float32)).to(self.device)

       
        # 현재 Q 값
        state_action_values = self.policy_net(grid_batch, hit_points_batch, agent_direction_batch, agent_position_batch).gather(1, action_batch.unsqueeze(1)).squeeze(1)

        # 타겟 Q 값 계산
        with torch.no_grad():
            next_state_values = self.target_net(next_grid_batch, next_hit_points_batch, next_agent_direction_batch, next_agent_position_batch).max(1)[0]
            expected_state_action_values = reward_batch + (1 - done_batch) * self.gamma * next_state_values

        # TD 오류 계산
        td_errors = state_action_values - expected_state_action_values
        abs_errors = td_errors.abs().detach().cpu().numpy()  # 우선순위 업데이트를 위해 절대값 사용

        # 손실 계산 (Huber 손실 사용)
        losses = F.smooth_l1_loss(state_action_values, expected_state_action_values, reduction='none')
        loss = (is_weights * losses).mean()

        # 그래디언트 업데이트
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        # 우선순위 업데이트
        for idx, error in zip(idxs, abs_errors):
            self.memory.update(idx, error)

        return loss.item()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def encode_grid(self,grid):
        """그리드 데이터를 원-핫 인코딩하여 4채널 텐서로 변환 (B, H, K, Agent Direction)"""
        channels = {
            'B': 0,   # Bee
            'H': 1,   # Hornet
            'K': 2,   # Killer Bee
            'W': 3,    # Wall
            'A': 4    # Agent
        }
        grid_encoded = np.zeros((len(channels), grid.shape[0], grid.shape[1]), dtype=np.float32)

        # 기존 객체들 인코딩
        for symbol, idx in channels.items():
            if symbol in ['B', 'H', 'K', 'W']:
                grid_encoded[idx][grid == symbol] = 1.0
            elif symbol.startswith('A'):
                grid_encoded[idx][grid == symbol] = 1.0

        return grid_encoded
    
    def encode_agent_direction(self,direction):
        direction_mapping = {
            'UP': [1, 0, 0, 0],
            'DOWN': [0, 1, 0, 0],
            'LEFT': [0, 0, 1, 0],
            'RIGHT': [0, 0, 0, 1]
        }
        direction_one_hot = direction_mapping.get(direction, [0, 1, 0, 0])  # 기본값 'DOWN'
        return np.array(direction_one_hot, dtype=np.float32)

    def encode_agent_position(self,position, grid_height=50, grid_width=50):
        """에이전트의 위치를 벡터로 정규화"""
        x, y = position
        pos_vector = np.zeros(2, dtype=np.float32)
        pos_vector[0] = x / grid_height  # 정규화
        pos_vector[1] = y / grid_width
        return pos_vector

    def extract_agent_info(self,grid):
        """그리드에서 에이전트의 위치와 방향을 추출합니다."""
        direction_symbols = {'AU': 'UP', 'AD': 'DOWN', 'AL': 'LEFT', 'AR': 'RIGHT'}
        positions = np.argwhere(np.isin(grid, list(direction_symbols.keys())))
        if len(positions) == 0:
            # 에이전트가 그리드에 없을 경우
            return (None, None)
        x, y = positions[0]
        symbol = grid[x, y]
        direction = direction_symbols.get(symbol, 'DOWN')
        return (x, y), direction

    def normalize_hit_points(self,hit_points, max_hp=100):
        """체력 정보를 0과 1 사이로 정규화"""
        return np.array([hit_points / max_hp], dtype=np.float32)

    def preprocess_state(self,state):
        grid=state['grid']
        hit_points=state['hit_points']  

        """환경 상태를 신경망 입력 형태로 전처리"""
        grid_encoded = self.encode_grid(grid)  # (채널, 높이, 너비)
        hit_points_normalized = self.normalize_hit_points(hit_points)  # (1,)
        
        # 에이전트의 위치와 방향을 grid에서 추출
        agent_pos, agent_dir = self.extract_agent_info(grid)
        if agent_pos is None:
            print("에이전트가 없음 뭐임?")
        else:
            agent_pos_vector = self.encode_agent_position(agent_pos)
            agent_dir_encoded = self.encode_agent_direction(agent_dir)
        
        return grid_encoded, hit_points_normalized, agent_dir_encoded, agent_pos_vector


    def calculate_reward(self,state, next_state, done, step):
        total_reward = 0

        base_reward = -0.1

        # survival_reward = 0.01 * (step / 100)
        survival_reward=0

        previous_bees = np.sum(state['grid'] == 'B')
        current_bees = np.sum(next_state['grid'] == 'B')
        rescued_bees = previous_bees - current_bees

        bee_reward = rescued_bees * 100

        # if 'B'==self.find_forward_obj(next_state):
        #     bee_reward+=5


        previous_hornet = np.sum(state['grid'] == 'H')
        current_hornet = np.sum(next_state['grid'] == 'H')
        rescued_hornet = previous_hornet - current_hornet
        if state['hit_points']>=40:
            hornet_penalty = rescued_hornet * -5
        else:
            hornet_penalty = rescued_hornet * -10


        health_reward = 0
        # health_diff = next_state['hit_points'] - state['hit_points']
        # health_reward = health_diff * 0.1


        early_termination = 0

        if current_bees != 0 and done:
            early_termination-=20


        position, direction = self.extract_agent_info(state['grid'])
        next_position, next_direction = self.extract_agent_info(next_state['grid'])
        collision_penalty = -0.1 if position == next_position  else 0
        
        first_visit_reward=0
        if self.visit_table[position]==1:
            first_visit_reward=1
            self.visit_table[position]=0
        else:
            self.visit_table[position]-=1

        bee_distance_reward = 0
        # if 'B' in next_state['grid']:
        #     min_distance = self.min_distance(next_state, next_position, 'B')
        #     if min_distance is not None:
        #         bee_distance_reward += 0.1 / (min_distance + 1e-6)

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


        if 0 == current_bees and done:
            total_reward += 500 / (step + 1e-6)

        total_reward = (base_reward + survival_reward + bee_reward + hornet_penalty + 
                    health_reward + early_termination + collision_penalty + 
                     bee_distance_reward + hornet_distance_penalty +
                    killerbee_distance_penalty+first_visit_reward)

        return total_reward
        
        