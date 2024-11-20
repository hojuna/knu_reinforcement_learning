import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque, namedtuple
import random
import os
from knu_rl_env.grid_survivor import make_grid_survivor, evaluate, run_manual, GridSurvivorAgent
import time
import math
import wandb

from DQN_cnn import DQN
from DQN_memory2 import PrioritizedReplayMemory


class DQNAgent(GridSurvivorAgent):

    def __init__(self, input_channels, grid_height, grid_width, num_actions, device, config=None, epsilon_start = 1.0):
        self.device = device
        self.policy_net = DQN(input_channels,num_actions).to(device)
        self.target_net = DQN(input_channels,num_actions).to(device)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.num_actions = num_actions
        self.action_counts = np.zeros(num_actions, dtype=np.float32)
        
        if config is None:
            config = {
                "learning_rate": 1e-4,
                "gamma": 0.99,
                "batch_size": 128,
                "memory_capacity": 100000,
                "target_update_steps": 1000,
                "epsilon_end": 0.01,
                "epsilon_decay": 2000
            }
        
        self.memory = PrioritizedReplayMemory(
            config["memory_capacity"]
        )
        
        self.batch_size = config["batch_size"]
        self.gamma = config["gamma"]
        self.target_update_steps = config["target_update_steps"]
        self.update_step_counter = 0
        self.steps_done = 0
        self.epsilon_start = epsilon_start
        self.epsilon_end = config.get("epsilon_end", 0.01)
        self.epsilon_decay = config.get("epsilon_decay", 10000)
        self.epsilon = 0
        
        self.eval_scores = []
        self.eval_interval = config.get("eval_interval", 1000)

        # Optimizer 및 스케줄러 설정
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config["learning_rate"])
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10000, gamma=0.1)
        
        self.beta_start = config.get("beta_start", 0.4)
        self.beta_frames = config.get("beta_frames", 100000)
        self.frame = 1  # 학습 프레임 수 카운트

        self.visit_table=np.ones((35,35))

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def get_epsilon(self):
        
        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * math.exp(-1. * self.steps_done / self.epsilon_decay)
        self.steps_done += 1

        if epsilon < self.epsilon_end:
            epsilon = self.epsilon_end
            
        return epsilon
    
    def act(self, grid, hit_points, direction, forward_obj, agent_position):
        self.epsilon = self.get_epsilon()
        if random.random() < self.epsilon:
            # 무작위 행동 선택 (탐험)
            action = random.randint(0, self.num_actions - 1)
        else:
            # 모델이 예측한 최적 행동 선택 (활용)
            with torch.no_grad():

                
                grid_tensor = torch.from_numpy(grid).float().to(self.device)
                hit_points_tensor = torch.from_numpy(hit_points).float().to(self.device)
                direction_tensor = torch.from_numpy(direction).float().to(self.device)
                forward_obj_tensor = torch.from_numpy(forward_obj).float().to(self.device)
                agent_position_tensor = torch.from_numpy(agent_position).float().to(self.device)
                
                # 배치 차원 추가
                grid_tensor = grid_tensor.unsqueeze(0) if grid_tensor.dim() == 3 else grid_tensor
                hit_points_tensor = hit_points_tensor.unsqueeze(0) if hit_points_tensor.dim() == 1 else hit_points_tensor
                direction_tensor = direction_tensor.unsqueeze(0) if direction_tensor.dim() == 1 else direction_tensor
                forward_obj_tensor = forward_obj_tensor.unsqueeze(0) if forward_obj_tensor.dim() == 1 else forward_obj_tensor
                agent_position_tensor = agent_position_tensor.unsqueeze(0) if agent_position_tensor.dim() == 1 else agent_position_tensor

                q_values = self.policy_net(
                    grid_tensor, 
                    hit_points_tensor, 
                    direction_tensor, 
                    forward_obj_tensor, 
                    agent_position_tensor
                )
                action = torch.argmax(q_values, dim=1).item()
    
        return action
    

    def sample(self, batch_size):
        # Override sample to include beta scheduling
        beta = min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)
        self.frame += 1
        batch, indices, weights = self.memory.sample(batch_size, beta)
        return batch, indices, weights

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return None
        
        transitions, indices, weights = self.sample(self.batch_size)
        
        # 배치 데이터 준비
        grid_batch = torch.FloatTensor(np.array([t.grid for t in transitions])).to(self.device)
        hit_points_batch = torch.FloatTensor(np.array([t.hit_points for t in transitions])).to(self.device)
        direction_batch = torch.FloatTensor(np.array([t.direction for t in transitions])).to(self.device)
        forward_obj_batch = torch.FloatTensor(np.array([t.forward_obj for t in transitions])).to(self.device)
        agent_position_batch = torch.FloatTensor(np.array([t.agent_position for t in transitions])).to(self.device)
        
        action_batch = torch.tensor([t.action for t in transitions], dtype=torch.long).to(self.device)
        reward_batch = torch.FloatTensor([t.reward for t in transitions]).to(self.device)
        
        next_grid_batch = torch.FloatTensor(np.array([t.next_grid for t in transitions])).to(self.device)
        next_hit_points_batch = torch.FloatTensor(np.array([t.next_hit_points for t in transitions])).to(self.device)
        next_direction_batch = torch.FloatTensor(np.array([t.next_direction for t in transitions])).to(self.device)
        next_forward_obj_batch = torch.FloatTensor(np.array([t.next_forward_obj for t in transitions])).to(self.device)
        next_agent_position_batch = torch.FloatTensor(np.array([t.next_agent_position for t in transitions])).to(self.device)
        
        done_batch = torch.FloatTensor([t.done for t in transitions]).to(self.device)

        # 현재 상태의 Q 값
        state_action_values = self.policy_net(
            grid_batch,
            hit_points_batch,
            direction_batch,
            forward_obj_batch,
            agent_position_batch
        ).gather(1, action_batch.unsqueeze(1))

        with torch.no_grad():
            # 다음 상태의 최적 행동 선택 (policy net)
            next_state_actions = self.policy_net(
                next_grid_batch,
                next_hit_points_batch,
                next_direction_batch,
                next_forward_obj_batch,
                next_agent_position_batch
            ).max(1)[1].unsqueeze(1)

            # 선택된 행동의 Q 값 계산 (target net)
            next_state_values = self.target_net(
                next_grid_batch,
                next_hit_points_batch,
                next_direction_batch,
                next_forward_obj_batch,
                next_agent_position_batch
            ).gather(1, next_state_actions)

            expected_state_action_values = reward_batch.unsqueeze(1) + \
                (1 - done_batch.unsqueeze(1)) * self.gamma * next_state_values

        # 손실 계산
        weights = weights.to(self.device)
        weights = weights.unsqueeze(1)
        loss = (F.smooth_l1_loss(state_action_values, expected_state_action_values, reduction='none') * weights).mean()

        # 최적화
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        # TD 에러 계산 및 우선순위 업데이트
        # td_errors = torch.abs(state_action_values - expected_state_action_values).detach().cpu().numpy()
        td_errors = torch.abs(state_action_values - expected_state_action_values).squeeze().detach().cpu().numpy()

        self.memory.update_priorities(indices, td_errors)

            # 타겟 네트워크 업데이트
        self.update_step_counter += 1
        if self.update_step_counter % self.target_update_steps == 0:
            self.update_target_network()
            print(f"Target network updated at step {self.update_step_counter}")

        return loss.item()

    def min_distance(self,grid,position,target=str):
            if target:
                bee_positions = np.argwhere(grid['grid'] == target)  # 벌의 위치 (N, 2) 배열
                # 에이전트의 위치를 numpy 배열로 변환
                agent_position = np.array(position)
                # 모든 벌과의 거리 계산
                distances = np.linalg.norm(bee_positions - agent_position, axis=1)
                # 최소 거리 찾기
                min_distance = np.min(distances)

                return min_distance
            else:
                return None

    
    def calculate_reward(self,state, next_state, done, step):
        total_reward = 0

        base_reward = -0.1

        # survival_reward = 0.01 * (step / 100)
        survival_reward=0

        previous_bees = np.sum(state['grid'] == 'B')
        current_bees = np.sum(next_state['grid'] == 'B')
        rescued_bees = previous_bees - current_bees

        bee_reward = rescued_bees * 100

        if 'B'==self.find_forward_obj(next_state):
            bee_reward+=5


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

    def encode_grid(self,grid):
        channels = {'B': 0, 'H': 1, 'K': 2,'W' : 3, 'A': 4}
        grid_encoded = np.zeros((len(channels), grid.shape[0], grid.shape[1]), dtype=np.float32)
        for symbol, idx in channels.items():
            if symbol in ['B', 'H', 'K','W']:
                grid_encoded[idx][grid == symbol] = 1.0

            elif symbol == 'A':
                # 에이전트의 위치를 표시
                agent_positions = np.argwhere(np.isin(grid, ['AU', 'AD', 'AL', 'AR']))
                for pos in agent_positions:
                    grid_encoded[idx][tuple(pos)] = 1.0

        return grid_encoded

    def encode_direction(self,direction):
        direction_map = {'UP': 0, 'DOWN': 1, 'LEFT': 2, 'RIGHT': 3}
        direction_vector = np.zeros(4)
        direction_vector[direction_map[direction]] = 1
        return direction_vector

    def normalize_hit_points(self,hit_points, max_hp=100):
        return np.array([hit_points / max_hp], dtype=np.float32)

    def extract_agent_info(self,grid):
        direction_symbols = {'AU': 'UP', 'AD': 'DOWN', 'AL': 'LEFT', 'AR': 'RIGHT'}
        positions = np.argwhere(np.isin(grid, list(direction_symbols.keys())))
        if len(positions) == 0:
            return (None, None)
        x, y = positions[0]
        symbol = grid[x, y]
        direction = direction_symbols.get(symbol, 'DOWN')
        return (x, y), direction
    

    def encode_forward_obj(self,forward_obj):
        forward_obj_map = {'W': 0, 'B': 1, 'K': 2, 'H': 3,'E':4}
        forward_obj_vector = np.zeros(5,dtype=np.float32)
        forward_obj_vector[forward_obj_map[forward_obj]] = 1
        return forward_obj_vector

    def preprocess_state(self,grid, hit_points, direction, agent_position, forward_obj):
        grid_encoded = self.encode_grid(grid)
        hit_points_normalized = self.normalize_hit_points(hit_points)
        direction_encoded = self.encode_direction(direction)
        forward_obj_encoded  = self.encode_forward_obj(forward_obj)
        
        # 에이전트의 위치 정규화
        if agent_position is not None:
            agent_x, agent_y = agent_position
            agent_position_normalized = np.array([agent_x / grid.shape[0], agent_y / grid.shape[1]], dtype=np.float32)
        else:
            agent_position_normalized = np.array([0.0, 0.0], dtype=np.float32)
        
        return grid_encoded, hit_points_normalized, direction_encoded, forward_obj_encoded, agent_position_normalized

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
            'UP': [(-2, 0), (-1, -1), (-1, 1)],
            'DOWN': [(2, 0), (1, -1), (1, 1)],
            'RIGHT': [(0, 2), (-1, 1), (1, 1)],
            'LEFT': [(0, -2), (-1, -1), (1, -1)]
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

    def save_checkpoint(self, episode, loss, reward, filepath):
        checkpoint = {
        'episode': episode,
        'model_state_dict': self.policy_net.state_dict(),
        'target_state_dict': self.target_net.state_dict(),
        'optimizer_state_dict': self.optimizer.state_dict(),
        'scheduler_state_dict': self.scheduler.state_dict(),
        'steps_done': self.steps_done,
        'loss': loss,
        'reward': reward
    }
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved at episode {episode} to {filepath}")

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.policy_net.load_state_dict(checkpoint['model_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.steps_done = checkpoint['steps_done']
        
        print(f"Loaded checkpoint from {checkpoint_path}")
