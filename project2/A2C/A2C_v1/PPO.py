import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from collections import deque

class EnhancedA2CNetwork(nn.Module):
    def __init__(self, input_size, hidden_size=128):
        super().__init__()
        
        # 컨볼루션 레이어로 grid 정보 처리
        self.conv = nn.Sequential(
            nn.Conv2d(9, 16, kernel_size=3, padding=1),  # 9는 cell type의 수
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # 추가 정보를 위한 FC 레이어
        self.extra_fc = nn.Sequential(
            nn.Linear(2, hidden_size//4),  # HP와 살인벌까지의 거리
            nn.ReLU()
        )
        
        conv_out_size = self._get_conv_out_size(input_size)
        
        # 공통 특징 추출층
        self.shared = nn.Sequential(
            nn.Linear(conv_out_size + hidden_size//4, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # 정책 헤드
        self.policy = nn.Sequential(
            nn.Linear(hidden_size, 3),
            nn.Softmax(dim=-1)
        )
        
        # 가치 헤드
        self.value = nn.Linear(hidden_size, 1)
        
    def _get_conv_out_size(self, input_size):
        # Grid 크기에 따른 conv 출력 크기 계산
        grid_size = int(np.sqrt(input_size / 9))  # 9는 cell type의 수
        x = torch.zeros(1, 9, grid_size, grid_size)
        x = self.conv(x)
        return x.size(1)
        
    def forward(self, grid, extra_info):
        # Grid 정보 처리
        conv_out = self.conv(grid)
        
        # 추가 정보 처리
        extra_out = self.extra_fc(extra_info)
        
        # 특징 결합
        combined = torch.cat([conv_out, extra_out], dim=1)
        shared_features = self.shared(combined)
        
        # 정책과 가치 출력
        action_probs = self.policy(shared_features)
        state_value = self.value(shared_features)
        
        return action_probs, state_value

class EnhancedA2CAgent(GridSurvivorAgent):
    def __init__(self, grid_size):
        super().__init__()
        self.grid_size = grid_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 하이퍼파라미터
        self.gamma = 0.99
        self.learning_rate = 0.001
        self.entropy_coef = 0.01
        self.value_coef = 0.5
        
        # 네트워크와 옵티마이저
        input_size = grid_size * grid_size * 9  # 9는 cell type의 수
        self.network = EnhancedA2CNetwork(input_size).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.learning_rate)
        
        # 경험 버퍼
        self.trajectory = []
        
        # Safety 체크를 위한 이전 상태 저장
        self.prev_states = deque(maxlen=3)
        
    def preprocess_state(self, state):
        """상태 전처리 - Grid를 3D 텐서로, 추가 정보는 별도 처리"""
        grid = state['grid']
        cell_types = {'E':0, 'W':1, 'B':2, 'H':3, 'K':4, 
                     'AL':5, 'AR':6, 'AU':7, 'AD':8}
        
        # Grid를 3D 텐서로 변환 (one-hot)
        grid_tensor = torch.zeros(9, self.grid_size, self.grid_size)
        
        # 살인벌 위치 추적
        killer_bee_pos = None
        player_pos = None
        
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                cell = grid[i][j]
                cell_idx = cell_types.get(cell, 0)
                grid_tensor[cell_idx, i, j] = 1
                
                if cell == 'K':
                    killer_bee_pos = (i, j)
                elif cell in ['AL', 'AR', 'AU', 'AD']:
                    player_pos = (i, j)
        
        # 살인벌까지의 거리 계산
        distance_to_killer = 1.0
        if killer_bee_pos and player_pos:
            distance_to_killer = np.sqrt(
                (killer_bee_pos[0] - player_pos[0])**2 + 
                (killer_bee_pos[1] - player_pos[1])**2
            ) / np.sqrt(2 * self.grid_size**2)  # 정규화
        
        # 추가 정보: [체력, 살인벌까지의 거리]
        extra_info = torch.tensor(
            [state['hit_points'] / 100.0, distance_to_killer],
            dtype=torch.float32
        )
        
        return (
            grid_tensor.unsqueeze(0).to(self.device),
            extra_info.unsqueeze(0).to(self.device)
        )
    
    def is_safe_action(self, state, action):
        """안전한 행동인지 확인"""
        # 현재 상태에서 action을 취했을 때 위험한지 예측
        grid = state['grid']
        player_pos = None
        killer_pos = None
        
        # 현재 위치와 방향 찾기
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if grid[i][j] in ['AL', 'AR', 'AU', 'AD']:
                    player_pos = (i, j)
                    direction = grid[i][j][1]
                elif grid[i][j] == 'K':
                    killer_pos = (i, j)
        
        if not player_pos or not killer_pos:
            return True
        
        # 이전 상태들로부터 살인벌의 이동 패턴 예측
        if len(self.prev_states) >= 3:
            # 이동 패턴 분석 로직
            pass
        
        # 기본적인 안전 체크
        if action == self.ACTION_FORWARD:
            next_pos = self._get_next_position(player_pos, direction)
            if self._is_near_killer(next_pos, killer_pos):
                return False
        
        return True
    
    def _get_next_position(self, pos, direction):
        """다음 위치 계산"""
        x, y = pos
        if direction == 'U': return (x-1, y)
        elif direction == 'D': return (x+1, y)
        elif direction == 'L': return (x, y-1)
        elif direction == 'R': return (x, y+1)
        return pos
    
    def _is_near_killer(self, pos, killer_pos):
        """살인벌 근처인지 확인"""
        distance = np.sqrt(
            (pos[0] - killer_pos[0])**2 + 
            (pos[1] - killer_pos[1])**2
        )
        return distance <= 2
    
    def act(self, state):
        """행동 선택"""
        # 상태 전처리
        grid_tensor, extra_info = self.preprocess_state(state)
        
        with torch.no_grad():
            action_probs, value = self.network(grid_tensor, extra_info)
        
        # 행동 선택
        dist = Categorical(action_probs)
        action = dist.sample()
        
        # 안전 체크
        if not self.is_safe_action(state, action.item()):
            # 다른 안전한 행동 선택
            safe_actions = [a for a in range(3) 
                          if self.is_safe_action(state, a)]
            if safe_actions:
                action = torch.tensor(np.random.choice(safe_actions))
        
        # 경험 저장
        self.trajectory.append({
            'grid': grid_tensor,
            'extra_info': extra_info,
            'action': action,
            'value': value
        })
        
        # 이전 상태 저장
        self.prev_states.append(state)
        
        return action.item()
    
    def update(self, reward, done):
        """경험 업데이트 및 학습"""
        if len(self.trajectory) == 0:
            return
        
        self.trajectory[-1]['reward'] = reward
        self.trajectory[-1]['done'] = done
        
        if done:
            self.learn()
    
    def learn(self):
        """A2C 학습"""
        # 리턴 계산
        returns = []
        R = 0
        for t in reversed(self.trajectory):
            R = t['reward'] + self.gamma * R * (1 - t['done'])
            returns.insert(0, R)
        
        returns = torch.tensor(returns).to(self.device)
        
        # 배치 데이터 준비
        grids = torch.cat([t['grid'] for t in self.trajectory])
        extra_infos = torch.cat([t['extra_info'] for t in self.trajectory])
        actions = torch.stack([t['action'] for t in self.trajectory])
        
        # 네트워크 통과
        action_probs, values = self.network(grids, extra_infos)
        
        # Advantage 계산
        advantages = returns - values.squeeze()
        
        # 손실 함수 계산
        dist = Categorical(action_probs)
        policy_loss = -(dist.log_prob(actions) * advantages.detach()).mean()
        value_loss = F.mse_loss(values.squeeze(), returns)
        entropy_loss = -self.entropy_coef * dist.entropy().mean()
        
        # 전체 손실
        total_loss = policy_loss + self.value_coef * value_loss + entropy_loss
        
        # 역전파 및 최적화
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        # 버퍼 초기화
        self.trajectory.clear()
    
    def save(self, path):
        """모델 저장"""
        torch.save({
            'network_state': self.network.state_dict(),
            'optimizer_state': self.optimizer.state_dict()
        }, path)
    
    def load(self, path):
        """모델 로드"""
        checkpoint = torch.load(path)
        self.network.load_state_dict(checkpoint['network_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])