import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Optional, Tuple
import random
from collections import deque
import torch.nn.functional as F
import math

from knu_rl_env.grid_survivor import GridSurvivorAgent
from DQN_network import DuelingDQN
from DQN_memory import PrioritizedReplayMemory, EpisodeBuffer

class GridSurvivorRLAgent(GridSurvivorAgent):
    def __init__(self, input_size: int = 35, device: str = 'cuda'):
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.input_size = input_size
        self.n_actions = 3
        
        # Networks
        self.policy_net = DuelingDQN(input_size).to(self.device)
        self.target_net = DuelingDQN(input_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.policy_net.parameters(), 
            lr=0.0001,
            weight_decay=0.01,
            amsgrad=True
        )
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, 
            step_size=1000,  # 1000 스텝마다 학습률 조정
            gamma=0.95       # 학습률을 95%로 감소
        )
        
        # Memory
        self.memory = PrioritizedReplayMemory(100000, alpha=0.6, beta=0.4)
        self.episode_buffer = EpisodeBuffer(1000)
        
        # Parameters
        self.batch_size = 128

        self.gamma = 0.99
        self.epsilon_start = 1.0
        self.epsilon_end = 0.01
        self.epsilon_decay = 5000  # 에피소드 기준으로 감소
        self.target_update = 500
        self.steps_done = 0
        
        # Training mode
        self.training = True
        
        # Experience replay parameters
        self.n_step = 3
        self.n_step_buffer = deque(maxlen=self.n_step)
        
        # Initialize action statistics
        self.action_stats = {i: 0 for i in range(self.n_actions)}
        
        # Additional tracking metrics
        self.episode_rewards = []
        self.training_stats = {
            'avg_q_values': [],
            'max_q_values': [],
            'loss_values': [],
            'gradient_norms': []
        }
        
        self.epsilon_current = self.epsilon_start  # 현재 epsilon 값을 저장할 변수 추가
        
        # 평가 지표 초기화
        self.best_reward = float('-inf')
        self.eval_scores = []
        self.eval_interval = 100  # 100 에피소드마다 평가
        self.episode = 0  # 스텝 대신 에피소드로 변경
    
    def process_state(self, state):
        grid = state['grid']
        hit_points = state['hit_points']
        
        # 모든 채널을 하나의 numpy 배열로 만들기
        channels = np.zeros((6, grid.shape[0], grid.shape[1]), dtype=np.float32)
        
        # Channel 1: Empty spaces (E)
        channels[0] = (grid == 'E')
        
        # Channel 2: Walls (W)
        channels[1] = (grid == 'W')
        
        # Channel 3: Player (모든 방향)
        channels[2] = ((grid == 'AL') | (grid == 'AR') | (grid == 'AU') | (grid == 'AD'))
        
        # Channel 4: Bees (B)
        channels[3] = (grid == 'B')
        
        # Channel 5: Dangerous entities (H, K)
        channels[4] = (grid == 'H') 
        
        channels[5] = (grid == 'K')
        
        # 한 번에 텐서로 변환
        state_tensor = torch.from_numpy(channels).unsqueeze(0).to(self.device)
        hit_points_tensor = torch.FloatTensor([hit_points/100.0]).to(self.device)
        
        return state_tensor, hit_points_tensor
    
    def calculate_reward(self, state, next_state, done):
        reward = 0.0
        
        # 기본 행동 페널티
        reward -= 0.1
        
        # 꿀벌 구출 보상
        current_bees = np.sum(state['grid'] == 'B')
        next_bees = np.sum(next_state['grid'] == 'B')
        bees_saved = current_bees - next_bees
        if bees_saved > 0:
            reward += 15.0 * bees_saved
        
        # 체력 관련 보상
        hp_loss = state['hit_points'] - next_state['hit_points']
        if hp_loss > 0:
            reward -= hp_loss * 0.5
        
        # 생존 보상
        if not done:
            reward += 0.1
        
        # 종료 상태 보상
        if done:
            if next_state['hit_points'] <= 0:
                reward -= 50.0  # 죽음 페널티
            elif next_bees == 0:
                reward += 100.0  # 모든 꿀벌 구출 보상
                if next_state['hit_points'] > 50:
                    reward += 50.0  # 추가 체력 보너스
        
        return reward
    
    def act(self, state):
        if not self.training:
            return self.get_action(state)

        
        self.epsilon_current = self.update_epsilon()
        # epsilon-greedy 전략
        if random.random() > self.epsilon_current:
            action = self.get_action(state)
        else:
            # 유효한 액션 중에서만 랜덤 
            action = random.randint(0, 3 - 1)
        
        return action
    
    def get_action(self, state):
        with torch.no_grad():
            state_tensor, hit_points = self.process_state(state)
            q_values = self.policy_net(state_tensor, hit_points)
            
            # action 2를 제외하고 가장 높은 Q-value를 가진 액션 선택
            q_values_np = q_values.cpu().numpy()[0]
            action = np.argmax(q_values_np)
            
            return action
    
    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return None

        # PER 샘플링
        transitions, indices, weights = self.memory.sample(self.batch_size)
        weights = torch.FloatTensor(weights).to(self.device)
        
        # 배치 데이터 준비
        state_batch_tensors = [self.process_state(t.state) for t in transitions]
        next_state_batch_tensors = [self.process_state(t.next_state) for t in transitions]
        
        # 상태와 히트포인트 분리
        state_batch = torch.cat([s[0] for s in state_batch_tensors])
        hit_points_batch = torch.cat([s[1] for s in state_batch_tensors])
        next_state_batch = torch.cat([s[0] for s in next_state_batch_tensors])
        next_hit_points_batch = torch.cat([s[1] for s in next_state_batch_tensors])
        
        # 기타 배치 데이터
        action_batch = torch.tensor([t.action for t in transitions], 
                                  device=self.device, dtype=torch.long)
        reward_batch = torch.tensor([t.reward for t in transitions], 
                                  device=self.device, dtype=torch.float)
        done_batch = torch.tensor([t.done for t in transitions], 
                                device=self.device, dtype=torch.float)

        # Double DQN
        with torch.no_grad():
            # 다음 상태에서의 행동 선택 (정책 네트워크 사용)
            next_action_values = self.policy_net(next_state_batch, next_hit_points_batch)
            # action 2를 제외하고 선택
            next_action_values[:, 2] = float('-inf')
            next_actions = next_action_values.max(1)[1].unsqueeze(1)
            
            # 선택된 행동의 가치 평가 (타겟 네트워크 사용)
            next_state_values = self.target_net(next_state_batch, next_hit_points_batch)
            next_state_values = next_state_values.gather(1, next_actions)
            
            expected_state_action_values = reward_batch.unsqueeze(1) + \
                                         (1 - done_batch.unsqueeze(1)) * self.gamma * next_state_values

        # 현재 상태의 행동 가치
        state_action_values = self.policy_net(state_batch, hit_points_batch).gather(1, action_batch.unsqueeze(1))

        # Huber 손실 계산
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values, reduction='none')
        weighted_loss = (loss * weights.unsqueeze(1)).mean()

        # 옵마이저 스텝
        self.optimizer.zero_grad()
        weighted_loss.backward()
        
        # 그래디언트 클리핑
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        
        self.optimizer.step()

        # PER 우선순위 업데이트
        td_errors = torch.abs(state_action_values - expected_state_action_values).detach().cpu().numpy()
        self.memory.update_priorities(indices, td_errors)

        return weighted_loss.item()
    
    def update_target_network(self):
        """Updates the target network parameters by copying from the policy network"""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def evaluate(self, env):
        """
        현재 정책 네트워크의 성능을 평가합니다.
        """
        self.training = False  # 평가 모드로 설정
        total_reward = 0
        eval_episodes = 5  # 평가할 에피소드 수
        
        for _ in range(eval_episodes):
            state, _ = env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                action = self.get_action(state)  # get_action 메소드 사용
                next_state, reward, done, _ ,_= env.step(action)
                episode_reward += reward
                state = next_state
                
            total_reward += episode_reward
        
        self.training = True  # 다시 학습 모드로 설정
        avg_reward = total_reward / eval_episodes
        self.eval_scores.append(avg_reward)
        
        # 최고 성능 갱신 시 모델 저장
        if avg_reward > self.best_reward:
            self.best_reward = avg_reward
            self.save_model('best_model.pth')
        
        return avg_reward
    
    def select_action(self, state, eval=False):
        """
        행동을 선택합니다.
        eval=True일 경우 탐험을 하지 않습니다.
        """
        if not eval and random.random() < self.epsilon_current:
            return random.randrange(self.n_actions)
        
        with torch.no_grad():
            state_tensor, hit_points = self.process_state(state)
            q_values = self.policy_net(state_tensor, hit_points)
            return q_values.max(1)[1].item()
    
    def save_model(self, filename):
        """
        모델을 저장합니다.
        """
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_reward': self.best_reward,
            'eval_scores': self.eval_scores
        }, filename)
    
    def update_epsilon(self):
        """에피소드 별로 엡실론 업데이트"""
        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * math.exp(-1. * self.steps_done / self.epsilon_decay)
        self.steps_done += 1

        return epsilon