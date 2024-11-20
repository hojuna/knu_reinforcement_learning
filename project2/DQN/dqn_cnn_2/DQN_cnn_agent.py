import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque, namedtuple
import random
from knu_rl_env.grid_survivor import GridSurvivorAgent
import torch.nn.functional as F


# 추가 필요한 import
from DQN_memory import ReplayMemory
from DQN_network import DQN



class DQNAgent(GridSurvivorAgent):
    def __init__(self, input_size=35, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # 네트워크 및 타겟 네트워크 초기화
        self.policy_net = DQN(input_channels=5, num_actions=3).to(self.device)
        self.target_net = DQN(input_channels=5, num_actions=3).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # 하이퍼파라미터 설정
        self.gamma = 0.99
        self.epsilon_start = 1.0
        self.epsilon_end = 0.01
        self.epsilon_decay = 100000
        self.batch_size = 128
        self.target_update = 1000
        self.learning_rate = 0.0001
        
        # 메모리 초기화
        self.memory = ReplayMemory(capacity=100000)
        
        # 옵티마이저 설정
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        
        # 학습 관련 변수들
        self.steps_done = 0
        self.episode_rewards = []
        self.training = True
        
        # 상태 전처리를 위한 변수들
        self.input_size = input_size
        self.previous_bee_count = None
        self.previous_hp = None

    def preprocess_state(self, state):
        """상태 전처리"""
        grid = state['grid']
        hit_points = state['hit_points']
        
        grid_size = grid.shape
        processed_grid = np.zeros((5, grid_size[0], grid_size[1]), dtype=np.float32)
        
        # 채널별 처리
        processed_grid[0] = (grid == 'W').astype(np.float32)
        processed_grid[1] = (grid == 'E').astype(np.float32)
        processed_grid[2] = np.isin(grid, ['AL', 'AR', 'AU', 'AD']).astype(np.float32)
        processed_grid[3] = (grid == 'B').astype(np.float32)
        processed_grid[4] = (grid == 'H').astype(np.float32)
        
        # 배치 차원 추가 (6, 34, 34) -> (1, 6, 34, 34)
        processed_grid = np.expand_dims(processed_grid, axis=0)
        
        return {
            'grid': processed_grid,
            'hit_points': hit_points
        }

    def calculate_reward(self, state, next_state, done):
        """보상 계산"""
        reward = 0
        
        # 현재와 다음 상태의 꿀벌 수 계산
        current_bees = np.sum(state['grid'] == 'B')
        next_bees = np.sum(next_state['grid'] == 'B')
        
        # 꿀벌 구출 보상
        if next_bees < current_bees:
            reward += 100
        
        # 체력 관련 보상
        current_hp = state['hit_points']
        next_hp = next_state['hit_points']
        if next_hp < current_hp:
            reward -= 20
        
        # 살인벌 접촉 패널티
        if next_hp == 0 and not done:
            reward -= 200
            
        # 시간 패널티
        reward -= 0.1
        
        # 에피소드 종료 보상
        if done:
            if next_bees == 0:  # 모든 꿀벌 구출 성공
                reward += 500
            else:  # 실패
                reward -= 100
                
        return reward

    def select_action(self, state):
        """입실론-그리디 정책에 따른 행동 선택"""
        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                 np.exp(-1. * self.steps_done / self.epsilon_decay)
        self.steps_done += 1
        
        if not self.training:
            epsilon = 0.01
            
        if random.random() > epsilon:
            with torch.no_grad():
                processed_state = self.preprocess_state(state)
                q_values = self.policy_net(processed_state['grid'], 
                                         processed_state['hit_points'])
                return q_values.max(1)[1].item()
        else:
            return random.randrange(3)

    def act(self, state):
        """에이전트의 행동 결정"""
        action = self.select_action(state)
        return action

    def optimize_model(self):
        if not self.memory.can_sample(self.batch_size):
            return 0.0

        transitions, indices, weights = self.memory.sample(self.batch_size)
        if transitions is None:
            return 0.0

        # 배치 데이터 준비
        batch = self.memory.Transition(*zip(*transitions))
        
        # 상태 전처리
        processed_states = [self.preprocess_state({'grid': grid, 'hit_points': hp}) 
                           for grid, hp in zip(batch.grid, batch.hit_points)]
        processed_next_states = [self.preprocess_state({'grid': grid, 'hit_points': hp}) 
                               if grid is not None else None 
                               for grid, hp in zip(batch.next_grid, batch.next_hit_points)]
        
        # 료 상태 마스크
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
            processed_next_states)), device=self.device, dtype=torch.bool)
        
        # 현재 상태 텐서 준비 (수정된 부분)
        state_batch = torch.tensor(np.array([s['grid'][0] for s in processed_states]), 
                                 device=self.device, dtype=torch.float32)  # [0] 추가하여 첫 번째 차원 제거
        hp_batch = torch.tensor(np.array([s['hit_points'] for s in processed_states]), 
                                device=self.device, dtype=torch.float32).unsqueeze(1)
        
        # 다음 상태 텐서 준비 (수정된 부분)
        non_final_next_states = torch.tensor(
            np.array([s['grid'][0] for s in processed_next_states if s is not None]),  # [0] 추가
            device=self.device, dtype=torch.float32)
        
        # 다음 상태의 체력값 준비 수정
        non_final_next_hp = torch.zeros(self.batch_size, device=self.device)
        if non_final_mask.any():
            non_final_next_hp[non_final_mask] = torch.tensor(
                [s['hit_points'] for s in processed_next_states if s is not None],
                device=self.device, dtype=torch.float32
            )
        non_final_next_hp = non_final_next_hp.unsqueeze(1)
        
        action = torch.tensor(batch.action, device=self.device)
        reward = torch.tensor(batch.reward, device=self.device)
        
        # 현재 Q 값 계산
        state_action_values = self.policy_net(state_batch, hp_batch).gather(1, action.unsqueeze(1))
        
        # 다음 상태의 V 값 계산
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        if non_final_mask.any():
            with torch.no_grad():
                next_state_values[non_final_mask] = self.target_net(
                    non_final_next_states, 
                    non_final_next_hp
                ).max(1)[0]
        
        # 기대 Q 값 계산
        expected_state_action_values = (next_state_values * self.gamma) + reward
        
        # Huber 손실 계산
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        
        # 옵티마이저 스텝
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 100)
        self.optimizer.step()
        
        # TD 에러 계산 및 우선순위 업데이트
        with torch.no_grad():
            td_errors = (expected_state_action_values - state_action_values.squeeze()).abs().cpu().numpy()
        self.memory.update_priorities(indices, td_errors)
        
        return loss.item()

    def update_target_network(self):
        """타겟 네트워크 업데이트"""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save_transition(self, state, action, next_state, reward, done):
        """경험 저장"""
        # 현재 상태 정보 추출
        grid = state['grid']
        hit_points = state['hit_points']
        
        # grid에서 에이전트 방향 추출
        agent_pos = np.where(np.isin(grid, ['AL', 'AR', 'AU', 'AD']))
        if len(agent_pos[0]) > 0:
            direction = grid[agent_pos][0]  # AL, AR, AU, AD 중 하나
        else:
            direction = None
        
        forward_obj = state.get('forward_obj', None)
        agent_position = (agent_pos[0][0], agent_pos[1][0]) if len(agent_pos[0]) > 0 else None
        
        # 다음 상태 정보 추출
        if next_state is not None:
            next_grid = next_state['grid']
            next_hit_points = next_state['hit_points']
            
            # 다음 상태의 에이전트 방향 추출
            next_agent_pos = np.where(np.isin(next_grid, ['AL', 'AR', 'AU', 'AD']))
            if len(next_agent_pos[0]) > 0:
                next_direction = next_grid[next_agent_pos][0]
                next_agent_position = (next_agent_pos[0][0], next_agent_pos[1][0])
            else:
                next_direction = None
                next_agent_position = None
            
            next_forward_obj = next_state.get('forward_obj', None)
        else:
            next_grid = None
            next_hit_points = 0
            next_direction = None
            next_forward_obj = None
            next_agent_position = None

        # ReplayMemory에 트랜지션 저장
        self.memory.push(
            grid=grid,
            hit_points=hit_points,
            direction=direction,
            forward_obj=forward_obj,
            agent_position=agent_position,
            action=action,
            reward=reward,
            next_grid=next_grid,
            next_hit_points=next_hit_points,
            next_direction=next_direction,
            next_forward_obj=next_forward_obj,
            next_agent_position=next_agent_position,
            done=done
        )
