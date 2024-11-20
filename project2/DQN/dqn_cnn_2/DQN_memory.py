import numpy as np
import torch
from collections import namedtuple
import random
import torch.nn.functional as F

class ReplayMemory:
    def __init__(self, capacity, priority_alpha=0.6, priority_beta=0.4):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.rare_buffer = []
        
        # Transition 네임드튜플 정의
        self.Transition = namedtuple('Transition', 
            ('grid', 'hit_points', 'direction', 'forward_obj', 'agent_position',
             'action', 'reward', 
             'next_grid', 'next_hit_points', 'next_direction', 'next_forward_obj', 'next_agent_position',
             'done'))
        
        self.priority_alpha = priority_alpha
        self.priority_beta = priority_beta
        self.max_priority = 1.0
        self.beta_increment = 0.001
        self.eps = 1e-6

    def push(self, *, grid, hit_points, direction, forward_obj, agent_position,
             action, reward, next_grid, next_hit_points, next_direction, 
             next_forward_obj, next_agent_position, done):
        """새로운 transition 저장"""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        
        transition = self.Transition(
            grid, hit_points, direction, forward_obj, agent_position,
            action, reward,
            next_grid, next_hit_points, next_direction, next_forward_obj, next_agent_position,
            done
        )
        
        if abs(reward) > 1.0:
            self.rare_buffer.append(transition)
            if len(self.rare_buffer) > 1000:
                self.rare_buffer.pop(0)
        
        self.memory[self.position] = transition
        self.priorities[self.position] = self.max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """배치 샘플링"""
        if len(self.memory) == 0:
            return None, None, None

        # 우선순위 기반 샘플링
        priorities = self.priorities[:len(self.memory)]
        probs = priorities ** self.priority_alpha
        probs /= probs.sum()

        # 일반 샘플링
        indices = np.random.choice(len(self.memory), batch_size - 2, p=probs)
        
        # 희귀 경험 추가
        if self.rare_buffer:
            rare_samples = random.sample(self.rare_buffer, min(2, len(self.rare_buffer)))
            regular_samples = [self.memory[idx] for idx in indices]
            samples = regular_samples + rare_samples
            
            while len(samples) < batch_size:
                samples.append(random.choice(regular_samples))
        else:
            samples = [self.memory[idx] for idx in indices]
            indices = np.concatenate([indices, np.random.choice(len(self.memory), batch_size - len(indices))])

        # IS 가중치 계산
        total_size = len(self.memory)
        weights = (total_size * probs[indices]) ** (-self.priority_beta)
        weights /= weights.max()
        weights = torch.FloatTensor(weights)
        
        self.priority_beta = min(1.0, self.priority_beta + self.beta_increment)

        return samples, indices, weights

    def update_priorities(self, indices, td_errors):
        """TD 에러를 기반으로 우선순위 업데이트"""
        for idx, td_error in zip(indices, td_errors):
            if idx < len(self.priorities):
                self.priorities[idx] = (abs(td_error) + self.eps) ** self.priority_alpha
                self.max_priority = max(self.max_priority, self.priorities[idx])

    def get_memory_stats(self):
        """메모리 상태 통계"""
        if len(self.memory) == 0:
            return None
            
        rewards = np.array([t.reward for t in self.memory])
        return {
            'size': len(self.memory),
            'rare_size': len(self.rare_buffer),
            'reward_mean': rewards.mean(),
            'reward_std': rewards.std(),
            'max_priority': self.max_priority,
            'beta': self.priority_beta
        }

    def can_sample(self, batch_size):
        """샘플링 가능 여부 확인"""
        return len(self.memory) >= batch_size

    def __len__(self):
        return len(self.memory)

    def clear(self):
        """메모리 초기화"""
        self.memory.clear()
        self.rare_buffer.clear()
        self.priorities = np.zeros((self.capacity,), dtype=np.float32)
        self.position = 0
        self.max_priority = 1.0