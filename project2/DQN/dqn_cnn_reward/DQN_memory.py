from collections import deque, namedtuple
import numpy as np
import torch

import random




# ReplayMemory 클래스
class ReplayMemory:
    def __init__(self, capacity, priority_alpha=0.6, priority_beta=0.4):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        
        # state와 next_state는 각각 grid만 포함
        self.Transition = namedtuple('Transition', 
            ('grid', 'hit_points', 'direction', 'forward_obj', 'agent_position',
             'action', 'reward', 
             'next_grid', 'next_hit_points', 'next_direction', 'next_forward_obj', 'next_agent_position',
             'done'))
        
        self.priority_alpha = priority_alpha
        self.priority_beta = priority_beta
        self.max_priority = 1.0

    def push(self, grid, hit_points, direction, forward_obj, agent_position,
             action, reward,
             next_grid, next_hit_points, next_direction, next_forward_obj, next_agent_position,
             done):
        """새로운 transition 저장"""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        
        # 모든 데이터를 분리해서 저장
        self.memory[self.position] = self.Transition(
            grid, hit_points, direction, forward_obj, agent_position,
            action, reward,
            next_grid, next_hit_points, next_direction, next_forward_obj, next_agent_position,
            done
        )
        
        self.priorities[self.position] = self.max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        if self.can_sample(batch_size):
            probs = self.priorities[:len(self.memory)]
            probs = probs ** self.priority_alpha
            probs = probs / probs.sum()

            indices = np.random.choice(len(self.memory), batch_size, p=probs)
            weights = (len(self.memory) * probs[indices]) ** (-self.priority_beta)
            weights = weights / weights.max()
            weights = torch.tensor(weights, dtype=torch.float)

            transitions = [self.memory[idx] for idx in indices]
            return transitions, indices, weights
        return None

    def update_priorities(self, indices, td_errors):
        """TD 에러를 기반으로 우선순위 업데이트"""
        for idx, td_error in zip(indices, td_errors):
            self.priorities[idx] = (abs(td_error) + 1e-6)  # 작은 값 추가로 0 방지
            self.max_priority = max(self.max_priority, self.priorities[idx])

    def clear_old_transitions(self, threshold=1000):
        """오래된 transition 제거 (메모리 관리)"""
        if len(self.memory) > threshold:
            # 가장 최근 threshold개만 유지
            self.memory = self.memory[-threshold:]
            self.priorities = self.priorities[-threshold:]
            self.position = len(self.memory) % self.capacity

    def __len__(self):
        return len(self.memory)

    def can_sample(self, batch_size):
        return len(self.memory) >= batch_size
    

    def __getitem__(self, idx):
        return self.memory[idx]