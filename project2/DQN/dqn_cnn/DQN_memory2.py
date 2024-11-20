from collections import namedtuple
import numpy as np
import torch
import random

from sumTree import SumTree

# Transition namedtuple 정의
Transition = namedtuple('Transition', 
                        ('grid', 'hit_points', 'direction', 'forward_obj', 'agent_position',
                         'action', 'reward', 
                         'next_grid', 'next_hit_points', 'next_direction', 'next_forward_obj', 'next_agent_position',
                         'done'))

# PrioritizedReplayMemory 클래스
class PrioritizedReplayMemory:
    def __init__(self, capacity, priority_alpha=0.6, priority_beta=0.4, epsilon=1e-6):
        self.capacity = capacity
        self.tree = SumTree(capacity)
        self.priority_alpha = priority_alpha
        self.priority_beta = priority_beta
        self.epsilon = epsilon  # 우선순위의 작은 값 추가하여 0 방지
        self.max_priority = 1.0  # 초기 최대 우선순위

    def push(self, grid, hit_points, direction, forward_obj, agent_position,
             action, reward,
             next_grid, next_hit_points, next_direction, next_forward_obj, next_agent_position,
             done):
        """새로운 transition 저장"""
        transition = Transition(
            grid, hit_points, direction, forward_obj, agent_position,
            action, reward,
            next_grid, next_hit_points, next_direction, next_forward_obj, next_agent_position,
            done
        )
        priority = self.max_priority ** self.priority_alpha
        self.tree.add(priority, transition)

    def sample(self, batch_size, beta=None):
        """우선순위에 기반한 샘플링"""
        if beta is None:
            beta = self.priority_beta_initial

        batch = []
        indices = []
        priorities = []
        segment = self.tree.total_priority / batch_size

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(s)
            batch.append(data)
            indices.append(idx)
            priorities.append(p)

        probs = np.array(priorities) / self.tree.total_priority
        weights = (self.tree.n_entries * probs) ** (-beta)
        weights /= weights.max()
        weights = torch.tensor(weights, dtype=torch.float32)

        return batch, indices, weights

    def update_priorities(self, indices, td_errors):
        """TD 에러를 기반으로 우선순위 업데이트"""
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + self.epsilon) ** self.priority_alpha
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)

    def __len__(self):
        return self.tree.n_entries