import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import math
from collections import namedtuple
import random

# 2. Prioritized Replay Memory 클래스 정의
class SumTree:
    """우선순위 합계를 위한 SumTree 구조"""
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)  # 트리 노드
        self.data = np.zeros(capacity, dtype=object)  # 실제 데이터 저장
        self.write = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0  # 덮어쓰기

    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return idx, self.tree[idx], self.data[dataIdx]

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class PrioritizedReplayMemory:
    """Prioritized Experience Replay 메모리"""
    def __init__(self, capacity, alpha=0.6):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.epsilon = 1e-5  # 작은 값 추가하여 우선순위가 0이 되지 않도록 함
        self.Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

    def push(self, *args):
        max_priority = np.max(self.tree.tree[-self.tree.capacity:])
        if max_priority == 0:
            max_priority = 1.0

        self.tree.add(max_priority, Transition(*args))

    def sample(self, batch_size, beta=0.4):
        batch = []
        idxs = []
        segment = self.tree.total() / batch_size
        priorities = []

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            idx, p, data = self.tree.get(s)
            batch.append(data)
            idxs.append(idx)
            priorities.append(p)

        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.capacity * sampling_probabilities, -beta)
        is_weight /= is_weight.max()

        return batch, idxs, is_weight

    def update(self, idx, error):
        p = (np.abs(error) + self.epsilon) ** self.alpha
        self.tree.update(idx, p)

    def __len__(self):
        return len(self.tree.data)