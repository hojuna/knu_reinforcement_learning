import numpy as np
import random
from collections import namedtuple

# SumTree 클래스 (수정 없음)
class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        self.n_entries = 0  # 실제로 저장된 항목의 수

    def add(self, p, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0  # 덮어쓰기
        if self.n_entries < self.capacity:
            self.n_entries += 1

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


    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1

        return idx, self.tree[idx], self.data[data_idx]


SequenceTransition = namedtuple('SequenceTransition', ('states', 'actions', 'rewards', 'next_states', 'dones'))

class PrioritizedReplayMemory:
    """시퀀스 데이터를 위한 Prioritized Experience Replay 메모리"""
    def __init__(self, capacity, alpha=0.6):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.epsilon = 1e-5  # 작은 값 추가하여 우선순위가 0이 되지 않도록 함

    def push(self, sequence_transition):
        """시퀀스 데이터를 메모리에 추가"""
        if not isinstance(sequence_transition, SequenceTransition):
            raise TypeError("데이터는 SequenceTransition 타입이어야 합니다")
        
        max_priority = np.max(self.tree.tree[-self.tree.capacity:])
        if max_priority == 0:
            max_priority = 1.0

        self.tree.add(max_priority, sequence_transition)

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
        is_weight = np.power(len(self.tree.data) * sampling_probabilities, -beta)
        is_weight /= is_weight.max()

        return batch, idxs, is_weight

    def update(self, idx, error):
        p = (np.abs(error) + self.epsilon) ** self.alpha
        self.tree.update(idx, p)

    def __len__(self):
        return self.tree.n_entries
