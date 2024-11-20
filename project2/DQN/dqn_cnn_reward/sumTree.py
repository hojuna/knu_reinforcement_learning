import numpy as np

# SumTree 클래스
class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity  # 리프 노드의 수
        self.tree = np.zeros(2 * capacity - 1)  # 트리 노드
        self.data = np.zeros(capacity, dtype=object)  # 실제 데이터
        self.write = 0  # 데이터를 쓸 위치
        self.n_entries = 0

    def _propagate(self, idx, change):
        """트리의 상위 노드에 변경 사항 전파"""
        parent = (idx - 1) // 2
        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def update(self, idx, priority):
        """리프 노드의 우선순위 업데이트"""
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def add(self, priority, data):
        """데이터 추가 및 우선순위 설정"""
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, priority)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    def get_leaf(self, s):
        """합 s에 해당하는 리프 노드 찾기"""
        idx = 0
        while True:
            left = 2 * idx + 1
            right = left + 1
            if left >= len(self.tree):
                leaf = idx
                break
            else:
                if s <= self.tree[left]:
                    idx = left
                else:
                    s -= self.tree[left]
                    idx = right
        data_idx = leaf - self.capacity + 1
        return leaf, self.tree[leaf], self.data[data_idx]

    @property
    def total_priority(self):
        return self.tree[0]