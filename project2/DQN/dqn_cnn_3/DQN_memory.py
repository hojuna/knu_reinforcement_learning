import random
from collections import namedtuple, deque
import numpy as np
import torch
from typing import List, Tuple, Dict
from dataclasses import dataclass

@dataclass
class Experience:
    state: Dict
    action: int
    next_state: Dict
    reward: float
    done: bool
    priority: float = 1.0

class PrioritizedReplayMemory:
    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4):
        self.capacity = capacity
        self.memory = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = 0.001
        self.epsilon = 1e-6
        self.max_priority = 1.0
    
    def push(self, state: Dict, action: int, next_state: Dict, 
            reward: float, done: bool) -> None:
        """새로운 경험을 메모리에 저장"""
        experience = Experience(state, action, next_state, reward, done)
        
        if len(self.memory) < self.capacity:
            self.memory.append(experience)
        else:
            self.memory[self.position] = experience
        
        self.priorities[self.position] = self.max_priority
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int) -> Tuple[List[Experience], np.ndarray, np.ndarray]:
        """우선순위에 기반하여 배치 샘플링"""
        if len(self.memory) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.position]
        
        prios = prios + self.epsilon
        
        probs = prios ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.memory), batch_size, p=probs)
        samples = [self.memory[idx] for idx in indices]
        
        total = len(self.memory)
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        return samples, indices, weights
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        """우선순위 업데이트"""
        for idx, priority in zip(indices, priorities.flatten()):
            priority = float(priority) + self.epsilon
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)
    
    def __len__(self) -> int:
        return len(self.memory)

class EpisodeBuffer:
    """에피소드 단위로 경험을 저장하는 버퍼"""
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.current_episode = []
    
    def add_step(self, experience: Experience) -> None:
        """현재 에피소드에 스텝 추가"""
        self.current_episode.append(experience)
        
        if experience.done:
            self.buffer.append(self.current_episode)
            self.current_episode = []
    
    def sample_episode(self) -> List[Experience]:
        """랜덤하게 하나의 에피소드 샘플링"""
        if not self.buffer:
            return []
        return random.choice(self.buffer)
    
    def __len__(self) -> int:
        return len(self.buffer)