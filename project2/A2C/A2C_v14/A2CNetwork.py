import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from collections import deque
import random
from knu_rl_env.grid_survivor import GridSurvivorAgent, make_grid_survivor
import wandb


class A2CNetwork(nn.Module):
   def __init__(self, input_size, hidden_size, num_actions):
       super(A2CNetwork, self).__init__()
       
       # 특징 추출을 위한 공통 레이어 (살짝 더 깊게)
       self.shared = nn.Sequential(
           nn.Linear(input_size, hidden_size),
           nn.ReLU(),
           nn.LayerNorm(hidden_size),  # 학습 안정화
           
           nn.Linear(hidden_size, hidden_size),
           nn.ReLU(),
           nn.LayerNorm(hidden_size),
           
           nn.Linear(hidden_size, hidden_size),
           nn.ReLU()
       )
       
       # Actor: 정책 (행동 확률 출력)
       self.policy = nn.Sequential(
           nn.Linear(hidden_size, hidden_size // 2),
           nn.ReLU(),
           nn.Linear(hidden_size // 2, num_actions),
           nn.Softmax(dim=-1)
       )
       
       # Critic: 가치 함수 (상태 가치 출력)
       self.value = nn.Sequential(
           nn.Linear(hidden_size, hidden_size // 2),
           nn.ReLU(),
           nn.Linear(hidden_size // 2, 1)
       )
   
   def forward(self, x):
       shared_features = self.shared(x)
       action_probs = self.policy(shared_features)
       state_value = self.value(shared_features)
       return action_probs, state_value

