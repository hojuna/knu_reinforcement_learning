import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import math
from collections import namedtuple
import random

from knu_rl_env.grid_survivor import make_grid_survivor, GridSurvivorAgent  # 환경 임포트

# 1. Dueling DQN 모델 정의 (풀링 레이어 제거)
class DuelingDQN(nn.Module):
    def __init__(self, input_channels, grid_height, grid_width, num_actions):
        super(DuelingDQN, self).__init__()
        
        # 컨볼루션 레이어 설정
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        # 컨볼루션 출력 크기 계산
        conv_output_size = 64 * grid_height * grid_width
        
        # 활성화 함수 정의
        self.activation = nn.LeakyReLU(negative_slope=0.01)
        
        # 드롭아웃
        # self.dropout = nn.Dropout(p=0.5)
        
        # 완전 연결 레이어 설정

        self.fc_value = nn.Linear(conv_output_size + 1 + 2 + 4, 512)

        self.value = nn.Linear(512, 1)

        self.fc_advantage = nn.Linear(conv_output_size + 1 + 2 + 4, 512)

        self.advantage = nn.Linear(512, num_actions)
    
    def forward(self, grid, hit_points, agent_direction, agent_position):
        # 공통 컨볼루션 레이어 통과
        x = self.activation(self.bn1(self.conv1(grid)))
        x = self.activation(self.bn2(self.conv2(x)))
        x = self.activation(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)  # 플래튼
        
        # 추가 피처 결합
        x = torch.cat((x, hit_points, agent_direction, agent_position), dim=1)
        
        # 가치 스트림
        value = self.activation(self.fc_value(x))
        # value = self.dropout(value)
        value = self.value(value)
        
        # 우선순위 스트림
        advantage = self.activation(self.fc_advantage(x))
        # advantage = self.dropout(advantage)
        advantage = self.advantage(advantage)
        
        # 최종 Q-값 계산
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values
