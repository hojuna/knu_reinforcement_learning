import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import optuna
from collections import deque, namedtuple
import random
import os
from knu_rl_env.grid_survivor import make_grid_survivor, evaluate, run_manual, GridSurvivorAgent
import time

import wandb

class DQN(nn.Module):
    def __init__(self,inputs_channels,num_actions):
        super(DQN,self).__init__()
    
        # CNN 부분 강화
        self.conv1 = nn.Sequential(
            nn.Conv2d(inputs_channels,64,kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(64,128,kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU()
        )

        # 레지듀얼 블록 추가
        self.res_block1 = self._make_res_block(128)
        self.res_block2 = self._make_res_block(128)

        self.pool = nn.AdaptiveMaxPool2d((4, 4))  # 2x2 대신 4x4로 변경

        conv_output_size = 128 * 4 * 4  # 더 많은 공간 정보 유지
        self.fc1 = nn.Linear(conv_output_size, 1024)
        
        extra_data_size = 12
        
        # 추가 정보 처리 부분 강화
        self.extra_fc1 = nn.Sequential(
            nn.Linear(extra_data_size, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(p=0.1)
        )
        self.extra_fc2 = nn.Sequential(
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(p=0.1)
        )
        
        # 결합 부분 강화
        self.combined_fc = nn.Sequential(
            nn.Linear(1024 + 256, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(p=0.1)
        )
        
        self.fc2 = nn.Linear(256, num_actions)

    def _make_res_block(self, channels):
        return nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels)
        )

    def forward(self, grid, hit_points, direction, forward_obj, agent_position):
        # CNN 특징 추출
        x = self.conv1(grid)
        x = self.conv2(x)
        
        # 레지듀얼 연결 + 활성화 함수
        identity = x
        x = self.res_block1(x)
        x = F.gelu(x + identity)
        
        identity = x
        x = self.res_block2(x)
        x = F.gelu(x + identity)
        
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.gelu(x)

        # 추가 상태 정보 처리
        hit_points = hit_points.view(-1, 1)  # 명시적 shape 지정
        direction = direction.view(-1, 4)     # 4방향
        forward_obj = forward_obj.view(-1, 5) # 5가지 객체
        agent_position = agent_position.view(-1, 2)  # x, y 좌표
        
        # 추가 데이터 결합 및 처리
        extra_data = torch.cat([hit_points, direction, forward_obj, agent_position], dim=1)
        extra = self.extra_fc1(extra_data)
        extra = self.extra_fc2(extra)  # 추가 레이어를 통한 특징 추출
        
        # CNN 특징과 추가 정보 결합
        combined = torch.cat([x, extra], dim=1)
        combined = self.combined_fc(combined)
        
        # 최종 Q-값 출력
        q_values = self.fc2(combined)
        
        return q_values