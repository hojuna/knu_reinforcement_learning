import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math
from torch.nn.init import kaiming_uniform_, zeros_

class AttentionModule(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv_query = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv_key = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv_value = nn.Conv2d(in_channels, in_channels, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.size()
        
        query = self.conv_query(x).view(batch, -1, height * width)
        key = self.conv_key(x).view(batch, -1, height * width)
        value = self.conv_value(x).view(batch, -1, height * width)
        
        attention = F.softmax(torch.bmm(query.transpose(1, 2), key), dim=-1)
        out = torch.bmm(value, attention.transpose(1, 2))
        
        return out.view(batch, channels, height, width)

class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)

class DuelingDQN(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        
        # 더 깊은 CNN 특징 추출기 (입력 채널 5개로 수정)
        self.features = nn.Sequential(
            # 첫 번째 블록
            nn.Conv2d(6, 32, kernel_size=3, stride=1, padding=1),  # 입력 채널 5로 수정
            nn.BatchNorm2d(32),
            nn.ReLU(),
            ResidualBlock(32),  # ResidualBlock 추가
            nn.MaxPool2d(2, 2),  # 크기 1/2로 감소
            
            # 두 번째 블록
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            ResidualBlock(64),  # ResidualBlock 추가
            nn.MaxPool2d(2, 2),  # 크기 1/2로 감소
            
            # 세 번째 블록
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            ResidualBlock(128),  # ResidualBlock 추가
            nn.MaxPool2d(2, 2),  # 크기 1/2로 감소
            
            # 네 번째 블록
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            ResidualBlock(256),  # ResidualBlock 추가
            nn.MaxPool2d(2, 2),  # 크기 1/2로 감소
        )
        
        # Spatial Attention
        self.attention = AttentionModule(256)
        
        # 특징 맵의 크기 계산
        def conv2d_size_out(size, kernel_size=2, stride=2):
            return size // stride
        
        convw = convh = input_size
        for _ in range(4):  # 4번의 MaxPool2d
            convw = conv2d_size_out(convw)
            convh = conv2d_size_out(convh)
        
        linear_input_size = convw * convh * 256
        
        # Value Stream (더 깊게)
        self.value_stream = nn.Sequential(
            nn.Linear(linear_input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # Advantage Stream (더 깊게)
        self.advantage_stream = nn.Sequential(
            nn.Linear(linear_input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3)  # 4개의 행동
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
    
    def forward(self, x, hit_points=None):
        # CNN 특징 추출
        x = self.features(x)
        
        # Spatial Attention 적용
        x = self.attention(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Dueling Network
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        
        # Q값 계산 (Dueling 구조)
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values