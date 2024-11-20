import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DQN(nn.Module):
    def __init__(self, input_channels, num_actions):
        super(DQN, self).__init__()
        
        # CNN 레이어
        self.conv_layers = nn.Sequential(
            # 첫 번째 블록 (35x35 -> 17x17)
            nn.Conv2d(input_channels, 48, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(48),
            nn.GELU(),
            nn.MaxPool2d(2, 2),
            
            # 두 번째 블록 (17x17 -> 8x8)
            nn.Conv2d(48, 96, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(96),
            nn.GELU(),
            nn.MaxPool2d(2, 2),
            
            # 세 번째 블록 (8x8 -> 8x8)
            nn.Conv2d(96, 192, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(192),
            nn.GELU(),
        )
        
        # Spatial Attention
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(192, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Channel Attention (SE Block)
        self.se_block = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(192, 12, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(12, 192, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 특징 추출을 위한 Flatten
        self.flatten = nn.Flatten()
        
        # HP 정보를 처리하는 레이어
        self.hp_encoder = nn.Sequential(
            nn.Linear(1, 64),
            nn.GELU(),
            nn.Linear(64, 128),
            nn.GELU()
        )
        
        # Dueling Network 구조
        feature_size = 192 * 8 * 8  # 수정된 크기

        self.advantage_stream = nn.Sequential(
            nn.Linear(feature_size + 128, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_actions)
        )
        
        self.value_stream = nn.Sequential(
            nn.Linear(feature_size + 128, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 1)
        )
        
        # 가중치 초기화
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(module.bias, 0)

    def forward(self, grid, hit_points):
        # NumPy 배열을 PyTorch 텐서로 변환하고 올바른 디바이스로 이동
        if isinstance(grid, np.ndarray):
            grid = torch.from_numpy(grid).float()
        if isinstance(hit_points, (int, float)):
            hit_points = torch.tensor([[hit_points]], dtype=torch.float)
        elif isinstance(hit_points, np.ndarray):
            hit_points = torch.from_numpy(hit_points).float()
        
        # 현재 모델의 디바이스로 텐서 이동
        device = next(self.parameters()).device
        grid = grid.to(device)
        hit_points = hit_points.to(device)
        
        # CNN 특징 추출
        x = self.conv_layers(grid)
        
        # Attention 적용
        spatial_weights = self.spatial_attention(x)
        channel_weights = self.se_block(x)
        
        x = x * spatial_weights * channel_weights
        
        # Flatten
        x = self.flatten(x)
        
        # HP 정보 처리
        hp_features = self.hp_encoder(hit_points)
        
        # 특징 결합
        combined_features = torch.cat([x, hp_features], dim=1)
        
        # Dueling Network
        advantage = self.advantage_stream(combined_features)
        value = self.value_stream(combined_features)
        
        # Q값 계산
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values 