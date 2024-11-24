import torch
import torch.nn as nn


class A2CNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_actions):
        super(A2CNetwork, self).__init__()
        
        # 배치 정규화를 제거하고 더 단순한 구조로 변경
        self.shared = nn.Sequential(
            nn.Linear(input_size, hidden_size, dtype=torch.float32),
            nn.ReLU(),
            nn.Dropout(0.1),  # 배치 정규화 대신 드롭아웃 사용
            
            nn.Linear(hidden_size, hidden_size // 2, dtype=torch.float32),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.policy = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 4, dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, num_actions, dtype=torch.float32),
            nn.Softmax(dim=-1)
        )
        
        self.value = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 4, dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 1, dtype=torch.float32)
        )
        
        # 가중치 초기화
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=1.0)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, x):
        # 입력 형태 보장
        if x.dim() == 1:
            x = x.unsqueeze(0)  # (N, input_size) 형태로 변환
        x = x.float()
        
        shared_features = self.shared(x)
        return self.policy(shared_features), self.value(shared_features)

