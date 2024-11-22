import torch
import torch.nn as nn
import torch.nn.functional as F

class DuelingDQN(nn.Module):
    def __init__(self, input_channels, grid_height, grid_width, num_actions):
        super(DuelingDQN, self).__init__()

        # CNN 레이어
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)

        # 컨볼루션 출력 크기 계산
        self.grid_height = grid_height
        self.grid_width = grid_width
        conv_output_size = 64 * grid_height * grid_width  # 출력 채널 * 높이 * 너비

        # LSTM 입력 및 hidden 크기
        self.lstm_input_size = conv_output_size + 1 + 4 + 2  # CNN 출력 + hit_points + agent_direction + agent_position
        self.lstm_hidden_size = 128

        # LSTM 레이어
        self.lstm_layer = nn.LSTM(input_size=self.lstm_input_size, hidden_size=self.lstm_hidden_size, num_layers=1, batch_first=True)

        # Advantage와 Value 스트림을 위한 완전 연결 레이어
        self.fc_advantage = nn.Linear(self.lstm_hidden_size, 256)
        self.fc_value = nn.Linear(self.lstm_hidden_size, 256)
        self.advantage = nn.Linear(256, num_actions)
        self.value = nn.Linear(256, 1)

    def forward(self, grid, hit_points, agent_direction, agent_position, hidden=None):
        batch_size = grid.size(0)

        # CNN 레이어 통과
        x = F.leaky_relu(self.conv1(grid))
        x = F.leaky_relu(self.conv2(x))

        # CNN 출력 플래튼
        x = x.view(batch_size, -1)

        # 추가 피처 결합
        x = torch.cat((x, hit_points, agent_direction, agent_position), dim=1)
        x = x.unsqueeze(1)  # 시퀀스 차원 추가: (batch_size, seq_len=1, input_size)

        # LSTM 레이어 통과
        output, hidden = self.lstm_layer(x, hidden)  # output: (batch_size, seq_len, hidden_size)
        output = output.squeeze(1)  # 시퀀스 차원 제거: (batch_size, hidden_size)

        # Advantage와 Value 계산
        advantage = F.leaky_relu(self.fc_advantage(output))
        value = F.leaky_relu(self.fc_value(output))

        advantage = self.advantage(advantage)
        value = self.value(value)

        # Advantage와 Value 결합하여 Q 값 계산
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)

        return q_values, hidden
