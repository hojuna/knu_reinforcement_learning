class DuelingDQN(nn.Module):
    def __init__(self, input_channels, grid_height, grid_width, num_actions):
        super(DuelingDQN, self).__init__()

        # 공통 컨볼루션 레이어
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((2, 2))  # 고정된 출력 크기

        # 컨볼루션 출력 크기 계산
        conv_output_size = 64 * 2 * 2  # AdaptiveAvgPool2d로 (2,2) 고정

        # 가치 스트림
        self.fc_value = nn.Linear(conv_output_size + 1, 512)  # +1 for hit_points
        self.value = nn.Linear(512, 1)

        # 우선순위 스트림
        self.fc_advantage = nn.Linear(conv_output_size + 1, 512)
        self.advantage = nn.Linear(512, num_actions)

    def forward(self, grid, hit_points):
        # grid: (batch_size, channels, height, width)
        # hit_points: (batch_size, 1)

        # 공통 컨볼루션 레이어 통과
        x = F.relu(self.conv1(grid))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)  # 플래튼

        # hit_points의 차원 조정
        hit_points = hit_points.view(x.size(0), -1)

        # 추가 피처(hit_points) 결합
        x = torch.cat((x, hit_points), dim=1)

        # 가치 스트림
        value = F.relu(self.fc_value(x))
        value = self.value(value)

        # 우선순위 스트림
        advantage = F.relu(self.fc_advantage(x))
        advantage = self.advantage(advantage)

        # 최종 Q-값 계산
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values

# 2. Replay Memory 클래스 정의
Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state', 'done'))