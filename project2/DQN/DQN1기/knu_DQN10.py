# 필요한 라이브러리 임포트
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import math
from collections import deque, namedtuple
import random
import os
import wandb  # wandb 사용 시

from knu_rl_env.grid_survivor import make_grid_survivor, evaluate, run_manual, GridSurvivorAgent  # 환경 임포트

# 1. 어텐션이 적용된 DQN 모델 정의
class AttentionDQN(nn.Module):
    def __init__(self, input_channels, grid_height, grid_width, num_actions):
        super(AttentionDQN, self).__init__()

        # CNN 레이어 정의 (padding 추가)
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4, padding=2)
        self.bn1 = nn.BatchNorm2d(32)  # 배치 정규화 레이어 추가
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)  # 배치 정규화 레이어 추가
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)  # 배치 정규화 레이어 추가

        # 어텐션 레이어 정의
        self.attention_conv = nn.Conv2d(64, 1, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)

        # CNN 출력 크기 계산
        conv_output_size = self.calculate_conv_output_size(input_channels, grid_height, grid_width) + 1  # +1은 hit_points

        # 완전 연결층
        self.fc1 = nn.Linear(conv_output_size, 512)
        self.fc2 = nn.Linear(512, num_actions)

    def calculate_conv_output_size(self, input_channels, height, width):
        # 입력 데이터의 크기 확인
        x = torch.zeros(1, input_channels, height, width)
        x = F.relu(self.bn1(self.conv1(x)))  # 배치 정규화 적용
        x = F.relu(self.bn2(self.conv2(x)))  # 배치 정규화 적용
        x = F.relu(self.bn3(self.conv3(x)))  # 배치 정규화 적용
        output_size = x.view(1, -1).size(1)
        return output_size

    def forward(self, grid, hit_points):
        # CNN 통과
        x = F.relu(self.bn1(self.conv1(grid)))   # 배치 정규화 적용
        x = F.relu(self.bn2(self.conv2(x)))      # 배치 정규화 적용
        x = F.relu(self.bn3(self.conv3(x)))      # 배치 정규화 적용

        # 어텐션 가중치 계산
        attn_weights = self.attention_conv(x)  # (batch_size, 1, H, W)
        attn_weights = attn_weights.view(x.size(0), -1)  # (batch_size, H*W)
        attn_weights = self.softmax(attn_weights)
        attn_weights = attn_weights.view(x.size(0), 1, x.size(2), x.size(3))  # (batch_size, 1, H, W)

        # 어텐션 가중치 적용
        x = x * attn_weights  # (batch_size, 64, H, W)

        # 플래튼
        x = x.view(x.size(0), -1)

        # hit_points 결합
        hit_points = hit_points.view(x.size(0), -1)
        x = torch.cat((x, hit_points), dim=1)

        # 완전 연결층 통과
        x = F.relu(self.fc1(x))
        q_values = self.fc2(x)
        return q_values

# 2. Replay Memory 클래스 정의 (변경 없음)
Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """새로운 경험을 메모리에 추가"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        """무작위로 경험을 샘플링"""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# 3. 어텐션 DQN 에이전트 클래스 정의
class AttentionDQNAgent(GridSurvivorAgent):
    def __init__(self, input_channels, grid_height, grid_width, num_actions, device, config):
        self.device = device
        self.policy_net = AttentionDQN(input_channels, grid_height, grid_width, num_actions).to(device)
        self.target_net = AttentionDQN(input_channels, grid_height, grid_width, num_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # 타겟 네트워크는 평가 모드로 설정
        self.num_actions = num_actions
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config["learning_rate"])
        self.memory = ReplayMemory(config["memory_capacity"])
        self.batch_size = config["batch_size"]
        self.gamma = config["gamma"]

        # 타겟 네트워크 업데이트 주기를 스텝 단위로 변경
        self.target_update_steps = config["target_update_steps"]  # 스텝 단위 업데이트 주기
        self.update_step_counter = 0  # 스텝 카운터 초기화

        self.steps_done = 0

    def load_model(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['model_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_state_dict'])  # 타겟 네트워크 로드
        print(f"모델이 {checkpoint_path}에서 로드되었습니다. (에피소드: {checkpoint['episode']})")

    def select_action(self, state):
        """
        주어진 상태에서 행동을 선택합니다.

        Args:
            state (tuple): 전처리된 상태 (grid_encoded, hit_points_normalized).

        Returns:
            int: 선택된 행동.
        """
        grid, hit_points = state
        grid_tensor = torch.from_numpy(np.array(grid, dtype=np.float32)).unsqueeze(0).to(self.device)
        hit_points_tensor = torch.from_numpy(np.array(hit_points, dtype=np.float32)).unsqueeze(0).to(self.device)

        with torch.no_grad():
            q_values = self.policy_net(grid_tensor, hit_points_tensor)
            action = q_values.argmax(dim=1).item()
        return action

    def act(self, state, epsilon):
        """
        ε-그리디 탐사 전략을 사용하여 행동을 선택합니다.

        Args:
            state (tuple): 전처리된 상태.
            epsilon (float): 탐사 확률.

        Returns:
            int: 선택된 행동.
        """
        if random.random() < epsilon:
            return random.randint(0, self.num_actions - 1)
        else:
            return self.select_action(state)

    def optimize_model(self):
        """경험 재생 메모리에서 샘플을 추출하여 모델 최적화"""
        if len(self.memory) < self.batch_size:
            return None  # 충분한 경험이 쌓이지 않음

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        # 상태 및 다음 상태 분리
        state_batch = batch.state
        action_batch = torch.tensor(batch.action, dtype=torch.int64).to(self.device)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float32).to(self.device)
        next_state_batch = batch.next_state
        done_batch = torch.tensor(batch.done, dtype=torch.float32).to(self.device)

        # 상태 텐서 변환 (np.stack 사용하여 배치 차원으로 쌓기)
        grid_batch = torch.from_numpy(np.stack([s[0] for s in state_batch])).to(self.device)
        hit_points_batch = torch.from_numpy(np.stack([s[1] for s in state_batch])).to(self.device)

        # 다음 상태 텐서 변환
        next_grid_batch = torch.from_numpy(np.stack([s[0] for s in next_state_batch])).to(self.device)
        next_hit_points_batch = torch.from_numpy(np.stack([s[1] for s in next_state_batch])).to(self.device)

        # 현재 Q 값
        state_action_values = self.policy_net(grid_batch, hit_points_batch).gather(1, action_batch.unsqueeze(1)).squeeze(1)

        # 타겟 Q 값 계산
        with torch.no_grad():
            next_state_values = self.target_net(next_grid_batch, next_hit_points_batch).max(1)[0]
            expected_state_action_values = reward_batch + (1 - done_batch) * self.gamma * next_state_values

        # 손실 계산 (MSE)
        loss = F.mse_loss(state_action_values, expected_state_action_values)

        # 최적화 단계
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)  # 그래디언트 클리핑
        self.optimizer.step()

        # 스텝 카운터 증가 및 타겟 네트워크 업데이트
        self.update_step_counter += 1
        if self.update_step_counter % self.target_update_steps == 0:
            self.update_target_network()

        return loss.item()

    def update_target_network(self):
        """타겟 네트워크를 정책 네트워크의 가중치로 업데이트"""
        self.target_net.load_state_dict(self.policy_net.state_dict())

# 4. 상태 전처리 함수 정의
def encode_grid(grid):
    """그리드 데이터를 원-핫 인코딩하여 5채널 텐서로 변환 (B, H, K, Agent Direction, Agent Position)"""
    channels = {
        'B': 0,   # Bee
        'H': 1,   # Hornet
        'K': 2,   # Killer Bee
        'A': 3,   # Agent Direction (0~1 값으로 정규화된 방향)
        'P': 4    # Agent Position (에이전트 위치 채널)
    }
    grid_encoded = np.zeros((len(channels), grid.shape[0], grid.shape[1]), dtype=np.float32)

    # 기존 객체들 인코딩
    for symbol, idx in channels.items():
        if symbol in ['B', 'H', 'K']:
            grid_encoded[idx][grid == symbol] = 1.0

    # 에이전트 방향 인코딩
    direction_map = {'AU': 0, 'AD': 1, 'AL': 2, 'AR': 3}
    for direction_symbol, value in direction_map.items():
        grid_encoded[channels['A']][grid == direction_symbol] = value / 3.0  # 0, 0.33, 0.67, 1.0 사이 값

    # 에이전트 위치 인코딩
    agent_position, _ = extract_agent_info(grid)
    if agent_position is not None:
        x, y = agent_position
        grid_encoded[channels['P']][x, y] = 1.0  # 에이전트 위치에 1.0 할당

    return grid_encoded

def normalize_hit_points(hit_points, max_hp=100):
    """체력 정보를 0과 1 사이로 정규화"""
    return np.array([hit_points / max_hp], dtype=np.float32)

def extract_agent_info(grid):
    """그리드에서 에이전트의 위치와 방향을 추출합니다."""
    direction_symbols = {'AU': 'UP', 'AD': 'DOWN', 'AL': 'LEFT', 'AR': 'RIGHT'}
    positions = np.argwhere(np.isin(grid, list(direction_symbols.keys())))
    if len(positions) == 0:
        # 에이전트가 그리드에 없을 경우
        return (None, None)
    x, y = positions[0]
    symbol = grid[x, y]
    direction = direction_symbols.get(symbol, 'DOWN')
    return (x, y), direction

def preprocess_state(grid, hit_points):
    """환경 상태를 신경망 입력 형태로 전처리 (전체 그리드 사용)"""
    # 그리드 인코딩
    grid_encoded = encode_grid(grid)
    hit_points_normalized = normalize_hit_points(hit_points)
    return grid_encoded, hit_points_normalized

# 5. 보상 계산 함수 정의 (변경 없음)
def calculate_adjusted_reward(state, next_state, done, step):
    """보상 구조를 조정하여 긍정적인 보상과 패널티를 균형 있게 설정"""
    # 기본 타임스텝 보상
    adjusted_reward = -0.05  # 기본 타임스텝 보상 감소

    # 꿀벌 구출 확인
    previous_bees = np.sum(state['grid'] == 'B')
    current_bees = np.sum(next_state['grid'] == 'B')
    rescued_bees = previous_bees - current_bees
    if rescued_bees > 0:
        adjusted_reward += 100  # 긍정적인 보상 증가

    # 말벌 구출 확인
    previous_hornet = np.sum(state['grid'] == 'H')
    current_hornet = np.sum(next_state['grid'] == 'H')
    rescued_hornet = previous_hornet - current_hornet
    if rescued_hornet > 0:
        adjusted_reward -= 10  # 패널티

    # 에피소드 조기 종료 시 패널티
    if done and step < 1200:
        adjusted_reward -= 50  # 패널티 절반으로 감소

    position, direction = extract_agent_info(state['grid'])
    next_position, next_direction = extract_agent_info(next_state['grid'])

    # 제자리 패널티 패널티
    if position == next_position and direction == next_direction:
        adjusted_reward -= 1

    return adjusted_reward

# 6. 그리드 시각화 함수 정의 (변경 없음)
def log_grid(grid, episode):
    """환경의 그리드 상태를 이미지로 저장"""
    # 그리드 기호를 숫자로 매핑하는 사전 정의
    symbol_to_num = {
        'E': 0,   # Empty
        'W': 1,   # Wall
        'B': 2,   # Bee
        'H': 3,   # Hornet
        'K': 4,   # Killer Bee
        'AU': 5,  # Agent Up
        'AD': 6,  # Agent Down
        'AL': 7,  # Agent Left
        'AR': 8   # Agent Right
    }

    # NumPy의 vectorize를 사용하여 기호를 숫자로 변환
    vectorized_mapping = np.vectorize(lambda x: symbol_to_num.get(x, -1))  # 정의되지 않은 기호는 -1로 설정
    numerical_grid = vectorized_mapping(grid)

    plt.figure(figsize=(5, 5))
    plt.imshow(numerical_grid, cmap='tab20', vmin=0, vmax=8)
    plt.title(f"Episode {episode}")
    plt.colorbar(ticks=range(0, 9))
    plt.savefig(f"project2/savefig3/grid_episode_{episode}.png")
    plt.close()
    print(f"Saved grid visualization for Episode {episode} as grid_episode_{episode}.png")

# 7. 체크포인트 저장 함수 정의 (변경 없음)
def save_checkpoint(agent, episode, loss, reward, filepath):
    checkpoint = {
        'episode': episode,
        'model_state_dict': agent.policy_net.state_dict(),
        'target_state_dict': agent.target_net.state_dict(),  # 타겟 네트워크 추가
        'loss': loss,
        'reward': reward
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved at episode {episode} to {filepath}")

# 8. 메인 학습 루프 구현
def main():
    # WandB 초기화 (선택 사항)
    wandb.init(
        project="grid-survivor-dqn",
        config={
            "learning_rate": 5e-4,
            "gamma": 0.95,
            "batch_size": 32,  # 메모리 사용량을 고려하여 배치 크기 조정
            "memory_capacity": 100000,
            "num_channels": 5,  # B, H, K, Agent Direction, Agent Position
            "grid_height": 34,   # 34x34 그리드
            "grid_width": 34,
            "num_actions": 3,  # 0: LEFT, 1: RIGHT, 2: FORWARD
            "target_update_steps": 1000,  # 1000 스텝마다 타겟 네트워크 업데이트
            "checkpoint_interval": 1000,
            "num_episodes": 40000,  # 학습 에피소드 수 증가
            "name" : "DQN_attention_34x34"
        }
    )

    config = wandb.config

    # 환경 초기화
    env = make_grid_survivor(show_screen=False)

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # 그리드 크기 설정
    grid_shape = (config.grid_height, config.grid_width)
    num_channels = config.num_channels
    num_actions = config.num_actions

    # 에이전트 초기화
    agent = AttentionDQNAgent(
        input_channels=num_channels,
        grid_height=grid_shape[0],
        grid_width=grid_shape[1],
        num_actions=num_actions,
        device=device,
        config=config
    )

    num_episodes = config.num_episodes
    checkpoint_interval = config.checkpoint_interval

    # 손실과 보상을 저장할 리스트
    loss_history = []
    reward_history = []

    # ε-그리디 탐사 파라미터 조정
    epsilon_start = 1.0
    epsilon_end = 0.1  # 최소 ε 값 설정
    epsilon_decay = 500000  # ε 값이 더 천천히 감소하도록 설정

    for episode in range(1, num_episodes + 1):
        state, _ = env.reset()
        grid = state['grid']
        hit_points = state['hit_points']

        # 상태 전처리
        state_input = preprocess_state(grid, hit_points)

        done = False
        total_reward = 0
        loss_value = None
        step_count = 0  # 에피소드 내 스텝 카운터

        while not done:
            # ε 값 계산
            epsilon = epsilon_end + (epsilon_start - epsilon_end) * math.exp(-1. * agent.steps_done / epsilon_decay)
            agent.steps_done += 1

            # 행동 선택
            action = agent.act(state_input, epsilon)

            # 행동 수행
            next_state, reward, done, _, _ = env.step(action)
            next_grid = next_state['grid']
            next_hit_points = next_state['hit_points']

            # 보상 계산
            movement_reward = calculate_adjusted_reward(state, next_state, done, step_count)
            total_reward += movement_reward

            # 상태 전처리
            next_state_input = preprocess_state(next_grid, next_hit_points)

            # 경험 저장
            agent.memory.push(state_input, action, movement_reward, next_state_input, done)

            # 상태 업데이트
            state_input = next_state_input
            state = next_state

            # 모델 최적화
            loss = agent.optimize_model()
            if loss is not None:
                loss_value = loss

            step_count += 1  # 스텝 카운터 증가

        # 체크포인트 저장
        if episode % checkpoint_interval == 0:
            checkpoint_path = f'project2/checkpoint3/attention_dqn_34x34_checkpoint_episode_{episode}.pth'
            save_checkpoint(agent, episode, loss_value, total_reward, checkpoint_path)
            wandb.save(checkpoint_path)

        # 로그 기록
        loss_history.append(loss_value)
        reward_history.append(total_reward)
        wandb.log({
            "Episode": episode,
            "Total Reward": total_reward,
            "Loss": loss_value,
            "Epsilon": epsilon
        })

        # 그리드 시각화 (선택 사항)
        if episode % 100 == 0:
            log_grid(next_state['grid'], episode)

        # 콘솔 출력
        if episode % 100 == 0:
            print(f"Episode {episode}: Total Reward: {total_reward}, Loss: {loss_value}, Epsilon: {epsilon:.4f}")

    # 학습 과정 시각화
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_episodes + 1), reward_history, label='Total Reward')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Episode vs Total Reward')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_episodes + 1), loss_history, label='Loss', color='orange')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.title('Episode vs Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_progress_34x34.png')
    plt.show()

    # 최종 모델 저장
    torch.save(agent.policy_net.state_dict(), "attention_dqn_grid_survivor_34x34_final.pth")
    wandb.finish()

# 9. 모델 테스트 함수 정의 (변경 없음)
def run_trained_model(checkpoint_path, num_episodes=10, render=True):
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    grid_shape = (34, 34)
    num_channels = 5
    num_actions = 3

    # 에이전트 초기화
    agent = AttentionDQNAgent(
        input_channels=num_channels,
        grid_height=grid_shape[0],
        grid_width=grid_shape[1],
        num_actions=num_actions,
        device=device,
        config={
            "learning_rate": 5e-4,
            "gamma": 0.95,
            "batch_size": 32,
            "memory_capacity": 100000,
            "target_update_steps": 1000,
            "num_channels": 5
        }
    )

    # 모델 로드 (타겟 네트워크도 포함)
    agent.load_model(checkpoint_path)
    agent.policy_net.eval()

    # 환경 초기화
    env = make_grid_survivor(show_screen=render)

    for episode in range(1, num_episodes + 1):
        state, _ = env.reset()
        grid = state['grid']
        hit_points = state['hit_points']

        # 상태 전처리
        state_input = preprocess_state(grid, hit_points)

        done = False
        total_reward = 0
        step_count = 0

        while not done:
            # 행동 선택 (ε-그리디 탐사 없이 행동 선택)
            action = agent.act(state_input, epsilon=0.0)  # 탐사 없이 결정론적 행동 선택
            next_state, reward, done, _, _ = env.step(action)
            next_grid = next_state['grid']
            next_hit_points = next_state['hit_points']

            # 보상 계산
            movement_reward = calculate_adjusted_reward(state, next_state, done, step_count)
            total_reward += movement_reward

            # 상태 전처리
            next_state_input = preprocess_state(next_grid, next_hit_points)

            # 상태 업데이트
            state_input = next_state_input
            state = next_state

            step_count += 1

            if render:
                env.render()

        print(f"Test Episode {episode}: Total Reward: {total_reward}, Steps: {step_count}")
        log_grid(next_state['grid'], episode)

    if render:
        env.render()
    print("Finished running trained model.")

# 10. 실행 부분
if __name__ == "__main__":
    # 학습을 진행하려면 main() 함수를 호출하세요.
    main()

    # 학습이 완료된 후 테스트를 실행하려면 아래 코드를 사용하세요.
    # run_trained_model("project2/checkpoint3/attention_dqn_34x34_checkpoint_episode_80000.pth", num_episodes=10, render=True)
