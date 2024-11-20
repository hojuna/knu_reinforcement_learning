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
import sys
import argparse
import time

# Optional wandb import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("WandB is not installed. Continuing without experiment tracking.")

from knu_rl_env.grid_survivor import make_grid_survivor, evaluate, run_manual, GridSurvivorAgent  # 환경 임포트

# 1. Dueling DQN 모델 정의 (더 깊게 설계)
class DuelingDQN(nn.Module):
    def __init__(self, input_channels, grid_height, grid_width, num_actions):
        super(DuelingDQN, self).__init__()

        # 공통 컨볼루션 레이어
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool = nn.AdaptiveAvgPool2d((2, 2))  # 고정된 출력 크기
        self.dropout = nn.Dropout(p=0.5)  # 과적합 방지를 위한 드롭아웃

        # 컨볼루션 출력 크기 계산
        conv_output_size = 128 * 2 * 2  # AdaptiveAvgPool2d로 (2,2) 고정

        # 가치 스트림
        self.fc_value1 = nn.Linear(conv_output_size + 1, 512)
        self.fc_value2 = nn.Linear(512, 256)
        self.value = nn.Linear(256, 1)

        # 우선순위 스트림
        self.fc_advantage1 = nn.Linear(conv_output_size + 1, 512)
        self.fc_advantage2 = nn.Linear(512, 256)
        self.advantage = nn.Linear(256, num_actions)

        # 가중치 초기화
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, grid, hit_points):
        # grid: (batch_size, channels, height, width)
        # hit_points: (batch_size, 1)

        # 공통 컨볼루션 레이어 통과
        x = F.relu(self.bn1(self.conv1(grid)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.dropout(x)
        x = x.view(x.size(0), -1)  # 플래튼

        # hit_points의 차원 조정
        hit_points = hit_points.view(x.size(0), -1)

        # 추가 피처(hit_points) 결합
        x = torch.cat((x, hit_points), dim=1)

        # 가치 스트림
        value = F.relu(self.fc_value1(x))
        value = F.relu(self.fc_value2(value))
        value = self.value(value)

        # 우선순위 스트림
        advantage = F.relu(self.fc_advantage1(x))
        advantage = F.relu(self.fc_advantage2(advantage))
        advantage = self.advantage(advantage)

        # 최종 Q-값 계산
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values

# 2. Replay Memory 클래스 정의
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

# 3. Dueling DQN 에이전트 클래스 정의 (UCB 탐사 전략 추가)
class DuelingDQNAgent(GridSurvivorAgent):
    def __init__(self, input_channels, grid_height, grid_width, num_actions, device, config):
        self.device = device
        self.policy_net = DuelingDQN(input_channels, grid_height, grid_width, num_actions).to(device)
        self.target_net = DuelingDQN(input_channels, grid_height, grid_width, num_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # 타겟 네트워크는 평가 모드로 설정
        self.num_actions = num_actions
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config["learning_rate"])
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10000, gamma=0.1)  # 학습률 스케줄러
        self.memory = ReplayMemory(config["memory_capacity"])
        self.gamma = config["gamma"]

        self.batch_size = config["batch_size"]
        self.target_update_steps = config["target_update_steps"]  # 스텝 단위 업데이트 주기
        self.update_step_counter = 0  # 스텝 카운터 초기화

        self.steps_done = 0

        # UCB 탐사 관련 변수
        self.action_counts = np.zeros(self.num_actions)  # 각 행동의 선택 횟수
        self.total_steps = 0  # 전체 행동 선택 횟수
        self.ucb_c = config.get("ucb_c", 1.0)  # UCB 상수

    def load_model(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['model_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_state_dict'])  # 타겟 네트워크 로드
        print(f"모델이 {checkpoint_path}에서 로드되었습니다. (에피소드: {checkpoint['episode']})")

    def select_action(self, state):
        """
        주어진 상태에서 UCB 기반으로 행동을 선택합니다.

        Args:
            state (tuple): 전처리된 상태 (grid_encoded, hit_points_normalized).

        Returns:
            int: 선택된 행동.
        """
        grid, hit_points = state
        grid_tensor = torch.from_numpy(np.array(grid, dtype=np.float32)).unsqueeze(0).to(self.device)
        hit_points_tensor = torch.from_numpy(np.array(hit_points, dtype=np.float32)).unsqueeze(0).to(self.device)

        with torch.no_grad():
            q_values = self.policy_net(grid_tensor, hit_points_tensor).cpu().numpy()[0]

        # UCB 점수 계산
        ucb_scores = q_values + self.ucb_c * np.sqrt(np.log(self.total_steps + 1) / (self.action_counts + 1))
        
        # 최댓값을 가진 행동 선택
        action = np.argmax(ucb_scores)
        
        # 선택 횟수 업데이트
        self.action_counts[action] += 1
        self.total_steps += 1

        return action

    def act(self, state, epsilon):
        """
        ε-그리디 탐사 전략 또는 UCB 기반 탐사 전략을 사용하여 행동을 선택합니다.

        Args:
            state (tuple): 전처리된 상태.
            epsilon (float): 탐사 확률 (UCB 적용 시 무시).

        Returns:
            int: 선택된 행동.
        """
        if random.random() < epsilon:
            action = random.randint(0, self.num_actions - 1)
            self.action_counts[action] += 1
            self.total_steps += 1
            return action
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

        # Double DQN 적용: 타겟 네트워크에서 최대 Q 값을 계산
        with torch.no_grad():
            next_actions = self.policy_net(next_grid_batch, next_hit_points_batch).argmax(1)
            next_state_values = self.target_net(next_grid_batch, next_hit_points_batch).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            expected_state_action_values = reward_batch + (1 - done_batch) * self.gamma * next_state_values

        # 손실 계산 (MSE)
        loss = F.mse_loss(state_action_values, expected_state_action_values)

        # 최적화 단계
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)  # 그래디언트 클리핑
        self.optimizer.step()
        self.scheduler.step()  # 학습률 스케줄러 업데이트

        # 스텝 카운터 증가 및 타겟 네트워크 업데이트
        self.update_step_counter += 1
        if self.update_step_counter % self.target_update_steps == 0:
            self.update_target_network()

        return loss.item()

    def update_target_network(self):
        """타겟 네트워크를 정책 네트워크의 가중치로 업데이트"""
        self.target_net.load_state_dict(self.policy_net.state_dict())
        print("타겟 네트워크가 업데이트되었습니다.")

# 4. 상태 전처리 함수 정의
def encode_grid(grid):
    """그리드 데이터를 원-핫 인코딩하여 4채널 텐서로 변환 (B, H, K, Agent Direction)"""
    channels = {
        'B': 0,   # Bee
        'H': 1,   # Hornet
        'K': 2,   # Killer Bee
        'A': 3    # Agent Direction (0~1 값으로 정규화된 방향)
    }
    grid_encoded = np.zeros((len(channels), grid.shape[0], grid.shape[1]), dtype=np.float32)

    # 기존 객체들 인코딩
    for symbol, idx in channels.items():
        if symbol in ['B', 'H', 'K']:
            grid_encoded[idx][grid == symbol] = 1.0

    # 에이전트 방향 인코딩
    # 방향을 정수로 매핑 후 정규화하여 0~1 사이로 변환
    direction_map = {'AU': 0, 'AD': 1, 'AL': 2, 'AR': 3}
    for direction_symbol, value in direction_map.items():
        grid_encoded[channels['A']][grid == direction_symbol] = value / 3.0  # 0, 0.33, 0.67, 1.0 사이 값

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
    """환경 상태를 신경망 입력 형태로 전처리"""
    grid_encoded = encode_grid(grid)  # (채널, 높이, 너비)
    hit_points_normalized = normalize_hit_points(hit_points)  # (1,)
    return grid_encoded, hit_points_normalized

# 5. 보상 계산 함수 정의
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

    # 어딘가에 부딪혔을 때 패널티
    if position == next_position and direction == next_direction:
        adjusted_reward -= 1

    return adjusted_reward

# 6. 그리드 시각화 함수 정의
# def log_grid(grid, episode, save_dir='project2/savefig5/'):
#     """환경의 그리드 상태를 이미지로 저장"""
#     # 디렉토리 생성
#     os.makedirs(save_dir, exist_ok=True)

#     # 그리드 기호를 숫자로 매핑하는 사전 정의
#     symbol_to_num = {
#         'E': 0,   # Empty
#         'W': 1,   # Wall
#         'B': 2,   # Bee
#         'H': 3,   # Hornet
#         'K': 4,   # Killer Bee
#         'AU': 5,  # Agent Up
#         'AD': 6,  # Agent Down
#         'AL': 7,  # Agent Left
#         'AR': 8   # Agent Right
#     }

#     # NumPy의 vectorize를 사용하여 기호를 숫자로 변환
#     vectorized_mapping = np.vectorize(lambda x: symbol_to_num.get(x, 0))  # 정의되지 않은 기호는 'E'로 설정
#     numerical_grid = vectorized_mapping(grid)

#     plt.figure(figsize=(5, 5))
#     plt.imshow(numerical_grid, cmap='tab20', vmin=0, vmax=8)
#     plt.title(f"Episode {episode}")
#     plt.colorbar(ticks=range(0, 9))
#     plt.savefig(os.path.join(save_dir, f"grid_episode_{episode}.png"))
#     plt.close()
#     print(f"Saved grid visualization for Episode {episode} as grid_episode_{episode}.png")

# 7. 체크포인트 저장 함수 정의
def save_checkpoint(agent, episode, loss, reward, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)  # 디렉토리 생성
    checkpoint = {
        'episode': episode,
        'model_state_dict': agent.policy_net.state_dict(),
        'target_state_dict': agent.target_net.state_dict(),  # 타겟 네트워크 추가
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'scheduler_state_dict': agent.scheduler.state_dict(),
        'loss': loss,
        'reward': reward
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved at episode {episode} to {filepath}")

# 8. 메인 학습 루프 구현
def main(config):
    # WandB 초기화 (선택 사항)
    if WANDB_AVAILABLE:
        wandb.init(
            project="grid-survivor-dueling-dqn-ucb",
            config=config
        )
        wandb_config = wandb.config
    else:
        wandb_config = config

    # 환경 초기화
    env = make_grid_survivor(show_screen=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 초기 상태 가져오기
    state, _ = env.reset()
    grid_shape = state['grid'].shape
    num_channels = config["num_channels"]
    num_actions = config["num_actions"]

    # 에이전트 초기화
    agent = DuelingDQNAgent(
        input_channels=num_channels,
        grid_height=grid_shape[0],
        grid_width=grid_shape[1],
        num_actions=num_actions,
        device=device,
        config=wandb_config if WANDB_AVAILABLE else config
    )

    num_episodes = config["num_episodes"]
    checkpoint_interval = config["checkpoint_interval"]

    # 손실과 보상을 저장할 리스트
    loss_history = []
    reward_history = []

    # ε-그리디 탐사 파라미터 조정 (UCB와 병행 사용 가능)
    epsilon_start = 0.5
    epsilon_end = 0.1  # 최소 ε 값 설정
    epsilon_decay = 4000000  # ε 값이 더 빠르게 감소하도록 설정

    try:
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
                checkpoint_path = f'project2/checkpoint7/dueling_dqn_ucb_checkpoint_episode_{episode}.pth'
                save_checkpoint(agent, episode, loss_value, total_reward, checkpoint_path)
                if WANDB_AVAILABLE:
                    wandb.save(checkpoint_path)

            # 로그 기록
            loss_history.append(loss_value)
            reward_history.append(total_reward)
            if WANDB_AVAILABLE:
                wandb.log({
                    "Episode": episode,
                    "Total Reward": total_reward,
                    "Loss": loss_value,
                    "Epsilon": epsilon
                })

            # # 그리드 시각화 (선택 사항)
            # if episode % 100 == 0:
            #     log_grid(next_state['grid'], episode)

            # 콘솔 출력
            if episode % 100 == 0:
                print(f"Episode {episode}: Total Reward: {total_reward}, Loss: {loss_value}, Epsilon: {epsilon:.4f}")

    except KeyboardInterrupt:
        print("학습이 중단되었습니다. 체크포인트를 저장합니다...")
        checkpoint_path = f'project2/checkpoint7/dueling_dqn_ucb_checkpoint_episode_{episode}.pth'
        save_checkpoint(agent, episode, loss_value, total_reward, checkpoint_path)
        if WANDB_AVAILABLE:
            wandb.save(checkpoint_path)
        sys.exit()

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
    plt.savefig('training_progress.png')
    plt.show()

    # 최종 모델 저장
    torch.save(agent.policy_net.state_dict(), "dueling_dqn_ucb_grid_survivor_final.pth")
    if WANDB_AVAILABLE:
        wandb.finish()

# 9. 모델 테스트 함수 정의
def run_trained_model(checkpoint_path, num_episodes=10, render=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    grid_shape = (34, 34)
    num_channels = 4
    num_actions = 3

    # 에이전트 초기화
    agent = DuelingDQNAgent(
        input_channels=num_channels,
        grid_height=grid_shape[0],
        grid_width=grid_shape[1],
        num_actions=num_actions,
        device=device,
        config={
            "learning_rate": 5e-4,
            "gamma": 0.95,
            "batch_size": 64,
            "memory_capacity": 100000,
            "target_update_steps": 1000,
            "ucb_c": 1.0,
            "num_channels": 4  # 실제 입력 채널 수와 일치하도록 수정
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
            action = agent.select_action(state_input)  # UCB 기반 행동 선택
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
        # log_grid(next_state['grid'], episode)

    if render:
        env.render()
    print("Finished running trained model.")

# 10. 실행 부분
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dueling DQN with UCB for GridSurvivor")
    parser.add_argument('--train', action='store_true', help='Train the agent')
    parser.add_argument('--test', action='store_true', help='Test the agent')
    parser.add_argument('--checkpoint', type=str, help='Path to the checkpoint file for testing')
    parser.add_argument('--episodes', type=int, default=40000, help='Number of training episodes')
    parser.add_argument('--test_episodes', type=int, default=10, help='Number of test episodes')
    parser.add_argument('--render', action='store_true', help='Render the environment during testing')
    args = parser.parse_args()

    if args.train:
        # 학습을 진행하려면 main() 함수를 호출하세요.
        config = {
            "learning_rate": 5e-4,
            "gamma": 0.95,
            "batch_size": 64,
            "memory_capacity": 100000,
            "num_channels": 4,  # B, H, K, Agent Direction
            "grid_height": 34,
            "grid_width": 34,
            "num_actions": 3,  # 0: LEFT, 1: RIGHT, 2: FORWARD
            "target_update_steps": 1000,  # 1000 스텝마다 타겟 네트워크 업데이트
            "checkpoint_interval": 1000,
            "num_episodes": args.episodes,
            "ucb_c": 1.0  # UCB 상수
        }
        main(config)
    elif args.test:
        if args.checkpoint is None:
            print("테스트를 위해서는 --checkpoint 경로를 제공해야 합니다.")
            sys.exit(1)
        run_trained_model(args.checkpoint, num_episodes=args.test_episodes, render=args.render)
    else:
        print("Please specify --train or --test.")
