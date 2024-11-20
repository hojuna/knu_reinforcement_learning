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

# 1. PPO 모델 정의 (Actor-Critic 구조)
class PPOActorCritic(nn.Module):
    def __init__(self, input_channels, grid_height, grid_width, num_actions):
        super(PPOActorCritic, self).__init__()

        # 공유된 CNN 레이어 정의 (padding 추가)
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        # CNN 출력 크기 계산
        conv_output_size = self.calculate_conv_output_size(input_channels, grid_height, grid_width) + 1  # +1은 hit_points

        # Actor (정책 네트워크)
        self.fc_actor = nn.Sequential(
            nn.Linear(conv_output_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions),
        )

        # Critic (가치 함수 네트워크)
        self.fc_critic = nn.Sequential(
            nn.Linear(conv_output_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

    def calculate_conv_output_size(self, input_channels, height, width):
        x = torch.zeros(1, input_channels, height, width)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        output_size = x.view(1, -1).size(1)
        return output_size

    def forward(self, grid, hit_points):
        x = F.relu(self.bn1(self.conv1(grid)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # 플래튼
        x = x.view(x.size(0), -1)

        # hit_points 결합
        hit_points = hit_points.view(x.size(0), -1)
        x = torch.cat((x, hit_points), dim=1)

        # Actor와 Critic 분리
        action_logits = self.fc_actor(x)
        state_values = self.fc_critic(x).squeeze(1)

        return action_logits, state_values

# 2. PPO 에이전트 클래스 정의
class PPOAgent(GridSurvivorAgent):
    def __init__(self, input_channels, grid_height, grid_width, num_actions, device, config):
        self.device = device
        self.policy_net = PPOActorCritic(input_channels, grid_height, grid_width, num_actions).to(device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config["learning_rate"])
        self.num_actions = num_actions

        # PPO 하이퍼파라미터
        self.clip_param = config["clip_param"]
        self.ppo_epochs = config["ppo_epochs"]
        self.batch_size = config["batch_size"]
        self.gamma = config["gamma"]
        self.lam = config["lambda"]  # GAE(lambda)에 사용

        # 경험 저장용 버퍼
        self.memory = []

    def select_action(self, state):
        grid, hit_points = state
        grid_tensor = torch.from_numpy(np.array(grid, dtype=np.float32)).unsqueeze(0).to(self.device)
        hit_points_tensor = torch.from_numpy(np.array(hit_points, dtype=np.float32)).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action_logits, _ = self.policy_net(grid_tensor, hit_points_tensor)
            action_probs = F.softmax(action_logits, dim=-1)
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()
            action_log_prob = action_dist.log_prob(action)
        return action.item(), action_log_prob.item()
    
    def act(self,state):
        return self.select_action(state)

    def store_transition(self, transition):
        self.memory.append(transition)

    def compute_gae(self, rewards, masks, values, next_value):
        values = values + [next_value]
        gae = 0
        returns = []
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * values[step + 1] * masks[step] - values[step]
            gae = delta + self.gamma * self.lam * masks[step] * gae
            returns.insert(0, gae + values[step])
        return returns

    def optimize_model(self):
        # 메모리에서 데이터 추출
        states = [t.state for t in self.memory]
        actions = torch.tensor([t.action for t in self.memory], dtype=torch.long).to(self.device)
        old_log_probs = torch.tensor([t.log_prob for t in self.memory], dtype=torch.float32).to(self.device)
        rewards = [t.reward for t in self.memory]
        masks = [1 - t.done for t in self.memory]

        # 상태 텐서 변환
        grid_batch = torch.from_numpy(np.stack([s[0] for s in states])).to(self.device)
        hit_points_batch = torch.from_numpy(np.stack([s[1] for s in states])).to(self.device)

        # 가치 함수 계산
        _, values = self.policy_net(grid_batch, hit_points_batch)
        values = values.detach().cpu().numpy().tolist()

        # GAE 계산
        next_state = self.memory[-1].next_state
        next_grid = torch.from_numpy(np.array(next_state[0], dtype=np.float32)).unsqueeze(0).to(self.device)
        next_hit_points = torch.from_numpy(np.array(next_state[1], dtype=np.float32)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            _, next_value = self.policy_net(next_grid, next_hit_points)
            next_value = next_value.item()
        returns = self.compute_gae(rewards, masks, values, next_value)

        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        advantages = returns - torch.tensor(values, dtype=torch.float32).to(self.device)

        # PPO 업데이트
        for _ in range(self.ppo_epochs):
            for i in range(0, len(states), self.batch_size):
                batch_slice = slice(i, i + self.batch_size)
                batch_grid = grid_batch[batch_slice]
                batch_hit_points = hit_points_batch[batch_slice]
                batch_actions = actions[batch_slice]
                batch_old_log_probs = old_log_probs[batch_slice]
                batch_returns = returns[batch_slice]
                batch_advantages = advantages[batch_slice]

                # 현재 정책에서의 확률과 가치 함수 계산
                action_logits, state_values = self.policy_net(batch_grid, batch_hit_points)
                action_probs = F.softmax(action_logits, dim=-1)
                action_dist = torch.distributions.Categorical(action_probs)
                action_log_probs = action_dist.log_prob(batch_actions)

                # 확률비 계산
                ratios = torch.exp(action_log_probs - batch_old_log_probs)

                # 클리핑된 손실 함수 계산
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1.0 - self.clip_param, 1.0 + self.clip_param) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                # 가치 함수 손실
                critic_loss = F.mse_loss(state_values, batch_returns)

                # 전체 손실
                loss = actor_loss + 0.5 * critic_loss  # 가치 함수 손실에 가중치 0.5 적용

                # 업데이트
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
                self.optimizer.step()

        # 메모리 초기화
        self.memory = []

        return loss.item()

# 3. Transition 클래스 정의
Transition = namedtuple('Transition',
                        ('state', 'action', 'log_prob', 'reward', 'next_state', 'done'))

# 4. 상태 전처리 함수 정의 (변경 없음)
def encode_grid(grid):
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
    return np.array([hit_points / max_hp], dtype=np.float32)

def extract_agent_info(grid):
    direction_symbols = {'AU': 'UP', 'AD': 'DOWN', 'AL': 'LEFT', 'AR': 'RIGHT'}
    positions = np.argwhere(np.isin(grid, list(direction_symbols.keys())))
    if len(positions) == 0:
        return (None, None)
    x, y = positions[0]
    symbol = grid[x, y]
    direction = direction_symbols.get(symbol, 'DOWN')
    return (x, y), direction

def preprocess_state(grid, hit_points):
    grid_encoded = encode_grid(grid)
    hit_points_normalized = normalize_hit_points(hit_points)
    return grid_encoded, hit_points_normalized

# 5. 보상 계산 함수 정의 (변경 가능)
def calculate_adjusted_reward(state, next_state, done, step):
    adjusted_reward = -0.05

    previous_bees = np.sum(state['grid'] == 'B')
    current_bees = np.sum(next_state['grid'] == 'B')
    rescued_bees = previous_bees - current_bees
    if rescued_bees > 0:
        adjusted_reward += 150  # 보상 증가

    previous_hornet = np.sum(state['grid'] == 'H')
    current_hornet = np.sum(next_state['grid'] == 'H')
    rescued_hornet = previous_hornet - current_hornet
    if rescued_hornet > 0:
        adjusted_reward -= 50  # 패널티 감소

    if done and step < 1200:
        adjusted_reward -= 50

    position, direction = extract_agent_info(state['grid'])
    next_position, next_direction = extract_agent_info(next_state['grid'])

    if position == next_position and direction == next_direction:
        adjusted_reward -= 1

    return adjusted_reward

# 6. 그리드 시각화 함수 정의 (변경 없음)
def log_grid(grid, episode):
    symbol_to_num = {
        'E': 0,
        'W': 1,
        'B': 2,
        'H': 3,
        'K': 4,
        'AU': 5,
        'AD': 6,
        'AL': 7,
        'AR': 8
    }

    vectorized_mapping = np.vectorize(lambda x: symbol_to_num.get(x, -1))
    numerical_grid = vectorized_mapping(grid)

    plt.figure(figsize=(5, 5))
    plt.imshow(numerical_grid, cmap='tab20', vmin=0, vmax=8)
    plt.title(f"Episode {episode}")
    plt.colorbar(ticks=range(0, 9))
    plt.savefig(f"project2/savefig6/grid_episode_{episode}.png")
    plt.close()
    print(f"Saved grid visualization for Episode {episode} as grid_episode_{episode}.png")

# 7. 체크포인트 저장 함수 정의
def save_checkpoint(agent, episode, reward, filepath):
    checkpoint = {
        'episode': episode,
        'model_state_dict': agent.policy_net.state_dict(),
        'reward': reward
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved at episode {episode} to {filepath}")

# 8. 메인 학습 루프 구현
def main():
    # WandB 초기화 (선택 사항)
    wandb.init(
        project="grid-survivor-ppo",
        config={
            "learning_rate": 3e-4,
            "gamma": 0.99,
            "lambda": 0.95,
            "clip_param": 0.2,
            "ppo_epochs": 4,
            "batch_size": 64,
            "num_channels": 5,
            "grid_height": 34,
            "grid_width": 34,
            "num_actions": 3,
            "checkpoint_interval": 1000,
            "num_episodes": 80000,
            "name": "PPO_agent_34x34"
        }
    )

    config = wandb.config

    # 환경 초기화
    env = make_grid_survivor(show_screen=False)

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    grid_shape = (config.grid_height, config.grid_width)
    num_channels = config.num_channels
    num_actions = config.num_actions

    # 에이전트 초기화
    agent = PPOAgent(
        input_channels=num_channels,
        grid_height=grid_shape[0],
        grid_width=grid_shape[1],
        num_actions=num_actions,
        device=device,
        config=config
    )

    num_episodes = config.num_episodes
    checkpoint_interval = config.checkpoint_interval

    reward_history = []

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
            # 행동 선택
            action, action_log_prob = agent.select_action(state_input)

            # 행동 수행
            next_state, reward, done, _, _ = env.step(action)
            next_grid = next_state['grid']
            next_hit_points = next_state['hit_points']

            # 보상 계산
            movement_reward = calculate_adjusted_reward(state, next_state, done, step_count)
            total_reward += movement_reward

            # 다음 상태 전처리
            next_state_input = preprocess_state(next_grid, next_hit_points)

            # 경험 저장
            transition = Transition(state_input, action, action_log_prob, movement_reward, next_state_input, done)
            agent.store_transition(transition)

            # 상태 업데이트
            state_input = next_state_input
            state = next_state

            step_count += 1

        # 에피소드 종료 후 모델 업데이트
        loss = agent.optimize_model()

        # 체크포인트 저장
        if episode % checkpoint_interval == 0:
            checkpoint_path = f'project2/checkpoint6/ppo_agent_34x34_checkpoint_episode_{episode}.pth'
            save_checkpoint(agent, episode, total_reward, checkpoint_path)
            wandb.save(checkpoint_path)

        # 로그 기록
        reward_history.append(total_reward)
        wandb.log({
            "Episode": episode,
            "Total Reward": total_reward,
            "Loss": loss
        })

        # 그리드 시각화 (선택 사항)
        if episode % 100 == 0:
            log_grid(next_state['grid'], episode)

        # 콘솔 출력
        if episode % 100 == 0:
            print(f"Episode {episode}: Total Reward: {total_reward}, Loss: {loss}")

    # 학습 과정 시각화
    plt.figure(figsize=(12, 5))
    plt.plot(range(1, num_episodes + 1), reward_history, label='Total Reward')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Episode vs Total Reward')
    plt.legend()
    plt.savefig('ppo_training_progress_34x34.png')
    plt.show()

    # 최종 모델 저장
    torch.save(agent.policy_net.state_dict(), "ppo_agent_grid_survivor_34x34_final.pth")
    wandb.finish()

# 9. 모델 테스트 함수 정의
def run_trained_model(checkpoint_path, num_episodes=10, render=True):
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    grid_shape = (34, 34)
    num_channels = 5
    num_actions = 3

    # 에이전트 초기화
    agent = PPOAgent(
        input_channels=num_channels,
        grid_height=grid_shape[0],
        grid_width=grid_shape[1],
        num_actions=num_actions,
        device=device,
        config={
            "learning_rate": 3e-4,
            "gamma": 0.99,
            "lambda": 0.95,
            "clip_param": 0.2,
            "ppo_epochs": 4,
            "batch_size": 64
        }
    )

    # 모델 로드
    checkpoint = torch.load(checkpoint_path, map_location=device)
    agent.policy_net.load_state_dict(checkpoint['model_state_dict'])
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
            # 행동 선택
            action, _ = agent.select_action(state_input)
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
    # run_trained_model("project2/checkpoint3/ppo_agent_34x34_checkpoint_episode_80000.pth", num_episodes=10, render=True)
