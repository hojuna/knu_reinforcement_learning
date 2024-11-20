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
import wandb
from knu_rl_env.grid_survivor import make_grid_survivor, evaluate, run_manual, GridSurvivorAgent

import time

# 1. Transition 정의
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done', 'direction', 'hit_points'))

# 2. Prioritized Experience Replay
class PrioritizedReplayMemory:
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=100000):
        self.capacity = capacity
        self.memory = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1
        
    def beta_by_frame(self, frame_idx):
        return min(1.0, self.beta_start + frame_idx * (1.0 - self.beta_start) / self.beta_frames)
    
    def push(self, state, action, reward, next_state, done, direction, hit_points):
        max_priority = self.priorities.max() if self.memory else 1.0
        
        if len(self.memory) < self.capacity:
            self.memory.append(Transition(state, action, reward, next_state, done, direction, hit_points))
        else:
            self.memory[self.position] = Transition(state, action, reward, next_state, done, direction, hit_points)
        
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        if len(self.memory) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.position]
        
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.memory), batch_size, p=probs)
        samples = [self.memory[idx] for idx in indices]
        
        beta = self.beta_by_frame(self.frame)
        self.frame += 1
        
        weights = (len(self.memory) * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = torch.FloatTensor(weights)
        
        return samples, indices, weights
    
    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            # priorities가 배열일 경우 첫 번째 요소만 추출
            if isinstance(priority, np.ndarray):
                priority = priority.item()  # 스칼라 값으로 변환
            self.priorities[idx] = priority

    def __len__(self):
        return len(self.memory)

# 3. 개선된 Dueling DQN 모델
class DuelingDQN(nn.Module):
    def __init__(self, input_channels, grid_height, grid_width, num_actions):
        super(DuelingDQN, self).__init__()
        
        # Improved Convolutional Layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        self.pool = nn.AdaptiveAvgPool2d((2, 2))
        conv_output_size = 64 * 2 * 2
        
        self.dropout = nn.Dropout(0.2)
        
        # Value Stream
        self.fc_value = nn.Sequential(
            nn.Linear(conv_output_size + 5, 512),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.value = nn.Linear(512, 1)
        
        # Advantage Stream
        self.fc_advantage = nn.Sequential(
            nn.Linear(conv_output_size + 5, 512),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.advantage = nn.Linear(512, num_actions)

    def forward(self, grid, hit_points, direction):
        x = self.conv1(grid)
        x = self.pool(self.conv2(x))
        x = self.pool(self.conv3(x))
        x = x.view(x.size(0), -1)
        
        hit_points = hit_points.view(x.size(0), -1)
        direction = direction.view(x.size(0), -1)
        
        x = torch.cat((x, hit_points, direction), dim=1)
        x = x.float()
        x = self.dropout(x)
        
        value = self.fc_value(x)
        value = self.value(value)
        
        advantage = self.fc_advantage(x)
        advantage = self.advantage(advantage)
        
        return value + (advantage - advantage.mean(dim=1, keepdim=True))

# 4. 개선된 DQN Agent
class ImprovedDQNAgent(GridSurvivorAgent):
    def __init__(self, input_channels, grid_height, grid_width, num_actions, device, config):
        self.device = device
        self.policy_net = DuelingDQN(input_channels, grid_height, grid_width, num_actions).to(device)
        self.target_net = DuelingDQN(input_channels, grid_height, grid_width, num_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.num_actions = num_actions
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config["learning_rate"])
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.95)
        
        self.memory = PrioritizedReplayMemory(
            config["memory_capacity"],
            alpha=config.get("priority_alpha", 0.6),
            beta_start=config.get("priority_beta_start", 0.4)
        )
        
        self.batch_size = config["batch_size"]
        self.gamma = config["gamma"]
        self.target_update_steps = config["target_update_steps"]
        self.update_step_counter = 0
        self.steps_done = 0
        self.frame_skip = config.get("frame_skip", 4)
        
        # Evaluation metrics
        self.eval_scores = []
        self.eval_interval = config.get("eval_interval", 1000)
        
    def load_model(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['model_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Model loaded from {checkpoint_path} (episode: {checkpoint['episode']})")
        
    def select_action(self, state):
        grid, hit_points, direction = state
        grid_tensor = torch.from_numpy(np.array(grid, dtype=np.float32)).unsqueeze(0).to(self.device)
        hit_points_tensor = torch.from_numpy(np.array(hit_points, dtype=np.float32)).unsqueeze(0).to(self.device)
        direction_tensor = torch.from_numpy(np.array(direction, dtype=np.float32)).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            q_values = self.policy_net(grid_tensor, hit_points_tensor, direction_tensor)
            return q_values.argmax(dim=1).item()
            
    def act(self, state, epsilon):
        if random.random() < epsilon:
            return random.randint(0, self.num_actions - 1)
        else:
            return self.select_action(state)
            
    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return None
            
        transitions, indices, weights = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        
        # Batch processing
        grid_batch = torch.from_numpy(np.stack([s[0] for s in batch.state])).to(self.device)
        hit_points_batch = torch.from_numpy(np.stack([s[1] for s in batch.state])).to(self.device)
        direction_batch = torch.from_numpy(np.stack([s[2] for s in batch.state])).to(self.device)
        
        next_grid_batch = torch.from_numpy(np.stack([s[0] for s in batch.next_state])).to(self.device)
        next_hit_points_batch = torch.from_numpy(np.stack([s[1] for s in batch.next_state])).to(self.device)
        next_direction_batch = torch.from_numpy(np.stack([s[2] for s in batch.next_state])).to(self.device)
        
        action_batch = torch.tensor(batch.action, dtype=torch.int64).to(self.device)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float32).to(self.device)
        done_batch = torch.tensor(batch.done, dtype=torch.float32).to(self.device)
        
        # Double DQN
        state_action_values = self.policy_net(
            grid_batch, hit_points_batch, direction_batch
        ).gather(1, action_batch.unsqueeze(1))
        
        with torch.no_grad():
            next_actions = self.policy_net(
                next_grid_batch, next_hit_points_batch, next_direction_batch
            ).max(1)[1].unsqueeze(1)
            
            next_state_values = self.target_net(
                next_grid_batch, next_hit_points_batch, next_direction_batch
            ).gather(1, next_actions)
            
            expected_state_action_values = reward_batch.unsqueeze(1) + \
                (1 - done_batch.unsqueeze(1)) * self.gamma * next_state_values
                
        # Huber loss with importance sampling weights
        weights = weights.to(self.device).unsqueeze(1)
        loss = (F.smooth_l1_loss(state_action_values, expected_state_action_values, reduction='none') * weights).mean()
        
        # Priority update
        td_errors = torch.abs(state_action_values - expected_state_action_values).detach().cpu().numpy()
        self.memory.update_priorities(indices, td_errors + 1e-6)
        
        # Optimization step
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.scheduler.step()
        
        self.update_step_counter += 1
        if self.update_step_counter % self.target_update_steps == 0:
            self.update_target_network()
            
        return loss.item()
        
    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
    def step(self, env, action):
        total_reward = 0
        next_state = None
        done = False
        info = None
        
        for _ in range(self.frame_skip):
            next_state, reward, done, info, _ = env.step(action)
            total_reward += reward
            if done:
                print("done 나오는디?")
                break
        
        return next_state, total_reward, done, info, _

# 5. 개선된 보상 계산
def calculate_improved_reward(state, next_state, done, step):
    base_reward = -0.01
    
    # Survival reward
    survival_reward = 0.01 * (step / 100)
    
    # Bee rescue reward
    previous_bees = np.sum(state['grid'] == 'B')
    current_bees = np.sum(next_state['grid'] == 'B')
    rescued_bees = previous_bees - current_bees
    bee_reward = rescued_bees * 100
    
    # Hornet penalty
    previous_hornet = np.sum(state['grid'] == 'H')
    current_hornet = np.sum(next_state['grid'] == 'H')
    rescued_hornet = previous_hornet - current_hornet
    hornet_penalty = rescued_hornet * -10
    
    # Health reward
    health_diff = next_state['hit_points'] - state['hit_points']
    health_reward = health_diff * 0.5
    
    # Early termination penalty
    early_termination = 0
    if done and step < 1200:
        early_termination = -50 * (1 - step/1200)
    
    # Collision penalty
    position, direction = extract_agent_info(state['grid'])
    next_position, next_direction = extract_agent_info(next_state['grid'])
    collision_penalty = -1 if position == next_position and direction == next_direction else 0
    
    total_reward = (base_reward + survival_reward + bee_reward + hornet_penalty + 
                   health_reward + early_termination + collision_penalty)
    
    return total_reward

# 6. 상태 전처리 함수들
def encode_grid(grid):
    channels = {'B': 0, 'H': 1, 'K': 2, 'A': 3}
    grid_encoded = np.zeros((len(channels), grid.shape[0], grid.shape[1]), dtype=np.float32)
    for symbol, idx in channels.items():
        if symbol in ['B', 'H', 'K']:
            grid_encoded[idx][grid == symbol] = 1.0
    return grid_encoded

def encode_direction(direction):
    direction_map = {'UP': 0, 'DOWN': 1, 'LEFT': 2, 'RIGHT': 3}
    direction_vector = np.zeros(4)
    direction_vector[direction_map[direction]] = 1
    return direction_vector

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

def preprocess_state(grid, hit_points, direction):
    grid_encoded = encode_grid(grid)
    hit_points_normalized = normalize_hit_points(hit_points)
    direction_encoded = encode_direction(direction)
    return grid_encoded, hit_points_normalized, direction_encoded

# 7. 체크포인트 저장 및 로드 함수
def save_checkpoint(agent, episode, loss, reward, filepath):
    checkpoint = {
        'episode': episode,
        'model_state_dict': agent.policy_net.state_dict(),
        'target_state_dict': agent.target_net.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'scheduler_state_dict': agent.scheduler.state_dict(),
        'steps_done': agent.steps_done,
        'loss': loss,
        'reward': reward,
        'eval_scores': agent.eval_scores
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved at episode {episode} to {filepath}")

# 8. 평가 함수
def evaluate_agent(agent, env, num_episodes=5):
    agent.policy_net.eval()
    total_rewards = []
    
    for _ in range(num_episodes):
        state, _ = env.reset()
        grid = state['grid']
        hit_points = state['hit_points']
        direction = extract_agent_info(grid)[1]
        
        state_input = preprocess_state(grid, hit_points, direction)
        done = False
        episode_reward = 0
        
        while not done:
            action = agent.select_action(state_input)
            next_state, reward, done, _, _ = env.step(action)
            episode_reward += reward
            
            if done:
                break
                
            grid = next_state['grid']
            hit_points = next_state['hit_points']
            direction = extract_agent_info(grid)[1]
            state_input = preprocess_state(grid, hit_points, direction)
            
        total_rewards.append(episode_reward)
    
    agent.policy_net.train()
    return np.mean(total_rewards)

# 9. 메인 학습 함수
def main(checkpoint_path=None):
   wandb.init(
       project="grid-survivor-dqn",
       name=f"dueling_dqn_run_{int(time.time())}",
       config={
           "learning_rate": 5e-4,
           "gamma": 0.99,
           "batch_size": 64,
           "memory_capacity": 100000,
           "num_channels": 4,
           "grid_height": 34,
           "grid_width": 34,
           "num_actions": 3,
           "target_update_steps": 1000,
           "checkpoint_interval": 1000,
           "eval_interval": 1000,
           "num_episodes": 40000,
           "frame_skip": 4,
           "priority_alpha": 0.6,
           "priority_beta_start": 0.4,
           "min_epsilon": 0.05,
           "epsilon_decay": 4000000,
           "max_steps_per_episode": 2000
       }
   )
   
   config = wandb.config
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   env = make_grid_survivor(show_screen=False)
   eval_env = make_grid_survivor(show_screen=False)
   
   agent = ImprovedDQNAgent(
       input_channels=config.num_channels,
       grid_height=config.grid_height,
       grid_width=config.grid_width,
       num_actions=config.num_actions,
       device=device,
       config=config
   )
   
   if checkpoint_path and os.path.exists(checkpoint_path):
       agent.load_model(checkpoint_path)
   
   # 학습 메트릭 초기화
   loss_history = []
   reward_history = []
   epsilon_history = []
   eval_scores = []
   best_eval_score = float('-inf')
   
   epsilon_start = 0.5
   epsilon_end = config.min_epsilon
   epsilon_decay = config.epsilon_decay

   for episode in range(1, config.num_episodes + 1):
       state, _ = env.reset()
       grid = state['grid']
       hit_points = state['hit_points']
       direction = extract_agent_info(grid)[1]
       
       state_input = preprocess_state(grid, hit_points, direction)
       episode_reward = 0
       episode_loss = []
       step_count = 0
       
       done = False
       
       while not done and step_count < config.max_steps_per_episode:
           epsilon = epsilon_end + (epsilon_start - epsilon_end) * \
                    math.exp(-1. * agent.steps_done / epsilon_decay)
           agent.steps_done += 1
           
           action = agent.act(state_input, epsilon)
           
        #    total_reward = 0
        #    for _ in range(config.frame_skip):
               
        #        total_reward += reward
        #        if done:
        #            break
           next_state, reward, done, _, _ = env.step(action)
           # 보상 계산
           movement_reward = calculate_improved_reward(state, next_state, done, step_count)
           episode_reward += movement_reward
           
           next_grid = next_state['grid']
           next_hit_points = next_state['hit_points']
           next_direction = extract_agent_info(next_grid)[1]
           next_state_input = preprocess_state(next_grid, next_hit_points, next_direction)
           
           agent.memory.push(
               state_input, action, movement_reward, next_state_input, 
               done, next_direction, next_hit_points
           )
           
           loss = agent.optimize_model()
           if loss is not None:
               episode_loss.append(loss)
           
           state = next_state
           state_input = next_state_input
           
           step_count += 1
           
           if step_count % 100 == 0:
               print(f"Episode {episode}, Step {step_count}, Reward: {episode_reward:.2f}")
           
           if done:
               break

       avg_episode_loss = np.mean(episode_loss) if episode_loss else 0.0
       
       log_dict = {
           "Episode": episode,
           "Episode Reward": float(episode_reward),
           "Episode Average Loss": float(avg_episode_loss),
           "Episode Steps": int(step_count),
           "Episode Final Epsilon": float(epsilon),
           "Learning Rate": float(agent.scheduler.get_last_lr()[0]),
           "Final Hit Points": float(next_hit_points)
       }
       
       wandb.log(log_dict)
       
       if episode % 10 == 0:
           print(f"\nEpisode {episode} completed:")
           for key, value in log_dict.items():
               print(f"{key}: {value}")
           print("-" * 50)
       
       if episode % config.checkpoint_interval == 0:
           save_checkpoint(agent, episode, avg_episode_loss, episode_reward,
                         f'project2/checkpoint6/dueling_dqn_checkpoint_episode_{episode}.pth')

   wandb.finish()

# 10. 모델 테스트 함수
def test_agent(checkpoint_path, num_episodes=10, render=True):
    config = {
        "learning_rate": 5e-4,
        "gamma": 0.99,
        "batch_size": 64,
        "memory_capacity": 100000,
        "target_update_steps": 1000,
        "frame_skip": 4,
        "priority_alpha": 0.6,
        "priority_beta_start": 0.4,
        "num_channels": 4
    }
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = make_grid_survivor(show_screen=render)
    
    agent = ImprovedDQNAgent(
        input_channels=config["num_channels"],
        grid_height=34,
        grid_width=34,
        num_actions=3,
        device=device,
        config=config
    )
    
    agent.load_model(checkpoint_path)
    agent.policy_net.eval()
    
    test_rewards = []
    test_steps = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        grid = state['grid']
        hit_points = state['hit_points']
        direction = extract_agent_info(grid)[1]
        
        state_input = preprocess_state(grid, hit_points, direction)
        episode_reward = 0
        step_count = 0
        done = False
        
        while not done:
            action = agent.select_action(state_input)
            next_state, reward, done, _, _ = env.step(action)
            episode_reward += reward
            
            if done:
                break
            
            grid = next_state['grid']
            hit_points = next_state['hit_points']
            direction = extract_agent_info(grid)[1]
            state_input = preprocess_state(grid, hit_points, direction)
            
            step_count += 1
            
            if render:
                env.render()
        
        test_rewards.append(episode_reward)
        test_steps.append(step_count)
        print(f"Test Episode {episode + 1}: Reward = {episode_reward:.2f}, Steps = {step_count}")
    
    print("\nTest Results:")
    print(f"Average Reward: {np.mean(test_rewards):.2f} ± {np.std(test_rewards):.2f}")
    print(f"Average Steps: {np.mean(test_steps):.2f} ± {np.std(test_steps):.2f}")
    
    if render:
        env.close()

if __name__ == "__main__":
    # 학습 실행
    main()
    
    # # 또는 체크포인트부터 이어서 학습
    # checkpoint_path = "project2/checkpoint6/dueling_dqn_checkpoint_episode_10000.pth"
    # if os.path.exists(checkpoint_path):
    #     main(checkpoint_path=checkpoint_path)
    # else:
    #     main()
    
    # 테스트 실행
    # test_agent("project2/final_model/dueling_dqn_final.pth", num_episodes=10, render=True)