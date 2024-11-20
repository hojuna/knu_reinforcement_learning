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
            if isinstance(priority, np.ndarray):
                priority = priority.item()
            self.priorities[idx] = priority

    def __len__(self):
        return len(self.memory)

# 3. 수정된 Dueling DQN 모델
class DuelingDQN(nn.Module):
    def __init__(self, input_channels, grid_height, grid_width, num_actions):
        super(DuelingDQN, self).__init__()
        
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
        self.res_block1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64)
        )
        self.res_block2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64)
        )
        
        self.pool = nn.AdaptiveAvgPool2d((2, 2))
        conv_output_size = 64 * 2 * 2
        
        self.dropout = nn.Dropout(0.2)
        
        # Attention mechanism 수정
        self.attention_dim = 256  # Fixed attention dimension
        self.num_heads = 4  # Must be a factor of attention_dim
        
        self.feature_projection = nn.Linear(conv_output_size + 5, self.attention_dim)
        self.attention = nn.MultiheadAttention(self.attention_dim, self.num_heads)
        
        self.fc_value = nn.Sequential(
            nn.Linear(self.attention_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.value = nn.Linear(512, 1)
        
        self.fc_advantage = nn.Sequential(
            nn.Linear(self.attention_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.advantage = nn.Linear(512, num_actions)

    def forward(self, grid, hit_points, direction):
        x = self.conv1(grid)
        x = self.conv2(x)
        x = F.relu(self.res_block1(x) + x)
        x = F.relu(self.res_block2(x) + x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        
        hit_points = hit_points.view(x.size(0), -1)
        direction = direction.view(x.size(0), -1)
        
        x = torch.cat((x, hit_points, direction), dim=1)
        x = self.feature_projection(x)  # Project to attention_dim
        x = self.dropout(x)
        
        # Reshape for attention
        x = x.unsqueeze(0)  # Add sequence length dimension
        x_attn, _ = self.attention(x, x, x)
        x_attn = x_attn.squeeze(0)
        
        value = self.fc_value(x_attn)
        value = self.value(value)
        
        advantage = self.fc_advantage(x_attn)
        advantage = self.advantage(advantage)
        
        return value + (advantage - advantage.mean(dim=1, keepdim=True))

# 4. 수정된 DQN Agent
class ImprovedDQNAgent(GridSurvivorAgent):
    def __init__(self, input_channels, grid_height, grid_width, num_actions, device, config=None):
        self.device = device
        self.policy_net = DuelingDQN(input_channels, grid_height, grid_width, num_actions).to(device)
        self.target_net = DuelingDQN(input_channels, grid_height, grid_width, num_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.policy_net.double().float()
        self.target_net.double().float()
        
        self.num_actions = num_actions
        self.action_counts = np.zeros(num_actions, dtype=np.float32)
        
        if config is None:
            config = {
                "learning_rate": 1e-4,
                "gamma": 0.99,
                "batch_size": 128,
                "memory_capacity": 100000,
                "target_update_steps": 1000,
                "frame_skip": 0,
                "epsilon_start": 1.0,
                "epsilon_end": 0.01,
                "epsilon_decay": 10000
            }
        
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
        self.epsilon_start = config.get("epsilon_start", 1.0)
        self.epsilon_end = config.get("epsilon_end", 0.01)
        self.epsilon_decay = config.get("epsilon_decay", 10000)
        
        self.eval_scores = []
        self.eval_interval = config.get("eval_interval", 1000)

    def select_action(self, state, evaluate=False):
        grid, hit_points, direction = state
        grid_tensor = torch.from_numpy(np.array(grid, dtype=np.float32)).unsqueeze(0).to(self.device)
        hit_points_tensor = torch.from_numpy(np.array(hit_points, dtype=np.float32)).unsqueeze(0).to(self.device)
        direction_tensor = torch.from_numpy(np.array(direction, dtype=np.float32)).unsqueeze(0).to(self.device)
        
        if evaluate:
            with torch.no_grad():
                q_values = self.policy_net(grid_tensor, hit_points_tensor, direction_tensor)
                return q_values.argmax(dim=1).item()
        else:
            if random.random() < self.get_epsilon():
                return random.randrange(self.num_actions)
            with torch.no_grad():
                q_values = self.policy_net(grid_tensor, hit_points_tensor, direction_tensor)
                return q_values.argmax(dim=1).item()
            
    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())


    def act(self, state, ucb_c):
        self.steps_done += 1
        # Convert input to float32
        grid_tensor = torch.from_numpy(state[0]).float().unsqueeze(0).to(self.device)
        hit_points_tensor = torch.from_numpy(state[1]).float().unsqueeze(0).to(self.device)
        direction_tensor = torch.from_numpy(state[2]).float().unsqueeze(0).to(self.device)
        
        q_values = self.policy_net(grid_tensor, hit_points_tensor, direction_tensor)
        q_values = q_values.squeeze().detach().cpu().numpy()
        ucb_values = q_values + ucb_c * np.sqrt(np.log(self.steps_done) / (self.action_counts + 1e-5))
        action = np.argmax(ucb_values)
        self.action_counts[action] += 1
        return action
            
    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return None
            
        transitions, indices, weights = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        
        # Convert all inputs to float32
        grid_batch = torch.from_numpy(np.stack([s[0] for s in batch.state])).float().to(self.device)
        hit_points_batch = torch.from_numpy(np.stack([s[1] for s in batch.state])).float().to(self.device)
        direction_batch = torch.from_numpy(np.stack([s[2] for s in batch.state])).float().to(self.device)
        
        next_grid_batch = torch.from_numpy(np.stack([s[0] for s in batch.next_state])).float().to(self.device)
        next_hit_points_batch = torch.from_numpy(np.stack([s[1] for s in batch.next_state])).float().to(self.device)
        next_direction_batch = torch.from_numpy(np.stack([s[2] for s in batch.next_state])).float().to(self.device)
        
        action_batch = torch.tensor(batch.action, dtype=torch.int64).to(self.device)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float32).to(self.device)
        done_batch = torch.tensor(batch.done, dtype=torch.float32).to(self.device)
        weights = weights.clone().detach().to(self.device)
        
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
                
        weights = weights.unsqueeze(1)
        loss = (F.smooth_l1_loss(state_action_values, expected_state_action_values, reduction='none') * weights).mean()
        
        td_errors = torch.abs(state_action_values - expected_state_action_values).detach().cpu().numpy()
        self.memory.update_priorities(indices, td_errors + 1e-6)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.scheduler.step()
        
        self.update_step_counter += 1
        if self.update_step_counter % self.target_update_steps == 0:
            self.update_target_network()
            
        return loss.item()
def calculate_improved_reward(state, next_state, done, step):
    base_reward = -0.01

    # survival_reward = 0.01 * (step / 100)
    survival_reward=0

    previous_bees = np.sum(state['grid'] == 'B')
    current_bees = np.sum(next_state['grid'] == 'B')
    rescued_bees = previous_bees - current_bees

    bee_reward = rescued_bees * 100



    previous_hornet = np.sum(state['grid'] == 'H')
    current_hornet = np.sum(next_state['grid'] == 'H')
    rescued_hornet = previous_hornet - current_hornet
    if state['hit_points']>=40:
        hornet_penalty = rescued_hornet * -5
    else:
        hornet_penalty = rescued_hornet * -10


    health_reward = 0
    # health_diff = next_state['hit_points'] - state['hit_points']
    # health_reward = health_diff * 0.1


    early_termination = 0

    if current_bees != 0 and done:
        early_termination-=50


    position, direction = extract_agent_info(state['grid'])
    next_position, next_direction = extract_agent_info(next_state['grid'])
    collision_penalty = -0.1 if position == next_position  else 0


    wall_penalty = -1 if next_state['grid'][next_position] == 'W' else 0


    bee_distance_reward = 0
    if current_bees > 0:
        bee_positions = np.argwhere(next_state['grid'] == 'B')
        distances = np.linalg.norm(bee_positions - np.array(next_position), axis=1)
        bee_distance_reward = 1 / (np.min(distances) + 1e-6) * 2


    hornet_distance_penalty = 0
    if current_bees > 0:
        bee_positions = np.argwhere(next_state['grid'] == 'H')
        distances = np.linalg.norm(bee_positions - np.array(next_position), axis=1)
        hornet_distance_penalty = -1 / (np.min(distances) + 1e-6)

    killerbee_distance_penalty = 0
    if current_bees > 0:
        bee_positions = np.argwhere(next_state['grid'] == 'K')
        distances = np.linalg.norm(bee_positions - np.array(next_position), axis=1)
        killerbee_distance_penalty = -1 / (np.min(distances) + 1e-6)


    total_reward = (base_reward + survival_reward + bee_reward + hornet_penalty + 
                   health_reward + early_termination + collision_penalty + 
                   wall_penalty + bee_distance_reward + hornet_distance_penalty +
                   killerbee_distance_penalty)
    return total_reward

def encode_grid(grid):
    channels = {'B': 0, 'H': 1, 'K': 2, 'A': 3}
    grid_encoded = np.zeros((len(channels), grid.shape[0], grid.shape[1]), dtype=np.float32)
    for symbol, idx in channels.items():
        if symbol in ['B', 'H', 'K']:
            grid_encoded[idx][grid == symbol] = 1.0

        elif symbol == 'A':
            # 에이전트의 위치를 표시
            agent_positions = np.argwhere(np.isin(grid, ['AU', 'AD', 'AL', 'AR']))
            for pos in agent_positions:
                grid_encoded[idx][tuple(pos)] = 1.0
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


def test_agent(agent, env, num_episodes=10, render=True):
    agent.policy_net.eval()
    total_rewards = []
    for episode in range(num_episodes):
        state, _ = env.reset()
        grid = state['grid']
        hit_points = state['hit_points']
        direction = extract_agent_info(grid)[1]
        state_input = preprocess_state(grid, hit_points, direction)
        done = False
        episode_reward = 0
        while not done:
            action = agent.select_action(state_input, evaluate=True)
            next_state, reward, done, _, _ = env.step(action)
            episode_reward += reward
            if done:
                break
            grid = next_state['grid']
            hit_points = next_state['hit_points']
            direction = extract_agent_info(grid)[1]
            state_input = preprocess_state(grid, hit_points, direction)
            if render:
                env.render()
        total_rewards.append(episode_reward)
        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}")
    agent.policy_net.train()
    print(f"Average Reward over {num_episodes} episodes: {np.mean(total_rewards):.2f}")



def load_checkpoint(agent, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=agent.device)
    
    agent.policy_net.load_state_dict(checkpoint)
    agent.target_net.load_state_dict(checkpoint)
    
    print(f"Loaded checkpoint from {checkpoint_path}")
    
    return None

def main(trial):
    
    wandb.init(
        project="grid-survivor-dqn",
        name=f"dueling_dqn_run_{int(time.time())}",
        config={
            "learning_rate": trial.suggest_loguniform("learning_rate", 1e-5, 1e-3),
            "gamma": trial.suggest_uniform("gamma", 0.95, 0.99),
            "batch_size": trial.suggest_categorical("batch_size", [128]),
            "memory_capacity": trial.suggest_categorical("memory_capacity", [100000, 200000]),
            "num_channels": 4,
            "grid_height": 34,
            "grid_width": 34,
            "num_actions": 3,
            "target_update_steps": trial.suggest_categorical("target_update_steps", [500, 1000, 2000]),
            "checkpoint_interval": 100,
            "eval_interval": 1000,
            "num_episodes": 40000,
            "frame_skip": 4,
            "priority_alpha": 0.6,
            "priority_beta_start": 0.4,
            "ucb_c": trial.suggest_uniform("ucb_c", 0.1, 2.0),
            "max_steps_per_episode": 1200
        }
    )
    
    config = wandb.config   
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    env = make_grid_survivor(show_screen=False)
    eval_env = make_grid_survivor(show_screen=False)
    num_agents = 3
    agents = []
    for _ in range(num_agents):
        agent = ImprovedDQNAgent(
            input_channels=config["num_channels"],
            grid_height=config["grid_height"],
            grid_width=config["grid_width"],
            num_actions=config["num_actions"],
            device=device,
            config=config
        )
        agents.append(agent)
    best_eval_score = float('-inf')
    for episode in range(1, config["num_episodes"] + 1):
        state, _ = env.reset()
        grid = state['grid']
        hit_points = state['hit_points']
        direction = extract_agent_info(grid)[1]
        state_input = preprocess_state(grid, hit_points, direction)
        episode_rewards = [0] * num_agents
        episode_losses = [[] for _ in range(num_agents)]
        step_count = 0
        done = False
        while not done and step_count < config["max_steps_per_episode"]:
            for i, agent in enumerate(agents):
                action = agent.act(state_input, config["ucb_c"])
                next_state, reward, done, _, _ = env.step(action)

                
                movement_reward = calculate_improved_reward(state, next_state, done, step_count)
                
                remain_bee = np.sum(next_state['grid'] == 'B')
                if 0 == remain_bee:
                    movement_reward += 500 / (step_count + 1e-6)

                episode_rewards[i] += movement_reward
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
                    episode_losses[i].append(loss)
                if done:
                    break
            state = next_state
            state_input = next_state_input
            step_count += 1
        avg_episode_reward = np.mean(episode_rewards)
        avg_episode_loss = np.mean([np.mean(losses) for losses in episode_losses if losses])
        if episode % config["eval_interval"] == 0:
            eval_scores = [evaluate_agent(agent, eval_env) for agent in agents]
            avg_eval_score = np.mean(eval_scores)
            if avg_eval_score > best_eval_score:
                best_eval_score = avg_eval_score
                for i, agent in enumerate(agents):
                    torch.save(agent.policy_net.state_dict(), f"best_model_{i}.pth")
            print(f"Episode {episode}: Avg. Reward = {avg_episode_reward:.2f}, Avg. Loss = {avg_episode_loss:.4f}, Avg. Eval Score = {avg_eval_score:.2f}")
            trial.report(avg_eval_score, episode)
            if trial.should_prune():
               raise optuna.TrialPruned()
           
        log_dict = {
            "Episode": episode,
            "Episode Reward": float(avg_episode_reward),
            "Episode Average Loss": float(avg_episode_loss),
            "Episode Steps": int(step_count),
            "Learning Rate": float(agent.scheduler.get_last_lr()[0]),
            "Final Hit Points": float(next_hit_points),
            "remain Bee": int(remain_bee)
        }
        
        wandb.log(log_dict)
       
        if episode % config["checkpoint_interval"] == 0:
            for i, agent in enumerate(agents):
                torch.save(agent.policy_net.state_dict(), f"/home/comoz/main_project/knu_reinforcement_learning/project2/checkpoint3/checkpoint_model_{i}_episode_{episode}.pth")
            
    
    wandb.finish()
    return best_eval_score

def objective(trial):
   try:
       best_eval_score = main(trial)
       return best_eval_score
   except Exception as e:
       print(f"Trial failed with exception: {e}")
       return float('-inf')

if __name__ == "__main__":
    # 학습 또는 테스트 모드 선택
    # mode = input("Enter 'train' for training or 'test' for testing: ")

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)
    print("Best hyperparameters:")
    print(study.best_params)
    # mode = 'test'

    # if mode == 'train':
    #     # 학습 모드
    #     checkpoint_path = input("Enter the path to the checkpoint file (leave blank for fresh training): ")
    #     study = optuna.create_study(direction="maximize")
        
    #     if checkpoint_path:
    #         # 체크포인트 로드 및 추가 학습
    #         best_params = torch.load(checkpoint_path)['best_params']
    #         study.optimize(objective, n_trials=100, **best_params)
    #     else:
    #         # 새로운 학습 시작
    #         study.optimize(objective, n_trials=100)
        
    #     print("Best hyperparameters:")
    #     print(study.best_params)
    
    # elif mode == 'test':
    #     # 테스트 모드
    #     checkpoint_path = f"/Users/hojun/2024_project/knu_grid_adventure/project2/checkpoint2/checkpoint_model_2_episode_100.pth"
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #     env = make_grid_survivor(show_screen=True)
        
    #     agent = ImprovedDQNAgent(
    #         input_channels=4,
    #         grid_height=34,
    #         grid_width=34,
    #         num_actions=3,
    #         device=device
    #     )
        
        
    #     load_checkpoint(agent, checkpoint_path)
    #     test_agent(agent, env, num_episodes=10, render=True)
    #     env.close()
        
    # else:
    #     print("Invalid mode. Please enter 'train' or 'test'.")