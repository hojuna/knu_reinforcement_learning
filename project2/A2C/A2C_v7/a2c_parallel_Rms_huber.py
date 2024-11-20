import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from collections import deque
import random
from knu_rl_env.grid_survivor import make_grid_survivor
import wandb
import torch.multiprocessing as mp
import os
from multiprocessing import Lock
import signal
import sys
import time
import collections
import threading 
import queue
import logging
import signal
import time
import json
import math
import copy

# GPU 관련 정보 확인
# GPU 관련 정보 확인
# gpu_available = torch.cuda.is_available()
# if gpu_available:
#     gpu_count = torch.cuda.device_count()
target_device = 1  # cuda:1 사용
device = torch.device(f"cuda:{target_device}")
#     gpu_memory = torch.cuda.get_device_properties(target_device).total_memory / 1024**3
# else:
#     gpu_count = 0
#     gpu_memory = 0
#     device = torch.device("cpu")

# num_cores = multiprocessing.cpu_count()
# memory_per_process = 0.5
# gpu_based_process_limit = int(gpu_memory / memory_per_process) if gpu_available else 0
# num_processes = min(num_cores - 1, gpu_based_process_limit) if gpu_available else num_cores - 1
num_processes=12

# print(f"Available CPU cores: {num_cores}")
# print(f"GPU available: {gpu_available}")
# if gpu_available:
    # print(f"GPU count: {gpu_count}")
    # print(f"GPU memory: {gpu_memory:.2f}GB")
print(f"Using processes: {num_processes}")

class A2CNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_actions):
        super(A2CNetwork, self).__init__()
        
        self.shared = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size),
            
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size),
            
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        self.policy = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, num_actions),
            nn.Softmax(dim=-1)
        )
        
        self.value = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
    
    def forward(self, x):
        shared_features = self.shared(x)
        action_probs = self.policy(shared_features)
        state_value = self.value(shared_features)
        return action_probs, state_value

class A2CAgent:
    def __init__(self, state_size, save_dir=f"/home/comoz/main_project/knu_reinforcement_learning/project2/A2C/A2C_v7/save_model"):
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        self.network = A2CNetwork(state_size, 128, 3).to(self.device)
        self.network.share_memory()  # 병렬 처리를 위한 메모리 공유

        self.optimizer = optim.RMSprop(
            self.network.parameters(),
            lr=0.00025,
            alpha=0.95,
            eps=1e-5,
            momentum=0.0,
            centered=True
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=800,
            eta_min=5e-5
        )
        
        self.gamma = 0.99
        self.entropy_coef = 0.01
        self.value_coef = 0.5
        
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.raw_states = []
        
        self.visit_table = np.ones((35,35))
        self.save_dir = save_dir
    def compute_gae(self, rewards, values, dones, gamma=0.99, lambda_=0.95):
        advantages = []
        gae = 0
        next_value = 0
        
        for r, v, d in zip(reversed(rewards), reversed(values), reversed(dones)):
            delta = r + gamma * next_value * (1 - d) - v
            gae = delta + gamma * lambda_ * (1 - d) * gae
            advantages.insert(0, gae)
            next_value = v
            
        return torch.tensor(advantages, device=self.device)
    
    def preprocess_state(self, state):
        cell_types = {'E':0, 'W':1, 'B':2, 'H':3, 'K':4, 'A':5}
        directions = {'L':0, 'R':1, 'U':2, 'D':3}
        grid = state['grid']
        N = len(grid)
        
        grid_tensor = torch.zeros(N, N, 6)
        direction_tensor = torch.zeros(4)
        
        for i in range(N):
            for j in range(N):
                cell = grid[i][j]
                if cell.startswith('A'):
                    grid_tensor[i, j, 5] = 1
                    direction_tensor[directions[cell[1]]] = 1
                else:
                    grid_tensor[i, j, cell_types[cell]] = 1
        
        grid_flat = grid_tensor.flatten()
        state_vector = torch.cat([
            grid_flat,
            direction_tensor,
            torch.tensor([state['hit_points'] / 100.0])
        ])
        
        return state_vector.to(self.device)
    
    def extract_agent_info(self, grid):
        direction_symbols = {'AU': 'UP', 'AD': 'DOWN', 'AL': 'LEFT', 'AR': 'RIGHT'}
        positions = np.argwhere(np.isin(grid, list(direction_symbols.keys())))
        
        if len(positions) == 0:
            return (None, None)
        
        x, y = positions[0]
        symbol = grid[x, y]
        direction = direction_symbols.get(symbol, 'DOWN')
        
        return (x, y), direction
    
    def find_forward_obj(self, grid):
        priority = ['K', 'H','B','W','E']
        position, direction = self.extract_agent_info(grid)
        
        if position is None or direction is None:
            return None
        
        x, y = position
        grid_height, grid_width = grid.shape
        
        direction_offsets = {
            'UP': [(-2, 0), (-1, -1), (-1, 1),(-1,0)],
            'DOWN': [(2, 0), (1, -1), (1, 1),(1,0)],
            'RIGHT': [(0, 2), (-1, 1), (1, 1),(0,1)],
            'LEFT': [(0, -2), (-1, -1), (1, -1),(0,-1)]
        }
        
        offsets = direction_offsets.get(direction.upper())
        if offsets is None:
            return None
            
        found_symbols = []
        
        for dx, dy in offsets:
            cx, cy = x + dx, y + dy
            if 0 <= cx < grid_height and 0 <= cy < grid_width:
                cell_symbol = grid[cx][cy]
                if cell_symbol in priority:
                    found_symbols.append(cell_symbol)
        
        if not found_symbols:
            return None
            
        for symbol in priority:
            if symbol in found_symbols:
                return symbol
        
        return None
    
    def _get_next_position(self, position, direction):
        x, y = position
        if direction == 'UP':
            return (x-1, y)
        elif direction == 'DOWN':
            return (x+1, y)
        elif direction == 'LEFT':
            return (x, y-1)
        elif direction == 'RIGHT':
            return (x, y+1)
        return position
    
    def _is_valid_move(self, next_pos, grid):
        x, y = next_pos
        if not (0 <= x < len(grid) and 0 <= y < len(grid[0])):
            return False
        if grid[x][y] == 'W' or self.find_forward_obj(grid) == 'K':
            return False
        return True
    
    def get_valid_actions(self, state):
        grid = state['grid']
        valid_actions = torch.ones(3, device=self.device)
        
        position, direction = self.extract_agent_info(grid)
        if position is None:
            return valid_actions
            
        next_pos = self._get_next_position(position, direction)
        if not self._is_valid_move(next_pos, grid):
            valid_actions[2] = 0
        elif self.find_forward_obj(grid) == 'B':
            valid_actions[2] += 1
            
        return valid_actions

    def act(self, state):
        """행동 선택"""
        state_tensor = self.preprocess_state(state)
        valid_actions = self.get_valid_actions(state)
        
        with torch.no_grad():
            action_probs, value = self.network(state_tensor)

                        # 불가능한 행동의 확률을 0으로 만들고 재정규화
            masked_probs = action_probs * valid_actions
            if masked_probs.sum() > 0:  # 적어도 하나의 유효한 행동이 있는 경우
                masked_probs = masked_probs / masked_probs.sum()
            else:
                # 모든 행동이 불가능한 경우 (발생하면 안되지만 안전장치)
                masked_probs = action_probs
            
            # 행동 샘플링
            dist = Categorical(masked_probs)
            action = dist.sample()
        
        
        # 경험 저장
        self.states.append(state_tensor)
        self.raw_states.append(state)
        self.actions.append(action)
        self.values.append(value)
        
        return action.item()
    
    def update(self, reward, done):
        self.rewards.append(reward)
        self.dones.append(done)
        
        if done:
            loss = self.learn()
            self.reset_episode()
            return loss
        return 0

    def calculate_reward(self,state, next_state, done, step):
        total_reward = 0

        base_reward = -0.1

        # survival_reward = 0.01 * (step / 100)
        survival_reward=0

        previous_bees = np.sum(state['grid'] == 'B')
        current_bees = np.sum(next_state['grid'] == 'B')
        rescued_bees = previous_bees - current_bees

        bee_reward = rescued_bees * 100

        if 'B'==self.find_forward_obj(next_state):
            bee_reward+=5


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
            early_termination-=20


        position, direction = self.extract_agent_info(state['grid'])
        next_position, next_direction = self.extract_agent_info(next_state['grid'])
        collision_penalty = -0.1 if position == next_position  else 0
        
        first_visit_reward=0
        if self.visit_table[position]==1:
            first_visit_reward=1
            self.visit_table[position]=0

        bee_distance_reward = 0
        # if 'B' in next_state['grid']:
        #     min_distance = self.min_distance(next_state, next_position, 'B')
        #     if min_distance is not None:
        #         bee_distance_reward += 0.1 / (min_distance + 1e-6)

        hornet_distance_penalty = 0
        # if 'H' in next_state['grid']:
        #     min_distance = self.min_distance(next_state, next_position, 'H')
        #     if min_distance is not None:
            
        #         hornet_distance_penalty -= 1 / (min_distance + 1e-6)

        killerbee_distance_penalty = 0
        # if 'K' in next_state['grid']:
        #     min_distance = self.min_distance(next_state, next_position, 'K')
        #     if min_distance is not None:
        #         killerbee_distance_penalty -= 1 / (min_distance + 1e-6)


        if 0 == current_bees and done:
            total_reward += 500 / (step + 1e-6)

        total_reward = (base_reward + survival_reward + bee_reward + hornet_penalty + 
                    health_reward + early_termination + collision_penalty + 
                     bee_distance_reward + hornet_distance_penalty +
                    killerbee_distance_penalty+first_visit_reward)

        return total_reward
    
    
    def learn(self):

        returns = []
        R = 0
        for r, d in zip(reversed(self.rewards), reversed(self.dones)):
            R = r + self.gamma * R * (1-d)
            returns.insert(0, R)
        returns = torch.tensor(returns).to(self.device)
        
        states = torch.stack(self.states)
        actions = torch.stack(self.actions)
        values = torch.stack(self.values)

        # 각 상태에 대한 유효 행동 마스크 준비
        valid_actions_batch = torch.stack([self.get_valid_actions(s) for s in self.raw_states])

          # 네트워크 통과
        action_probs, state_values = self.network(states)
        
        # 마스킹 적용
        masked_probs = action_probs * valid_actions_batch
        masked_probs = masked_probs / (masked_probs.sum(dim=1, keepdim=True) + 1e-8)
        
        # Advantage 계산
        advantages = returns - values.squeeze()
        
        # 정책 손실 계산 (마스킹된 확률 사용)
        dist = Categorical(masked_probs)
        policy_loss = -(dist.log_prob(actions) * advantages.detach()).mean()
        
        # 가치 손실 계산
        # value_loss = F.mse_loss(values.squeeze(), returns)
        value_loss = F.huber_loss(values.squeeze(), returns, delta=1.0)
        
        # 엔트로피 보너스
        entropy_loss = -self.entropy_coef * dist.entropy().mean()
        
        # 전체 손실
        total_loss = policy_loss + self.value_coef * value_loss + entropy_loss
        
        # 역전파 및 최적화
        self.optimizer.zero_grad()

        #학습 불안정할때 넣어주기
        # torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=0.5)

        total_loss.backward()
        self.optimizer.step()
        self.scheduler.step()  # 스케줄러 업데이트
        
        return total_loss.item()
    
    def reset_episode(self):
        self.states.clear()
        self.raw_states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()

    def save(self, episode):
        """모델 저장"""
        torch.save({
            'episode': episode,
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, f"{self.save_dir}/a2c_checkpoint_{episode}.pth")
    
    def load(self, path):
        """모델 로드"""
        checkpoint = torch.load(path)
        self.network.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['episode']

def worker(rank, shared_network, num_episodes,save_interval,return_dict,best_reward,sync_interval):


    torch.cuda.set_device(1)  # cuda:1 명시적 설정
    
    env = make_grid_survivor(show_screen=False)
    state_size = calculate_state_size(env.reset()[0])
    
        # 에이전트 생성 전에 device 설정
    device = torch.device("cuda:1")

    agent = A2CAgent(state_size)
    agent.device = device
    agent.network = shared_network.to(device)
    
    
    episode_rewards = []
    lock = Lock()
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        step = 0
        
        while not done and step < 1200:
            action = agent.act(state)
            next_state, reward, done, _, _ = env.step(action)
            step += 1
            
            reward = agent.calculate_reward(state, next_state, done, step)
            episode_reward += reward
            
            if step >= 1200:
                done = True
                
            loss = agent.update(reward, done)
            state = next_state
        
        # episode_rewards.append(episode_reward)
        

        if rank == 0 and (episode + 1) % save_interval == 0:
            agent.save(episode)
            print(f"Model saved at episode {episode+1}")

        if episode_reward > best_reward.value:
            with lock:  
                best_reward.value = episode_reward
                agent.save('best')
        

        return_dict[f"worker_{rank}_{episode}"] = {
            "reward": episode_reward,
            "step": step,
            "remain_bees": np.sum(next_state['grid'] == 'B'),
            "loss": loss if loss else 0
        }
        if episode % sync_interval == 0:
            agent.network.load_state_dict(shared_network.state_dict())
    
    return episode_rewards

def calculate_state_size(state):
    grid = state['grid']
    N = len(grid)
    grid_info_size = N * N * 6
    direction_size = 4
    hit_points_size = 1
    return grid_info_size + direction_size + hit_points_size

def signal_handler(signum, frame):
    print("\nSignal received. Cleaning up...")
    cleanup_processes()
    sys.exit(0)

def cleanup_processes():
    for p in processes:
        if p.is_alive():
            print(f"Terminating process {p.pid}")
            p.terminate()
            p.join(timeout=3)  # 3초 동안 종료 대기
            if p.is_alive():
                print(f"Force killing process {p.pid}")
                p.kill()  # 강제 종료

if __name__ == "__main__":
    # 시그널 핸들러 등록
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    save_interval = 500
    sync_interval = 10
    processes = []  # 전역 변수로 이동
    
    try:
        mp.set_start_method('spawn')
        
        wandb.init(project="a2c_v1", name="a2c_parallel_Rms_huber")
        
        env = make_grid_survivor(show_screen=False)
        state, _ = env.reset()
        state_size = calculate_state_size(state)
        
        shared_network = A2CNetwork(state_size, 128, 3)
        shared_network.share_memory()
        
        manager = mp.Manager()
        return_dict = manager.dict()
        best_reward = manager.Value('d', 1000)  # 공유 변수로 변경
        logged_keys = set()


        episodes_per_process = 20000 // num_processes
        
        for rank in range(num_processes):
            p = mp.Process(
                target=worker,
                args=(rank, shared_network, episodes_per_process, save_interval,return_dict,best_reward,sync_interval)
            )
            p.start()
            processes.append(p)
        episode_history = collections.deque(maxlen=100) 
        # 메인 프로세스에서 wandb 로깅
        while any(p.is_alive() for p in processes):
            for key, value in return_dict.items():
                if key not in logged_keys:
                    episode_history.append(value["reward"])
                    wandb.log({
                        "reward": value["reward"],
                        "step": value["step"],
                        "remain_bees": value["remain_bees"],
                        "loss": value["loss"],
                        "average_reward": np.mean(episode_history),
                        "best_reward": best_reward.value
                })
                    logged_keys.add(key)
            time.sleep(1)

        for p in processes:
            p.join()
        
        print("Training completed successfully!")
        
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt received. Cleaning up...")
        cleanup_processes()
        sys.exit(1)
    except Exception as e:
        print(f"\nError occurred: {e}")
        cleanup_processes()
        sys.exit(1)
    finally:
        # wandb 종료
        if wandb.run is not None:
            wandb.finish()