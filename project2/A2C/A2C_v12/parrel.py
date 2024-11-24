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
from A2Cagent import A2CAgent
from A2CNetwork import A2CNetwork   

def worker(rank, shared_network, num_episodes, save_interval, return_dict, best_reward, sync_interval):
    # GPU 설정
    torch.cuda.set_device(1)  # cuda:1 명시적 설정
    
    env = make_grid_survivor(show_screen=False)
    state_size = calculate_state_size(env.reset()[0])
    
    device = torch.device("cuda:1")
    agent = A2CAgent(state_size)
    agent.device = device
    agent.network = shared_network.to(device)
    
    
    # best_reward = float('-inf')
    episode_rewards = []
    lock = Lock()
    max_step = 1200

    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        step = 0
        agent.visit_table_reset()

                # 현재 에피소드 시작 전 모델 상태 저장
        current_model_state = {
            k: v.cpu().clone() for k, v in agent.network.state_dict().items()
        }
        current_optimizer_state = {
            k: v.cpu().clone() if isinstance(v, torch.Tensor) else v 
            for k, v in agent.optimizer.state_dict().items()
        }

        
        while not done and step < max_step:
            action = agent.train_act(state)
            next_state, reward, done, _, _ = env.step(action)
            step += 1

            if agent.visit_table.min() < -150:
                done = True
                
            if step >= max_step:
                done = True

            reward = agent.calculate_reward(state, next_state, done, step, action)
            episode_reward += reward

            reward += agent.visit_table_update(state, next_state)
            
            loss = agent.update(reward, done)
            state = next_state
        
        # episode_rewards.append(episode_reward)
        

        if rank == 0 and (episode + 1) % save_interval == 0:
            agent.save(episode)
            print(f"Model saved at episode {episode+1}")

        if episode_reward > best_reward:
            best_reward = episode_reward
            # 이전에 저장해둔 모델 상태를 베스트 모델로 저장
            torch.save({
                'episode': episode,
                'model_state_dict': current_model_state,
                'optimizer_state_dict': current_optimizer_state,
                'reward': episode_reward
            }, f"{agent.save_dir}/a2c_best_model.pth")

        

        return_dict[f"worker_{rank}_{episode}"] = {
            "reward": episode_reward,
            "step": step,
            "remain_bees": np.sum(next_state['grid'] == 'B'),
            "loss": loss if loss else 0,
            "visit_table": agent.visit_table.min(),
        }
        if episode % sync_interval == 0:
            with lock:
                agent.network.load_state_dict(shared_network.state_dict())
        
        episode_rewards.append(episode_reward)
    
    return episode_rewards

def calculate_state_size(state):
    grid = state['grid']
    N = len(grid)
    grid_info_size = N * N * 6  # 6가지 셀 타입
    direction_size = 4          # 4가지 방향
    hit_points_size = 1        # 체력 정보
    visit_table_size = 1
    
    return grid_info_size + direction_size + hit_points_size + visit_table_size

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
    num_processes = 6
    # 시그널 핸들러 등록
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    save_interval = 500
    sync_interval = 10
    processes = []  # 전역 변수로 이동
    
    try:
        mp.set_start_method('spawn')
        
        wandb.init(project="a2c_v1", name="a2c_v12")
        
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
                args=(rank, shared_network, episodes_per_process, 
                      save_interval, return_dict, best_reward, sync_interval)
            )
            p.start()
            processes.append(p)

        # 메인 프로세스에서 wandb 로
        episode_history = collections.deque(maxlen=100)
        
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
                        "best_reward": best_reward.value,
                        "visit_table_min": value["visit_table"]
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