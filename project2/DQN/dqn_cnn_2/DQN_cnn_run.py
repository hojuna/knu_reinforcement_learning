import torch
import numpy as np
from knu_rl_env.grid_survivor import make_grid_survivor
import wandb
import time
from tqdm import tqdm
import sys
import os

# 현재 디렉토리의 모듈들을 import
from DQN_cnn_agent import DQNAgent
from DQN_memory import ReplayMemory
from DQN_network import DQN

# 프로젝트 루트 디렉토리 추가 (필요한 경우)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def train():
    # 체크포인트 저장 경로 설정
    save_dir = '/home/comoz/main_project/knu_reinforcement_learning/project2/checkpoint8'
    # os.makedirs(save_dir, exist_ok=True)  # 디렉토리가 없으면 생성
    
    # wandb 초기화
    wandb.init(
        project="grid-survivor",
        config={
            "architecture": "DQN-PER",
            "learning_rate": 0.0001,
            "batch_size": 128,
            "gamma": 0.99,
            "memory_size": 100000,
            "target_update": 1000,
            "epsilon_start": 1.0,
            "epsilon_end": 0.05,
            "epsilon_decay": 200000,
        }
    )

    # 환경 및 에이전트 초기화
    env = make_grid_survivor(show_screen=False)
    agent = DQNAgent(input_size=35,device='cuda')
    
    # 학습 파라미터
    num_episodes = 100000
    print_interval = 100
    save_interval = 1000
    eval_interval = 200
    
    # 메트릭 추적
    best_reward = float('-inf')
    episode_rewards = []
    episode_lengths = []
    episode_bees_saved = []
    
    for episode in tqdm(range(num_episodes)):
        state,_ = env.reset()
        episode_reward = 0
        episode_length = 0
        initial_bees = np.sum(state['grid'] == 'B')
        
        while True:
            # 행동 선택 및 실행
            action = agent.act(state)
            next_state, reward, done, _ ,_= env.step(action)
            
            # 보상 계산 및 경험 저장
            reward = agent.calculate_reward(state, next_state, done)
            agent.save_transition(state, action, next_state, reward, done)
            
            # 모델 최적화
            loss = agent.optimize_model()
            
            # 타겟 네트워크 업데이트
            if agent.steps_done % agent.target_update == 0:
                agent.update_target_network()
            
            # 상태 업데이트
            state = next_state
            episode_reward += reward
            episode_length += 1
            
            if done:
                break
        
        # 에피소드 메트릭 저장
        final_bees = np.sum(state['grid'] == 'B')
        bees_saved = initial_bees - final_bees
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        episode_bees_saved.append(bees_saved)
        
        # wandb 로깅
        wandb.log({
            "episode": episode,
            "reward": episode_reward,
            "length": episode_length,
            "bees_saved": bees_saved,
            "epsilon": agent.epsilon_end + (agent.epsilon_start - agent.epsilon_end) * \
                      np.exp(-1. * agent.steps_done / agent.epsilon_decay),
            "loss": loss if loss is not None else 0,
            "memory_size": len(agent.memory),
            "rare_experiences": len(agent.memory.rare_buffer)
        })
        
        # 주기적인 평가 및 저장
        if episode % eval_interval == 0:
            eval_reward = evaluate_agent(agent, env, num_episodes=5)
            wandb.log({"eval_reward": eval_reward})
            
            if eval_reward > best_reward:
                best_reward = eval_reward
                torch.save({
                    'episode': episode,
                    'model_state_dict': agent.policy_net.state_dict(),
                    'optimizer_state_dict': agent.optimizer.state_dict(),
                    'best_reward': best_reward,
                }, f'{save_dir}/best_model.pth')
        
        # 주기적인 모델 저장
        if episode % save_interval == 0:
            torch.save({
                'episode': episode,
                'model_state_dict': agent.policy_net.state_dict(),
                'optimizer_state_dict': agent.optimizer.state_dict(),
                'reward': episode_reward,
            }, f'{save_dir}/checkpoint_{episode}.pth')
        
        # 진행상황 출력
        if episode % print_interval == 0:
            avg_reward = np.mean(episode_rewards[-print_interval:])
            avg_length = np.mean(episode_lengths[-print_interval:])
            avg_bees = np.mean(episode_bees_saved[-print_interval:])
            print(f'Episode {episode}')
            print(f'Average Reward: {avg_reward:.2f}')
            print(f'Average Length: {avg_length:.2f}')
            print(f'Average Bees Saved: {avg_bees:.2f}')
            print(f'Memory Size: {len(agent.memory)}')
            print(f'Rare Experiences: {len(agent.memory.rare_buffer)}')
            print('-------------------')

def evaluate_agent(agent, env, num_episodes=5):
    """에이전트 평가"""
    agent.training = False
    total_rewards = []
    
    for _ in range(num_episodes):
        state,_ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action = agent.act(state)
            next_state, reward, done, _,_ = env.step(action)
            episode_reward += reward
            state = next_state
            
        total_rewards.append(episode_reward)
    
    agent.training = True
    return np.mean(total_rewards)

if __name__ == '__main__':
    train()
    