import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from collections import deque
import random
from knu_rl_env.grid_survivor import GridSurvivorAgent, make_grid_survivor
import wandb


from A2Cagent import A2CAgent



def train(env, agent, num_episodes, save_interval=100):
    """학습 함수"""
    best_reward = float('-inf')
    reward_history = []

    wandb.init(project="a2c_v1", name="a2c_v2_Rms_huber_v10")
    config = wandb.config
    config.num_episodes = num_episodes
    config.save_interval = save_interval

    for episode in range(num_episodes):
        state,_ = env.reset()
        episode_reward = 0
        done = False
        step=0
        max_step = 1200
        agent.visit_table_reset()

        # 현재 에피소드 시작 전 모델 상태 저장
        current_model_state = {
            k: v.cpu().clone() for k, v in agent.network.state_dict().items()
        }
        current_optimizer_state = {
            k: v.cpu().clone() if isinstance(v, torch.Tensor) else v 
            for k, v in agent.optimizer.state_dict().items()
        }

        while not done and step<max_step:
            # 행동 선택 및 환경과 상호작용
            action = agent.train_act(state)
            next_state, reward, done, _ , _= env.step(action)
            step+=1
            
            if step >= max_step:
                done=True

            pos = agent.extract_agent_info(state['grid'])[0]
            if agent.visit_table.min() < -150:
                done=True

            # 보상 계산 및 업데이트
            reward = agent.calculate_reward(state, next_state, done,step,action)
            episode_reward += reward

            reward += agent.visit_table_update(state, next_state)


                
            loss = agent.update(reward, done)
            state = next_state
        
        # 성과 기록
        reward_history.append(episode_reward)
        
        # 에피소드가 끝난 후 성능 체크
        if episode_reward > best_reward:
            best_reward = episode_reward
            # 이전에 저장해둔 모델 상태를 베스트 모델로 저장
            torch.save({
                'episode': episode,
                'model_state_dict': current_model_state,
                'optimizer_state_dict': current_optimizer_state,
                'reward': episode_reward
            }, f"{agent.save_dir}/a2c_best_model.pth")

        # 일반적인 체크포인트 저장
        if episode % save_interval == 0:
            agent.save(episode)
        
        # # 최고 성능 모델 저장
        # if episode_reward > best_reward:
        #     best_reward = episode_reward
        #     agent.save('best')

        remain_bees = np.sum(next_state['grid'] == 'B')
        visit_table_min = agent.visit_table.min()

        wandb.log({"episode_reward": episode_reward, "best_reward": best_reward,"step":step,"episode":episode,"remain_bees":remain_bees,"loss":loss,"visit_table_min":visit_table_min})
        
        # 진행상황 출력
        if episode % 10 == 0:
            avg_reward = np.mean(reward_history[-10:])
            print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}, Best Reward: {best_reward:.2f}, remain_bees: {remain_bees}, step: {step}, visit_table_min: {visit_table_min}")
    
    return reward_history

def calculate_state_size(state):
    grid = state['grid']
    N = len(grid)
    grid_info_size = N * N * 6  # 6가지 셀 타입
    direction_size = 4          # 4가지 방향
    hit_points_size = 1        # 체력 정보
    visit_table_size = 1
    
    return grid_info_size + direction_size + hit_points_size + visit_table_size

if __name__ == "__main__":
    # 환경 생성
    env = make_grid_survivor(show_screen=False)
    state,_ = env.reset()
    
    # 임시 테스트용 state_size (실제 환경에 맞게 수정 필요)
    # 에이전트 생성
    agent = A2CAgent(calculate_state_size(state))
    
    # 학습 설정
    num_episodes = 20000
    save_interval = 10000
    
    # 학습 실행
    reward_history = train(env, agent, num_episodes, save_interval)
    
    # 최종 모델 저장
    agent.save('final')
    
    print("Training completed!")