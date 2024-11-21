import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import math
from collections import namedtuple
import random

from knu_rl_env.grid_survivor import make_grid_survivor, GridSurvivorAgent  # 환경 임포트


from DQN_agnet import DuelingDQNAgent
from DQN_network import DuelingDQN
from memory import PrioritizedReplayMemory



def main():
    import wandb  # wandb 사용 시

    # WandB 초기화
    wandb.init(
        project="dqn_2024_111_21",
        name="dqn_2024_111_21",
        config={
            "learning_rate": 5e-4,
            "gamma": 0.95,
            "batch_size": 64,
            "memory_capacity": 100000,
            "alpha": 0.6,          # Prioritized Experience Replay의 alpha
            "beta_start": 0.4,     # 초기 beta 값
            "beta_frames": 100000, # beta가 1에 도달하는 프레임 수
            "num_channels": 5,
            "grid_height": 35,
            "grid_width": 35,
            "num_actions": 3,
            "target_update": 100,
            "checkpoint_interval": 1000,
            "num_episodes": 40000
        }
    )

    config = wandb.config

    # 환경 초기화
    env = make_grid_survivor(show_screen=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 초기 상태 가져오기
    state, _ = env.reset()
    grid_shape = state['grid'].shape
    num_channels = config.num_channels
    num_actions = config.num_actions

    # 에이전트 초기화
    agent = DuelingDQNAgent(
        input_channels=num_channels,
        grid_height=grid_shape[0],
        grid_width=grid_shape[1],
        num_actions=num_actions,
        device=device,
        config=config
    )

    num_episodes = config.num_episodes
    target_update = config.target_update
    checkpoint_interval = config.checkpoint_interval

    # 손실과 보상을 저장할 리스트
    loss_history = []
    reward_history = []

    # ε-그리디 탐사 파라미터
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay = 20000

    for episode in range(1, num_episodes + 1):
        state, _ = env.reset()
        grid = state['grid']
        hit_points = state['hit_points']

        # 상태 전처리
        state_input = agent.preprocess_state(state)

        done = False
        total_reward = 0
        loss_value = None

        agent.reset_visit_table()

        step = 0

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


            if agent.visit_table.min() < -100:
                done = True
        
            if step >= 1200:
                done = True

            # 보상 계산
            movement_reward = agent.calculate_reward(state, next_state, done, agent.steps_done)
            total_reward += movement_reward

            step += 1

            # 상태 전처리
            next_state_input = agent.preprocess_state(next_state)


            # 경험 저장
            agent.memory.push(state_input, action, movement_reward, next_state_input, done)

            # 상태 업데이트
            state_input = next_state_input
            state = next_state

            # 모델 최적화
            loss = agent.optimize_model()
            if loss is not None:
                loss_value = loss

        # 타겟 네트워크 업데이트
        if episode % target_update == 0:
            agent.update_target_network()

        # 체크포인트 저장
        if episode % checkpoint_interval == 0:
            checkpoint_path = f'/home/comoz/main_project/knu_reinforcement_learning/project2/DQN/last_DQN/saved_model/dueling_dqn_checkpoint_episode_{episode}.pth'
            agent.save_checkpoint(episode, loss_value, total_reward, checkpoint_path)
            wandb.save(checkpoint_path)

        # 로그 기록
        loss_history.append(loss_value)
        reward_history.append(total_reward)
        remain_bees = np.sum(next_state['grid'] == 'B')
        v_min= agent.visit_table.min()
        wandb.log({
            "Episode": episode,
            "Total Reward": total_reward,
            "Loss": loss_value,
            "Epsilon": epsilon,
            "steps_done": step,
            "remain_bees": remain_bees,
            "visit_table": v_min
        })


        print(f"Episode {episode}: Total Reward: {total_reward:.4f}, Loss: {loss_value:.4f}, Epsilon: {epsilon:.4f}, steps_: {step}, remain_bees: {remain_bees}, visit_table: {v_min}")

    # 최종 모델 저장
    torch.save(agent.policy_net.state_dict(), f'/home/comoz/main_project/knu_reinforcement_learning/project2/DQN/last_DQN/saved_model/dueling_dqn_grid_survivor_final.pth')
    wandb.finish()

# 9. 모델 테스트 함수 정의 (생략 - 이전 코드와 동일)

# 10. 실행 부분
if __name__ == "__main__":
    main()
    # 학습된 모델을 테스트하려면 아래 주석을 해제하세요.
    # run_trained_model("project2/checkpoint/dueling_dqn_checkpoint_episode_11000.pth", num_episodes=10)