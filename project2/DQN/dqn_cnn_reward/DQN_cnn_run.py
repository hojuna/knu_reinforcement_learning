import numpy as np
import torch
import wandb
import os
from knu_rl_env.grid_survivor import make_grid_survivor, evaluate, run_manual, GridSurvivorAgent
from DQN_cnn import DQN
from DQN_memory import ReplayMemory
from DQN_cnn_agent import DQNAgent

def evaluate_agent(agent, env, num_eval_episodes=10, max_steps=1000):
    """
    에이전트를 평가하는 함수입니다.
    
    Parameters:
    - agent (DQNAgent): 평가할 DQN 에이전트.
    - env: GridSurvivor 환경 인스턴스.
    - num_eval_episodes (int): 평가할 에피소드 수.
    - max_steps (int): 각 에피소드에서의 최대 스텝 수.
    
    Returns:
    - avg_reward (float): 평균 보상.
    """
    agent.policy_net.eval()
    total_rewards = []
    
    with torch.no_grad():
        for _ in range(num_eval_episodes):
            state , _ = env.reset()
            pos,dir =agent.extract_agent_info(state['grid'])
            f_obj=agent.find_forward_obj(state['grid'])

            grid, hit_points, direction, forward_obj, agent_position = agent.preprocess_state(
                state['grid'], 
                state['hit_points'], 
                dir, 
                pos, 
                f_obj
            )
            total_reward = 0
            
            for step in range(1, max_steps + 1):
                action = agent.act(grid, hit_points, direction, forward_obj, agent_position)
                next_state, reward, done, _, _  = env.step(action)

                next_pos,next_dir =agent.extract_agent_info(next_state['grid'])
                next_f_obj=agent.find_forward_obj(next_state['grid'])
                next_grid, next_hit_points, next_direction, next_forward_obj, next_agent_position = agent.preprocess_state(
                    next_state['grid'], 
                    next_state['hit_points'], 
                    next_dir, 
                    next_pos, 
                    next_f_obj
                )
                reward = agent.calculate_reward(state, next_state, done, step)
                total_reward += reward
                
                state = next_state
                grid, hit_points, direction, forward_obj, agent_position = next_grid, next_hit_points, next_direction, next_forward_obj, next_agent_position
                
                if done:
                    break
            
            total_rewards.append(total_reward)
    
    agent.policy_net.train()
    avg_reward = np.mean(total_rewards)
    return avg_reward

def train_agent(agent, env, num_episodes=1000, max_steps=1000, checkpoint_path=None, log_interval=100):
    """
    DQN 에이전트를 학습시키는 함수입니다.
    
    Parameters:
    - agent (DQNAgent): 학습할 DQN 에이전트.
    - env: GridSurvivor 환경 인스턴스.
    - num_episodes (int): 학습할 에피소드 수.
    - max_steps (int): 각 에피소드에서의 최대 스텝 수.
    - checkpoint_path (str or None): 체크포인트를 저장할 경로. 지정하지 않으면 저장하지 않음.
    - log_interval (int): 체크포인트 저장 및 평가 주기 (에피소드 단위).
    """
    # WandB 초기화
    wandb.init(
        project="grid_survivor_dqn",
        config={
            "num_episodes": num_episodes,
            "max_steps": max_steps,
            "learning_rate": agent.optimizer.param_groups[0]['lr'],
            "gamma": agent.gamma,
            "batch_size": agent.batch_size,
            "memory_capacity": agent.memory.capacity,
            "target_update_steps": agent.target_update_steps,
            "epsilon_start": agent.epsilon_start,
            "epsilon_end": agent.epsilon_end,
            "epsilon_decay": agent.epsilon_decay,
            "eval_interval": agent.eval_interval
        }
    )

    for episode in range(1, num_episodes + 1):
        state , _ = env.reset()
        pos,dir =agent.extract_agent_info(state['grid'])
        f_obj=agent.find_forward_obj(state['grid'])

        grid, hit_points, direction, forward_obj, agent_position = agent.preprocess_state(
            state['grid'], 
            state['hit_points'], 
            dir, 
            pos, 
            f_obj
        )
        total_reward = 0
        loss_total = 0

        for step in range(1, max_steps + 1):
            # 행동 선택
            action = agent.act(grid, hit_points, direction, forward_obj, agent_position)
            next_state, reward, done, _, _  = env.step(action)

            next_pos,next_dir =agent.extract_agent_info(next_state['grid'])
            next_f_obj=agent.find_forward_obj(next_state['grid'])
            next_grid, next_hit_points, next_direction, next_forward_obj, next_agent_position = agent.preprocess_state(
                next_state['grid'], 
                next_state['hit_points'], 
                next_dir, 
                next_pos, 
                next_f_obj
            )
            
            # 보상 계산
            reward = agent.calculate_reward(state, next_state, done, step)
            total_reward += reward
            
            # 경험 저장
            
            agent.memory.push(
                grid, hit_points, direction, forward_obj, agent_position,  # 현재 상태 정보
                action, reward,  # 행동과 보상
                next_grid, next_hit_points, next_direction, next_forward_obj, next_agent_position,  # 다음 상태 정보
                done  # 종료 여부
            )
           
            
            # 모델 최적화
            loss = agent.optimize_model()
            if loss is not None:
                loss_total += loss
            
            # 상태 업데이트
            state = next_state
            grid, hit_points, direction, forward_obj, agent_position = next_grid, next_hit_points, next_direction, next_forward_obj, next_agent_position
            
            if done:
                remain_bee = np.sum(state['grid'] == 'B')
                break
        
        # 에피소드 종료 후 WandB 로그 기록
        avg_loss = loss_total / step if step > 0 else 0
        wandb.log({
            "Episode": episode,
            "Total Reward": total_reward,
            "Average Loss": avg_loss,
            "Epsilon": agent.epsilon,
            'remain_bee':remain_bee,
            'step': step
        })
        
        print(f"Episode {episode}/{num_episodes}, Total Reward: {total_reward}, Average Loss: {avg_loss:.4f}, Epsilon: {agent.epsilon:.4f}")
        
        # 체크포인트 저장 및 평가
        if  episode % log_interval == 0:
            save_path = f"/home/comoz/main_project/knu_reinforcement_learning/project2/checkpoint8/checkpoint_episode_{episode}.pth"
            agent.save_checkpoint(episode, avg_loss, total_reward, save_path)
            print(f"Checkpoint saved at episode {episode} to {save_path}")
            
            # 평가 (선택사항)
            eval_reward = evaluate_agent(agent, env)
            wandb.log({"Evaluation Reward": eval_reward})
            print(f"Evaluation after episode {episode}: {eval_reward}")
    
    wandb.finish()

# 예시 사용
if __name__ == "__main__":
    # 환경 초기화
    env = make_grid_survivor(False)
    
    # DQNAgent 초기화
    input_channels = 5
    grid_height, grid_width = 35, 35
    num_actions = 3 # 실제 환경에 맞게 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    config = {
        "learning_rate": 5e-5,
        "gamma": 0.99,
        "batch_size": 256,
        "memory_capacity": 100000,
        "target_update_steps": 1000,
        "epsilon_end": 0.01,
        "epsilon_decay": 5000,
        "eval_interval": 1000
    }
    
    agent = DQNAgent(
        input_channels=input_channels, 
        grid_height=grid_height, 
        grid_width=grid_width, 
        num_actions=num_actions, 
        device=device, 
        config=config,
        epsilon_start=1.0
    )

    
    
    # 체크포인트 저장 디렉토리 설정
    # checkpoint_path = 
    # os.makedirs(checkpoint_path, exist_ok=True)
    
    # 학습 파라미터 설정
    num_episodes = 25000
    max_steps = 1200
    log_interval = 1000  # 예: 100 에피소드마다 체크포인트 저장
    
        # 학습 시작
    train_agent(agent, env, num_episodes=num_episodes, max_steps=max_steps, checkpoint_path=None, log_interval=log_interval)
    
    # python /project2/DQN/DQN_cnn/DQN_cnn_run.py