import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import math
from collections import namedtuple
import random

import torch.multiprocessing as mp
from multiprocessing import Manager, Lock
import signal
import sys
import time  # 추가된 라이브러리

from knu_rl_env.grid_survivor import make_grid_survivor, GridSurvivorAgent  # 환경 임포트

from DQN_agnet import DuelingDQNAgent
from DQN_network import DuelingDQN
from memory import PrioritizedReplayMemory, SequenceTransition



# 멀티프로세싱을 위한 worker 함수 정의
def worker(rank, config, shared_policy_net, shared_target_net, lock, device, stop_event, return_dict):
    # 환경 생성
    env = make_grid_survivor(show_screen=False)

    # 에이전트 초기화
    agent = DuelingDQNAgent(
        input_channels=config['num_channels'],
        grid_height=config['grid_height'],
        grid_width=config['grid_width'],
        num_actions=config['num_actions'],
        device=device,
        config=config
    )

    # 공유된 네트워크 로드
    agent.policy_net.load_state_dict(shared_policy_net.state_dict())
    agent.target_net.load_state_dict(shared_target_net.state_dict())

    # 옵티마이저 초기화
    agent.optimizer = optim.Adam(agent.policy_net.parameters(), lr=config['learning_rate'])

    num_episodes = config['num_episodes_per_process']
    target_update = config['target_update']

    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay = 5000

    try:
        for episode in range(1, num_episodes + 1):
            if stop_event.is_set():
                print(f"Process {rank} received stop event. Exiting.")
                break

            state, _ = env.reset()

            # 상태 전처리
            state_input = agent.preprocess_state(state)

            done = False
            total_reward = 0
            loss_value = None

            agent.reset_visit_table()

            step = 0
            hidden = None  # 에피소드 시작 시 hidden state 초기화
            sequence = []
            sequence_length = config['sequence_length']

            while not done:
                if stop_event.is_set():
                    print(f"Process {rank} received stop event during episode. Exiting.")
                    break

                # ε 값 계산
                epsilon = epsilon_end + (epsilon_start - epsilon_end) * math.exp(-1. * agent.steps_done / epsilon_decay)
                agent.steps_done += 1
                agent.epsilon = epsilon

                # 행동 선택
                action, hidden = agent.act(state_input, epsilon, hidden)

                # 행동 수행
                next_state, reward, done, _, _ = env.step(action)
                next_state_input = agent.preprocess_state(next_state)


                # if agent.visit_table.min() < -100:
                #     done = True

                if step >= 1200:
                    done = True

                # 보상 계산
                movement_reward = agent.calculate_reward(state, next_state, done, step)
                total_reward += movement_reward

                movement_reward += agent.visit_table_update(state, next_state)

                # 시퀀스에 현재 transition 추가
                sequence.append((
                    state_input,
                    action,
                    movement_reward,
                    next_state_input,
                    done
                ))

                # 시퀀스가 충분히 쌓였거나 에피소드가 끝났을 때
                if len(sequence) >= sequence_length or done:
                    if sequence:
                        # 시퀀스 데이터 분리
                        states, actions, rewards, next_states, dones = zip(*sequence)
                        
                        # SequenceTransition 객체 생성
                        sequence_transition = SequenceTransition(
                            states=list(states),  # 리스트로 변환
                            actions=list(actions),
                            rewards=list(rewards),
                            next_states=list(next_states),
                            dones=list(dones)
                        )
                        
                        # 메모리에 저장
                        with lock:
                            agent.memory.push(sequence_transition)
                    
                    # 시퀀스 초기화
                    sequence = []

                # 다음 상태로 업데이트
                state = next_state
                state_input = next_state_input
                step += 1

                # 모델 최적화
                loss = agent.optimize_model()
                if loss is not None:
                    loss_value = loss

                # 주기적으로 공유된 네트워크와 동기화
                if episode % config['sync_interval'] == 0:
                    # 로컬 네트워크의 가중치를 글로벌 네트워크에 복사
                    with lock:
                        shared_policy_net.load_state_dict(agent.policy_net.state_dict())
                        shared_target_net.load_state_dict(agent.target_net.state_dict())
                        # 글로벌 네트워크의 가중치를 로컬 네트워크에 복사
                        agent.policy_net.load_state_dict(shared_policy_net.state_dict())
                        agent.target_net.load_state_dict(shared_target_net.state_dict())

            if episode % config['checkpoint_interval'] == 0:
                agent.save_model(f'/home/comoz/main_project/knu_reinforcement_learning/project2/DQN/LSTM_DQN_/saved_model/dueling_dqn_checkpoint_{episode}.pth')

            remain_bees = np.sum(next_state['grid'] == 'B')
            visit_min = agent.visit_table.min()
            # 에피소드 로그 출력
            print(f"Process {rank}, Episode {episode}: Total Reward: {total_reward:.4f}, Loss: {loss_value}, Epsilon: {epsilon:.4f}, Steps: {step}, v_min: {visit_min}, remain_bees: {remain_bees}")
            return_dict[rank] = {
                'episode': episode,
                'total_reward': total_reward,
                'loss': loss_value,
                'epsilon': epsilon,
                'steps': step,
                'v_min': visit_min,
                'remain_bees': remain_bees
            }

    except KeyboardInterrupt:
        print(f"Process {rank} received KeyboardInterrupt. Exiting.")
    finally:
        env.close()

# 메인 함수 정의
def main():
    import wandb

    # WandB 초기화
    wandb.init(
        project="dqn_parallel",
        name="dqn_parallel_run",
        config={
            "learning_rate": 1e-4,
            "gamma": 0.95,
            "batch_size": 64,
            "memory_capacity": 100000,
            "alpha": 0.6,
            "beta_start": 0.4,
            "beta_frames": 100000,
            "num_channels": 5,
            "grid_height": 34,
            "grid_width": 34,
            "num_actions": 3,
            "target_update": 100,
            "checkpoint_interval": 1000,
            "num_episodes": 40000,
            "num_processes": 1,
            "num_episodes_per_process": 20000,
            "sync_interval": 10,
            "epsilon": 0.01,
            "sequence_length": 10
        }
    )

    config = dict(wandb.config)

    mp.set_start_method('spawn', force=True)

    manager = mp.Manager()
    return_dict = manager.dict()

    # 공유된 네트워크 초기화
    shared_policy_net = DuelingDQN(
        input_channels=config['num_channels'],
        grid_height=config['grid_height'],
        grid_width=config['grid_width'],
        num_actions=config['num_actions']
    )
    shared_policy_net.share_memory()

    shared_target_net = DuelingDQN(
        input_channels=config['num_channels'],
        grid_height=config['grid_height'],
        grid_width=config['grid_width'],
        num_actions=config['num_actions']
    )
    shared_target_net.load_state_dict(shared_policy_net.state_dict())
    shared_target_net.share_memory()

    lock = Lock()

    # 안전 종료를 위한 이벤트 객체 생성
    stop_event = mp.Event()

    processes = []

    def signal_handler(sig, frame):
        print(f"Main process received signal {sig}. Initiating shutdown...")
        torch.save(shared_policy_net.state_dict(), f'/home/comoz/main_project/knu_reinforcement_learning/project2/DQN/LSTM_DQN_/saved_model/dueling_dqn_interrupt.pth')
        stop_event.set()
        for p in processes:
            if p.is_alive():
                p.join(timeout=5)
        wandb.finish()
        sys.exit(0)

    # 시그널 핸들러 등록
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        for rank in range(config['num_processes']):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            p = mp.Process(
                target=worker, 
                args=(
                    rank,                  # rank
                    config,                # config
                    shared_policy_net,     # shared_policy_net
                    shared_target_net,     # shared_target_net
                    lock,                  # lock
                    device,                # device
                    stop_event,            # stop_event
                    return_dict            # return_dict
                )
            )
            p.start()
            processes.append(p)

        while any(p.is_alive() for p in processes):
            for rank in range(config['num_processes']):
                if rank in return_dict:
                    log_data = return_dict.pop(rank)
                    wandb.log({
                        'episode': log_data['episode'],
                        'total_reward': log_data['total_reward'],
                        'loss': log_data['loss'],
                        'epsilon': log_data['epsilon'],
                        'steps': log_data['steps'],
                        'v_min': log_data['v_min'],
                        'remain_bees': log_data['remain_bees']
                    })
            time.sleep(1)

        print("Training completed successfully!")

    except KeyboardInterrupt:
        print("Main process received KeyboardInterrupt. Initiating shutdown...")
        stop_event.set()
        for p in processes:
            if p.is_alive():
                p.join(timeout=5)
    except Exception as e:
        print(f"An error occurred: {e}")
        stop_event.set()
        for p in processes:
            if p.is_alive():
                p.join(timeout=5)
    finally:
        for p in processes:
            if p.is_alive():
                p.terminate()
        wandb.finish()
        print("All processes have been terminated. Exiting program.")

if __name__ == "__main__":
    main()