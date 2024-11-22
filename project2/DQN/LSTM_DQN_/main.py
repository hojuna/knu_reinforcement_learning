import torch
import torch.nn as nn
import torch.optim as optim 
import torch.nn.functional as F
import numpy as np
import math
import wandb
from collections import namedtuple
import random

from knu_rl_env.grid_survivor import make_grid_survivor
from DQN_agnet import DuelingDQNAgent
from DQN_network import DuelingDQN
from memory import PrioritizedReplayMemory, SequenceTransition

def main():
    # WandB 초기화
    wandb.init(
        project="dqn_parallel",
        name="dqn_single",
        config={
            "learning_rate": 1e-4,
            "gamma": 0.95,
            "batch_size": 128,
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
            "sequence_length": 10,
            "epsilon": 0.01
        }
    )

    config = dict(wandb.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay = 5000

    try:
        for episode in range(1, config['num_episodes'] + 1):
            state, _ = env.reset()
            state_input = agent.preprocess_state(state)
            
            done = False
            total_reward = 0
            loss_value = None

            agent.reset_visit_table()
            
            step = 0
            hidden = None
            sequence = []

            while not done:
                epsilon = epsilon_end + (epsilon_start - epsilon_end) * math.exp(-1. * agent.steps_done / epsilon_decay)
                agent.steps_done += 1
                agent.epsilon = epsilon

                action, hidden = agent.act(state_input, epsilon, hidden)
                next_state, reward, done, _, _ = env.step(action)
                next_state_input = agent.preprocess_state(next_state)

                if step >= 1200:
                    done = True

                movement_reward = agent.calculate_reward(state, next_state, done, step)
                total_reward += movement_reward

                movement_reward += agent.visit_table_update(state, next_state)

                sequence.append((
                    state_input,
                    action,
                    movement_reward,
                    next_state_input,
                    done
                ))

                if len(sequence) >= config['sequence_length'] or done:
                    if sequence:
                        states, actions, rewards, next_states, dones = zip(*sequence)
                        sequence_transition = SequenceTransition(
                            states=list(states),
                            actions=list(actions),
                            rewards=list(rewards),
                            next_states=list(next_states),
                            dones=list(dones)
                        )
                        agent.memory.push(sequence_transition)
                    sequence = []

                state = next_state
                state_input = next_state_input
                step += 1

                loss = agent.optimize_model()
                if loss is not None:
                    loss_value = loss

            if episode % config['target_update'] == 0:
                agent.target_net.load_state_dict(agent.policy_net.state_dict())

            if episode % config['checkpoint_interval'] == 0:
                torch.save(agent.policy_net.state_dict(), f'/home/comoz/main_project/knu_reinforcement_learning/project2/DQN/LSTM_DQN_/saved_model/dueling_dqn_single_checkpoint_{episode}.pth')

            remain_bees = np.sum(next_state['grid'] == 'B')
            visit_min = agent.visit_table.min()
            
            print(f"Episode {episode}: Total Reward: {total_reward:.4f}, Loss: {loss_value}, Epsilon: {epsilon:.4f}, Steps: {step}, v_min: {visit_min}, remain_bees: {remain_bees}")
            
            wandb.log({
                'episode': episode,
                'total_reward': total_reward,
                'loss': loss_value,
                'epsilon': epsilon,
                'steps': step,
                'v_min': visit_min,
                'remain_bees': remain_bees
            })

    except KeyboardInterrupt:
        print("Training interrupted. Saving model...")
        torch.save(agent.policy_net.state_dict(), '/home/comoz/main_project/knu_reinforcement_learning/project2/DQN/LSTM_DQN_/saved_model/dueling_dqn_single_interrupt.pth')
    finally:
        env.close()
        wandb.finish()

if __name__ == "__main__":
    main()