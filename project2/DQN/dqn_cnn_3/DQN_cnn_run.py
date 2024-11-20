import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import numpy as np
import wandb
from tqdm import tqdm

from knu_rl_env.grid_survivor import GridSurvivorAgent, make_grid_survivor, evaluate
from DQN_cnn_agent import GridSurvivorRLAgent

CHECKPOINT_DIR = "/home/comoz/main_project/knu_reinforcement_learning/project2/checkpoint6"

def main():
    env = make_grid_survivor(False)
    agent = GridSurvivorRLAgent()
    
    config = {
        'batch_size': agent.batch_size,
        'num_episodes': 10000,
        'save_interval': 500,
        'target_update': agent.target_update,
        'learning_rate': agent.optimizer.param_groups[0]['lr'],
        'gamma': agent.gamma,
        'epsilon_start': agent.epsilon_start,
        'epsilon_end': agent.epsilon_end,
        'epsilon_decay': agent.epsilon_decay,
        'memory_size': agent.memory.capacity
    }
    
    wandb.init(
        project="grid-survivor",
        config=config,
        name=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        dir=CHECKPOINT_DIR
    )
    
    initial_bees = 50

    for episode in range(config['num_episodes']):
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        # agent.update_epsilon()

        while True:
            action = agent.act(state)
            next_state, reward, done, info, _ = env.step(action)
            
            reward = agent.calculate_reward(state, next_state, done)
            agent.memory.push(state, action, next_state, reward, done)
            
            loss = agent.optimize_model()
            episode_reward += reward
            episode_length += 1
            
            if episode_length % agent.target_update == 0:
                agent.update_target_network()
            
            state = next_state

            final_bees = np.sum(state['grid'] == 'B')
            bees_saved = initial_bees - final_bees
            
            if done:
                print(f"Episode {episode}, Evaluation Average Reward: {episode_reward:.2f}, Bees Saved: {bees_saved}, Epsilon: {agent.epsilon_current:.4f}")
                break


        
        wandb.log({
            'episode': episode,
            'reward': episode_reward,
            'length': episode_length,
            'epsilon': agent.epsilon_current,
            'loss': loss if loss is not None else 0,
            'bees_saved': bees_saved
        })
        
        if episode % config['save_interval'] == 0:
            save_path = os.path.join(CHECKPOINT_DIR, f"model_episode_{episode}.pth")
            agent.save_model(save_path)
        
        # if episode % agent.eval_interval == 0:
        #     avg_reward = agent.evaluate(env)
        #     print(f"Episode {episode}, Evaluation Average Reward: {avg_reward:.2f}, Bees Saved: {bees_saved}, Epsilon: {agent.epsilon_current:.4f}")
        #     wandb.log({'eval_reward': avg_reward})
    
    env.close()
    wandb.finish()

if __name__ == "__main__":
    main()