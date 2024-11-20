from knu_rl_env.grid_adventure import GridAdventureAgent, make_grid_adventure, evaluate, run_manual
import numpy as np
from itertools import product
'''
Implement your agent by overriding knu_rl_env.grid_adventure.GridAdventureAgent
'''
class GridAdventureRLAgent(GridAdventureAgent):
    def __init__(self, desc: tuple, phi: float = 1e-3, n_steps: int = 1000, gamma=0.99, init_V: float = 0):
        self.desc = desc
        self.phi = phi
        self.n_steps = n_steps
        self.gamma = gamma
        self.init_V = init_V

        self.rows, self.cols = self.desc.shape
        self.V = None
        self.PI = None
 
    def act(self, state):
        row, col = state // self.rows, state % self.cols
        return self.PI[row, col].astype(int)
'''
Implement how to train your agent
'''
def train():
    '''
    Below is to create the grid adventure environment.
    '''
    env = make_grid_adventure(
        show_screen=True # or, False
    )
    '''
    And your training code might be followed.
    '''
    state,_ = env.reset()
    agent = GridAdventureRLAgent(desc=state).fit()
    return agent

if __name__ == '__main__':
    # agent = train()
    run_manual()
