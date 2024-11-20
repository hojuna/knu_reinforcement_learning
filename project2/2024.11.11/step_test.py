from knu_rl_env.grid_survivor import GridSurvivorAgent, make_grid_survivor, evaluate

import numpy as np
'''
knu_rl_env.grid_adventure.GridAdventureAgent을 상속해서
자신만의 에이전트를 구현하세요.
'''
class GridSurvivorRLAgent(GridSurvivorAgent):
    def act(self, state):
        '''
        다음 중 하나를 반환해야 합니다:
        - GridSurvivorAgent.ACTION_LEFT
        - GridSurvivorAgent.ACTION_RIGHT
        - GridSurvivorAgent.ACTION_FORWARD
        '''
        
        return np.random.choice(3)

'''
에이전트를 훈련하는 코드를 구현하세요.
'''
def train():
    '''
    Grid Survivor 환경은 다음과 같이 생성할 수 있습니다.
    '''
    env = make_grid_survivor(
        show_screen=False # or, False
    )
    env.step_count=0

    obs, reward, terminated, truncated, info =env.step(1)

    print(*obs)


if __name__=='__main__':

    env = make_grid_survivor(
        show_screen=False # or, False
    )
    env.reset()
    dir=env.step(0)
    print(dir)