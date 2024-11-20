from knu_rl_env.grid_adventure import GridAdventureAgent, make_grid_adventure, evaluate, run_manual


'''
Implement your agent by overriding knu_rl_env.grid_adventure.GridAdventureAgent
'''
class GridAdventureRLAgent(GridAdventureAgent):
    def act(self, state):
        '''
        Return value is one of actions following:
        - GridAdventureAgent.ACTION_LEFT
        - GridAdventureAgent.ACTION_RIGHT
        - GridAdventureAgent.ACTION_FORWARD
        - GridAdventureAgent.ACTION_PICKUP
        - GridAdventureAgent.ACTION_DROP
        - GridAdventureAgent.ACTION_UNLOCK
        '''
        pass

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


if __name__ == '__main__':
    agent = '''Specify your learned agent'''
    evaluate(agent)