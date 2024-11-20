import numpy as np
import random
from knu_rl_env.grid_adventure import GridAdventureAgent, make_grid_adventure

class GridAdventureRLAgent(GridAdventureAgent):
    def __init__(self, env):
        self.q_table = np.zeros((26, 26, 4))  # (행, 열, 행동 공간)
        self.alpha = 0.1  # 학습률
        self.gamma = 0.9  # 할인 계수
        self.epsilon = 1.0  # 탐험 비율
        self.epsilon_min = 0.01  # 최소 탐험 비율
        self.epsilon_decay = 0.995  # 탐험 비율 감소
        self.env = env

    def act(self, state):
        row, col = state  # 에이전트의 현재 위치
        if np.random.rand() <= self.epsilon:
            # 무작위 행동 선택 (탐험)
            return random.choice([self.ACTION_LEFT, self.ACTION_RIGHT, self.ACTION_FORWARD, self.ACTION_PICKUP, self.ACTION_DROP, self.ACTION_UNLOCK])
        else:
            # Q-value가 가장 높은 행동 선택 (이용)
            return np.argmax(self.q_table[row, col])

    def update_q_value(self, state, action, reward, next_state):
        row, col = state
        next_row, next_col = next_state
        best_next_action = np.argmax(self.q_table[next_row, next_col])
        td_target = reward + self.gamma * self.q_table[next_row, next_col, best_next_action]
        td_error = td_target - self.q_table[row, col, action]
        self.q_table[row, col, action] += self.alpha * td_error

    def train(self, episodes):
        for e in range(episodes):
            state = self.env.reset()
            done = False
            while not done:
                action = self.act(state)
                next_state, reward, done, _ = self.env.step(action)
                self.update_q_value(state, action, reward, next_state)
                state = next_state

            # epsilon 감소
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

if __name__ == "__main__":
    env = make_grid_adventure(show_screen=False)
    agent = GridAdventureRLAgent(env)
    agent.train(1000)  # 1000번의 에피소드 훈련
