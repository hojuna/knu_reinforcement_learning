import numpy as np
import pickle  # For saving and loading the Q-table
from collections import deque
from knu_rl_env.grid_adventure import GridAdventureAgent, make_grid_adventure, evaluate

class GridAdventureRLAgent(GridAdventureAgent):
    def __init__(self, load_q_table=False, q_table_filename='q_table.pkl'):
        # Define possible actions (primitive actions)
        self.ACTION_LEFT = GridAdventureAgent.ACTION_LEFT
        self.ACTION_RIGHT = GridAdventureAgent.ACTION_RIGHT
        self.ACTION_FORWARD = GridAdventureAgent.ACTION_FORWARD
        self.ACTION_PICKUP = GridAdventureAgent.ACTION_PICKUP
        self.ACTION_DROP = GridAdventureAgent.ACTION_DROP
        self.ACTION_UNLOCK = GridAdventureAgent.ACTION_UNLOCK

        # Define composite actions
        self.knu_agent_left = [self.ACTION_LEFT, self.ACTION_FORWARD]
        self.knu_agent_right = [self.ACTION_RIGHT, self.ACTION_FORWARD]
        self.knu_agent_forward = [self.ACTION_FORWARD]
        self.knu_agent_pickup = [self.ACTION_PICKUP]
        self.knu_agent_unlock = [
            self.ACTION_UNLOCK,
            self.ACTION_LEFT,
            self.ACTION_LEFT,
            self.ACTION_DROP,
            self.ACTION_LEFT,
            self.ACTION_LEFT,
            self.ACTION_FORWARD
        ]
        self.knu_agent_actions = [
            self.knu_agent_left,
            self.knu_agent_right,
            self.knu_agent_forward,
            self.knu_agent_pickup,
            self.knu_agent_unlock
        ]
        # Define state and action space sizes
        self.state_size = (26, 26)
        self.action_size = len(self.knu_agent_actions)
        # Learning parameters
        self.alpha = 0.1  # Learning rate
        self.gamma = 0.99  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        # Initialize or load the Q-table
        if load_q_table:
            with open(q_table_filename, 'rb') as f:
                self.q_table = pickle.load(f)
            print("Q-table loaded from", q_table_filename)
        else:
            self.q_table = np.zeros(self.state_size + (self.action_size,))
        self.q_table_filename = q_table_filename
        # BFS table initialization
        self.bfs_table_flag = False
        self.bfs_table = np.zeros(self.state_size)
        # Initialize goals and current goal index
        self.goals = ['KB', 'DBO', 'KG', 'DGO', 'KR', 'DRO', 'G']
        self.current_goal_index = 0
        # Direction mapping
        self.directions = ['U', 'R', 'D', 'L']  # Clockwise order
        # Action queue initialization
        self.action_queue = []
        # Current composite action info
        self.current_action_index = None
        self.current_position = None
        self.current_direction = None

    def find_agent_position(self, observation):
        for i in range(self.state_size[0]):
            for j in range(self.state_size[1]):
                cell = observation[i][j]
                if cell in ['AL', 'AR', 'AU', 'AD']:
                    direction = cell[-1]  # 'L', 'R', 'U', 'D'
                    return (i, j), direction
        return None, None

    def find_current_goal(self, observation):
        if self.current_goal_index >= len(self.goals):
            return None  # All goals achieved
        current_goal = self.goals[self.current_goal_index]
        goal_positions = []
        for i in range(self.state_size[0]):
            for j in range(self.state_size[1]):
                if observation[i][j] == current_goal:
                    goal_positions.append((i, j))
        if goal_positions:
            return goal_positions[0]  # Return the first goal position
        else:
            return None  # Goal not found

    def update_goal_flag(self):
        # Move to the next goal
        self.current_goal_index += 1
        self.bfs_table_flag = False  # Need to update BFS table

    def custom_reward(self, current_position, action_index, next_position, grid):
        # Calculate movement reward based on BFS table
        move_reward = self.bfs_table[next_position[0], next_position[1]]

        # Apply lava penalty
        if grid[next_position[0]][next_position[1]] == 'L':
            lava_penalty = -10  # Lava penalty
        else:
            lava_penalty = 0

        # Initialize goal reward
        goal_reward = 0

        # Get current goal position
        goal_position = self.find_current_goal(grid)
        if goal_position is not None:
            # Check if the agent reached the goal
            if next_position == goal_position:
                goal_reward = 100  # Goal reward
                self.update_goal_flag()  # Move to next goal
        else:
            # All goals achieved
            pass

        # Total reward calculation
        total_reward = move_reward + goal_reward + lava_penalty
        return total_reward

    def act(self, observation):
        # If there are remaining actions in the queue, return the next action
        if self.action_queue:
            return self.action_queue.pop(0)
        else:
            # Select a new composite action
            position, direction = self.find_agent_position(observation)
            if position is None:
                action_sequence = self.knu_agent_forward
                action_index = self.knu_agent_actions.index(self.knu_agent_forward)
            elif np.random.rand() < self.epsilon:
                # Random action (exploration)
                action_index = np.random.choice(self.action_size)
                action_sequence = self.knu_agent_actions[action_index]
            else:
                # Best action based on policy (exploitation)
                state_actions = self.q_table[position[0], position[1], :]
                action_index = np.argmax(state_actions)
                action_sequence = self.knu_agent_actions[action_index]

            # Add the selected composite action to the action queue
            self.action_queue.extend(action_sequence)
            # Store current composite action info
            self.current_action_index = action_index
            self.current_position = position
            self.current_direction = direction

            # Return the first action
            return self.action_queue.pop(0)

    def get_next_position(self, current_position, current_direction, action_sequence, grid):
        x, y = current_position
        direction = current_direction

        for action in action_sequence:
            if action == self.ACTION_LEFT:
                # Rotate left
                direction = self.rotate_direction(direction, 'LEFT')
            elif action == self.ACTION_RIGHT:
                # Rotate right
                direction = self.rotate_direction(direction, 'RIGHT')
            elif action == self.ACTION_FORWARD:
                # Move forward
                dx, dy = self.direction_to_delta(direction)
                new_x, new_y = x + dx, y + dy
                # Check if within grid bounds
                if 0 <= new_x < self.state_size[0] and 0 <= new_y < self.state_size[1]:
                    x, y = new_x, new_y
                # Else, position remains the same
            # For PICKUP, DROP, UNLOCK, position doesn't change
        return (x, y), direction

    def rotate_direction(self, current_direction, turn):
        idx = self.directions.index(current_direction)
        if turn == 'LEFT':
            idx = (idx - 1) % 4
        elif turn == 'RIGHT':
            idx = (idx + 1) % 4
        return self.directions[idx]

    def direction_to_delta(self, direction):
        if direction == 'U':
            return (-1, 0)
        elif direction == 'D':
            return (1, 0)
        elif direction == 'L':
            return (0, -1)
        elif direction == 'R':
            return (0, 1)

    def update_q_table(self, state, action_index, reward, next_state, done):
        x, y = state
        next_x, next_y = next_state
        current_q = self.q_table[x, y, action_index]
        if done:
            target = reward
        else:
            next_max_q = np.max(self.q_table[next_x, next_y, :])
            target = reward + self.gamma * next_max_q
        # Q-learning update rule
        self.q_table[x, y, action_index] += self.alpha * (target - current_q)

    def save_q_table(self):
        with open(self.q_table_filename, 'wb') as f:
            pickle.dump(self.q_table, f)
        print("Q-table saved to", self.q_table_filename)

    def bfs_shortest_path(self, grid, start, goal):
        rows, cols = len(grid), len(grid[0])
        # Directions: Up, Down, Left, Right
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        visited = [[False for _ in range(cols)] for _ in range(rows)]
        queue = deque([(start, 0)])  # (position, distance)
        visited[start[0]][start[1]] = True

        while queue:
            (current, distance) = queue.popleft()
            if current == goal:
                return distance
            for direction in directions:
                new_row, new_col = current[0] + direction[0], current[1] + direction[1]
                if (0 <= new_row < rows and 0 <= new_col < cols and
                        not visited[new_row][new_col] and grid[new_row][new_col] != 'W'):
                    visited[new_row][new_col] = True
                    queue.append(((new_row, new_col), distance + 1))
        return float('inf')

    def update_bfs_table(self, grid):
        goal_position = self.find_current_goal(grid)
        if goal_position is None:
            return  # No goal to update
        for i in range(26):
            for j in range(26):
                if grid[i][j] != 'W':  # If not a wall
                    distance = self.bfs_shortest_path(grid, (i, j), goal_position)
                    if distance != float('inf') and distance != 0:
                        self.bfs_table[i, j] = 1 / distance
                    else:
                        self.bfs_table[i, j] = 0
                else:
                    self.bfs_table[i, j] = 0  # Wall positions set to 0

    def update(self, observation, done):
        # Check if composite action is completed
        if not self.action_queue:
            # Simulate next position
            simulated_next_position, _ = self.get_next_position(
                self.current_position, self.current_direction,
                self.knu_agent_actions[self.current_action_index], observation
            )
            # Calculate custom reward
            custom_reward = self.custom_reward(self.current_position, self.current_action_index, simulated_next_position, observation)
            # Update Q-table
            self.update_q_table(self.current_position, self.current_action_index, custom_reward, simulated_next_position, done)
            # Decrease exploration rate
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            # If goal was updated, reset BFS table flag to update it in the next step
            if not self.bfs_table_flag:
                self.bfs_table_flag = True

    def reset(self):
        # Reset variables at the start of each episode
        self.action_queue = []
        self.current_action_index = None
        self.current_position = None
        self.current_direction = None
        self.current_goal_index = 0
        self.bfs_table_flag = False

def train(load_q_table=False, q_table_filename='q_table.pkl'):
    env = make_grid_adventure(show_screen=True)
    agent = GridAdventureRLAgent(load_q_table=load_q_table, q_table_filename=q_table_filename)
    num_episodes = 1000  # Number of training episodes

    for episode in range(num_episodes):
        observation,_ = env.reset()
        agent.reset()  # Reset agent at the start of each episode
        (state, direction) = agent.find_agent_position(observation)
        done = False
        total_reward = 0

        while not done:
            if not agent.bfs_table_flag:
                # Update BFS table
                grid = observation  # Get the grid from observation
                agent.update_bfs_table(grid)
                agent.bfs_table_flag = True

            # Get an action from the agent
            action = agent.act(observation)
            observation_next, env_reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Update agent's position and direction
            (next_state, next_direction) = agent.find_agent_position(observation_next)
            if next_state is None:
                done = True

            # Agent update (learning)
            agent.update(observation, done)

            # Update state and observation
            state = next_state
            direction = next_direction
            observation = observation_next

            if done:
                print(f"Episode: {episode+1}, Epsilon: {agent.epsilon:.4f}")
                break

    # Save the trained Q-table
    agent.save_q_table()

    # Return the trained agent
    return agent

if __name__ == '__main__':
    # Run training (set load_q_table to True to load existing Q-table)
    trained_agent = train(load_q_table=False, q_table_filename='q_table.pkl')
    # Load the saved Q-table and evaluate the agent
    agent_for_evaluation = GridAdventureRLAgent(load_q_table=True, q_table_filename='q_table.pkl')
    evaluate(agent_for_evaluation)
