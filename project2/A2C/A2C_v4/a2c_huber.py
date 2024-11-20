import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from collections import deque
import random
from knu_rl_env.grid_survivor import GridSurvivorAgent, make_grid_survivor
import wandb
# A2C 네트워크 정의
# # A2C 네트워크 정의
# class A2CNetwork(nn.Module):

#     def __init__(self, input_size, hidden_size, num_actions):
#         super(A2CNetwork, self).__init__()

#         # 특징 추출을 위한 공통 레이어
#         self.shared = nn.Sequential(
#             nn.Linear(input_size, hidden_size),
#             nn.ReLU(),
#             nn.Linear(hidden_size, hidden_size),
#             nn.ReLU()
#         )

#         # Actor: 정책 (행동 확률 출력)
#         self.policy = nn.Sequential(
#             nn.Linear(hidden_size, num_actions),
#             nn.Softmax(dim=-1)
#         )

#         # Critic: 가치 함수 (상태 가치 출력)
#         self.value = nn.Linear(hidden_size, 1)

#     def forward(self, x):
#         shared_features = self.shared(x)
#         action_probs = self.policy(shared_features)
#         state_value = self.value(shared_features)
#         return action_probs, state_value
   

class A2CNetwork(nn.Module):
   def __init__(self, input_size, hidden_size, num_actions):
       super(A2CNetwork, self).__init__()
       
       # 특징 추출을 위한 공통 레이어 (살짝 더 깊게)
       self.shared = nn.Sequential(
           nn.Linear(input_size, hidden_size),
           nn.ReLU(),
           nn.LayerNorm(hidden_size),  # 학습 안정화
           
           nn.Linear(hidden_size, hidden_size),
           nn.ReLU(),
           nn.LayerNorm(hidden_size),
           
           nn.Linear(hidden_size, hidden_size),
           nn.ReLU()
       )
       
       # Actor: 정책 (행동 확률 출력)
       self.policy = nn.Sequential(
           nn.Linear(hidden_size, hidden_size // 2),
           nn.ReLU(),
           nn.Linear(hidden_size // 2, num_actions),
           nn.Softmax(dim=-1)
       )
       
       # Critic: 가치 함수 (상태 가치 출력)
       self.value = nn.Sequential(
           nn.Linear(hidden_size, hidden_size // 2),
           nn.ReLU(),
           nn.Linear(hidden_size // 2, 1)
       )
   
   def forward(self, x):
       shared_features = self.shared(x)
       action_probs = self.policy(shared_features)
       state_value = self.value(shared_features)
       return action_probs, state_value



# A2C 에이전트 정의
class A2CAgent(): # GridSurvivorAgent 상속 제거
    def __init__(self, state_size, save_dir=f"/home/comoz/main_project/knu_reinforcement_learning/project2/A2C/A2C_v4/save_model"):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = A2CNetwork(state_size, 128, 3).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=0.001)
        
        # 하이퍼파라미터
        self.gamma = 0.99
        self.entropy_coef = 0.01
        self.value_coef = 0.5
        
        # 경험 저장용 버퍼
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.raw_states = []       # 원본 상태
        
        # 모델 저장 경로
        self.save_dir = save_dir

        self.visit_table = np.ones((35,35))
        
    def preprocess_state(self,state):
        # 위치와 방향 분리
        cell_types = {'E':0, 'W':1, 'B':2, 'H':3, 'K':4, 'A':5}  # 6차원
        directions = {'L':0, 'R':1, 'U':2, 'D':3}  # 추가 방향 정보
        grid = state['grid']
        N = len(grid)
        
        # 기본 그리드 정보 (6차원)
        grid_tensor = torch.zeros(N, N, 6)  # 6개 셀 타입
        # 방향 정보 (4차원)
        direction_tensor = torch.zeros(4)    # 4개 방향
        
        agent_pos = None
        
        for i in range(N):
            for j in range(N):
                cell = grid[i][j]
                if cell.startswith('A'):  # 에이전트 셀
                    grid_tensor[i, j, 5] = 1  # A 표시
                    direction_tensor[directions[cell[1]]] = 1  # 방향 표시
                    agent_pos = (i, j)
                else:
                    grid_tensor[i, j, cell_types[cell]] = 1
        
        # 1D로 변환
        grid_flat = grid_tensor.flatten()
        
        # 최종 상태 벡터 구성
        state_vector = torch.cat([
            grid_flat,
            direction_tensor,
            torch.tensor([state['hit_points'] / 100.0])
        ])
        
        return state_vector.to(self.device)

    def find_forward_obj(self, grid):
        """
        에이전트의 방향에 따라 앞으로 있는 객체를 탐색하고, 우선순위에 따라 가장 높은 우선순위의 객체를 반환합니다.
        
        Parameters:
        - grid (2D numpy array): 현재 그리드 상태. 각 셀은 'W', 'B', 'H', 'K', 'AU', 'AD', 'AL', 'AR' 등으로 표시됩니다.
        
        Returns:
        - target (str or None): 우선순위가 가장 높은 객체의 기호 또는 객체가 없으면 None.
        """
        # 객체 우선순위 정의 (낮은 인덱스일수록 높은 우선순위)
        priority = ['K', 'H','B','W','E' ]
        
        # 에이전트의 위치와 방향 추출
        position, direction = self.extract_agent_info(grid)
        
        if position is None or direction is None:
            return None
        
        x, y = position
        grid_height, grid_width = grid.shape
        
        # 방향별 탐색할 상대적 셀 좌표 정의
        direction_offsets = {
            'UP': [(-2, 0), (-1, -1), (-1, 1),(-1,0)],
            'DOWN': [(2, 0), (1, -1), (1, 1),(1,0)],
            'RIGHT': [(0, 2), (-1, 1), (1, 1),(0,1)],
            'LEFT': [(0, -2), (-1, -1), (1, -1),(0,-1)]
        }
        
        # 현재 방향에 따른 탐색할 셀 좌표 가져오기
        offsets = direction_offsets.get(direction.upper())
        if offsets is None:
            return None  # 유효하지 않은 방향일 경우
        
        found_symbols = []
        
        for dx, dy in offsets:
            cx, cy = x + dx, y + dy
            # 그리드 범위 내인지 확인
            if 0 <= cx < grid_height and 0 <= cy < grid_width:
                cell_symbol = grid[cx][cy]
                if cell_symbol in priority:
                    found_symbols.append(cell_symbol)
        
        if not found_symbols:
            return None  # 탐색된 객체가 없을 경우
        
        # 우선순위에 따라 가장 높은 우선순위의 객체 선택
        for symbol in priority:
            if symbol in found_symbols:
                return symbol
        
        return None

    def extract_agent_info(self,grid):
        direction_symbols = {'AU': 'UP', 'AD': 'DOWN', 'AL': 'LEFT', 'AR': 'RIGHT'}
        positions = np.argwhere(np.isin(grid, list(direction_symbols.keys())))

        if len(positions) == 0:
            return (None, None)

        x, y = positions[0]
        symbol = grid[x, y]
        direction = direction_symbols.get(symbol, 'DOWN')

        return (x, y), direction


    def get_valid_actions(self, state):
        """유효한 행동 마스크 생성"""
        grid = state['grid']
        valid_actions = torch.ones(3, device=self.device)  # [LEFT, RIGHT, FORWARD]
        
        # 에이전트 위치와 방향 파악
        position, direction = self.extract_agent_info(grid)
        if position is None or direction is None:
            return valid_actions
        
        x, y = position
        # FORWARD 액션이 유효한지 확인
        next_pos = self._get_next_position(position, direction)
        if not self._is_valid_move(next_pos, grid):
            valid_actions[2] = 0  # FORWARD 불가능
        elif self.find_forward_obj(grid) == 'B':
            valid_actions[2] += 1  # FORWARD 가능
        
        return valid_actions
    
    def _get_next_position(self, position, direction):
        """주어진 방향으로 한 칸 앞의 위치 반환"""
        x, y = position
        if direction == 'UP':
            return (x-1, y)
        elif direction == 'DOWN':
            return (x+1, y)
        elif direction == 'LEFT':
            return (x, y-1)
        elif direction == 'RIGHT':
            return (x, y+1)
        return position
    
    def _is_valid_move(self, next_pos, grid):
        """해당 위치로 이동이 가능한지 확인"""
        x, y = next_pos
        # 격자 범위 체크
        if not (0 <= x < len(grid) and 0 <= y < len(grid[0])):
            return False
        # 벽 체크
        if grid[x][y] == 'W' or self.find_forward_obj(grid) == 'K':
            return False

        
        return True

    def calculate_reward(self,state, next_state, done, step):
        total_reward = 0

        base_reward = -0.1

        # survival_reward = 0.01 * (step / 100)
        survival_reward=0

        previous_bees = np.sum(state['grid'] == 'B')
        current_bees = np.sum(next_state['grid'] == 'B')
        rescued_bees = previous_bees - current_bees

        bee_reward = rescued_bees * 100

        if 'B'==self.find_forward_obj(next_state):
            bee_reward+=5


        previous_hornet = np.sum(state['grid'] == 'H')
        current_hornet = np.sum(next_state['grid'] == 'H')
        rescued_hornet = previous_hornet - current_hornet
        if state['hit_points']>=40:
            hornet_penalty = rescued_hornet * -5
        else:
            hornet_penalty = rescued_hornet * -10


        health_reward = 0
        # health_diff = next_state['hit_points'] - state['hit_points']
        # health_reward = health_diff * 0.1


        early_termination = 0

        if current_bees != 0 and done:
            early_termination-=20


        position, direction = self.extract_agent_info(state['grid'])
        next_position, next_direction = self.extract_agent_info(next_state['grid'])
        collision_penalty = -0.1 if position == next_position  else 0
        
        first_visit_reward=0
        if self.visit_table[position]==1:
            first_visit_reward=1
            self.visit_table[position]=0

        bee_distance_reward = 0
        # if 'B' in next_state['grid']:
        #     min_distance = self.min_distance(next_state, next_position, 'B')
        #     if min_distance is not None:
        #         bee_distance_reward += 0.1 / (min_distance + 1e-6)

        hornet_distance_penalty = 0
        # if 'H' in next_state['grid']:
        #     min_distance = self.min_distance(next_state, next_position, 'H')
        #     if min_distance is not None:
            
        #         hornet_distance_penalty -= 1 / (min_distance + 1e-6)

        killerbee_distance_penalty = 0
        # if 'K' in next_state['grid']:
        #     min_distance = self.min_distance(next_state, next_position, 'K')
        #     if min_distance is not None:
        #         killerbee_distance_penalty -= 1 / (min_distance + 1e-6)


        if 0 == current_bees and done:
            total_reward += 500 / (step + 1e-6)

        total_reward = (base_reward + survival_reward + bee_reward + hornet_penalty + 
                    health_reward + early_termination + collision_penalty + 
                     bee_distance_reward + hornet_distance_penalty +
                    killerbee_distance_penalty+first_visit_reward)

        return total_reward
    
    def act(self, state):
        """행동 선택"""
        state_tensor = self.preprocess_state(state)
        valid_actions = self.get_valid_actions(state)
        
        with torch.no_grad():
            action_probs, value = self.network(state_tensor)

                        # 불가능한 행동의 확률을 0으로 만들고 재정규화
            masked_probs = action_probs * valid_actions
            if masked_probs.sum() > 0:  # 적어도 하나의 유효한 행동이 있는 경우
                masked_probs = masked_probs / masked_probs.sum()
            else:
                # 모든 행동이 불가능한 경우 (발생하면 안되지만 안전장치)
                masked_probs = action_probs
            
            # 행동 샘플링
            dist = Categorical(masked_probs)
            action = dist.sample()
        
        
        # 경험 저장
        self.states.append(state_tensor)
        self.raw_states.append(state)
        self.actions.append(action)
        self.values.append(value)
        
        return action.item()
    
    def update(self, reward, done):
        """경험 업데이트"""
        self.rewards.append(reward)
        self.dones.append(done)
        
        if done:
            loss = self.learn()
            self.reset_episode()
            return loss
        return 0
    
    def learn(self):
        """A2C 학습"""
        # 리턴 계산
        returns = []
        R = 0
        for r, d in zip(reversed(self.rewards), reversed(self.dones)):
            R = r + self.gamma * R * (1-d)
            returns.insert(0, R)
        returns = torch.tensor(returns).to(self.device)
        
        # 배치 데이터 준비
        states = torch.stack(self.states)
        actions = torch.stack(self.actions)
        values = torch.stack(self.values)
        
        # 각 상태에 대한 유효 행동 마스크 준비
        valid_actions_batch = torch.stack([self.get_valid_actions(s) for s in self.raw_states])

          # 네트워크 통과
        action_probs, state_values = self.network(states)
        
        # 마스킹 적용
        masked_probs = action_probs * valid_actions_batch
        masked_probs = masked_probs / (masked_probs.sum(dim=1, keepdim=True) + 1e-8)
        
        # Advantage 계산
        advantages = returns - values.squeeze()
        
        # 정책 손실 계산 (마스킹된 확률 사용)
        dist = Categorical(masked_probs)
        policy_loss = -(dist.log_prob(actions) * advantages.detach()).mean()
        
        # 가치 손실 계산
        # value_loss = F.mse_loss(values.squeeze(), returns)
        value_loss = F.huber_loss(values.squeeze(), returns, delta=1.0)
        
        # 엔트로피 보너스
        entropy_loss = -self.entropy_coef * dist.entropy().mean()
        
        # 전체 손실
        total_loss = policy_loss + self.value_coef * value_loss + entropy_loss
        
        # 역전파 및 최적화
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return total_loss.item()
    
    def reset_episode(self):
        """에피소드 버퍼 초기화"""
        self.states.clear()
        self.raw_states.clear()  # 원본 상태 버퍼도 초기화
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()
    
    def save(self, episode):
        """모델 저장"""
        torch.save({
            'episode': episode,
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, f"{self.save_dir}/a2c_checkpoint_{episode}.pth")
    
    def load(self, path):
        """모델 로드"""
        checkpoint = torch.load(path)
        self.network.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['episode']

def train(env, agent, num_episodes, save_interval=100):
    """학습 함수"""
    best_reward = float('-inf')
    reward_history = []

    wandb.init(project="a2c_v1", name="a2c_v2_huber")
    config = wandb.config
    config.num_episodes = num_episodes
    config.save_interval = save_interval

    for episode in range(num_episodes):
        state,_ = env.reset()
        episode_reward = 0
        done = False
        step=0
        max_step = 1200

        while not done and step<max_step:
            # 행동 선택 및 환경과 상호작용
            action = agent.act(state)
            next_state, reward, done, _ , _= env.step(action)
            step+=1
            
            # 보상 계산 및 업데이트
            reward = agent.calculate_reward(state, next_state, done,step)
            episode_reward += reward

            if step >= max_step:
                done=True
                
            loss = agent.update(reward, done)
            state = next_state
        
        # 성과 기록
        reward_history.append(episode_reward)
        
        # 모델 저장
        if episode % save_interval == 0:
            agent.save(episode)
        
        # 최고 성능 모델 저장
        if episode_reward > best_reward:
            best_reward = episode_reward
            agent.save('best')

        remain_bees = np.sum(next_state['grid'] == 'B')

        wandb.log({"episode_reward": episode_reward, "best_reward": best_reward,"step":step,"episode":episode,"remain_bees":remain_bees,"loss":loss})
        
        # 진행상황 출력
        if episode % 10 == 0:
            avg_reward = np.mean(reward_history[-10:])
            print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}, Best Reward: {best_reward:.2f}, remain_bees: {remain_bees}, step: {step}")
    
    return reward_history

def calculate_state_size(state):
    grid = state['grid']
    N = len(grid)
    grid_info_size = N * N * 6  # 6가지 셀 타입
    direction_size = 4          # 4가지 방향
    hit_points_size = 1        # 체력 정보
    
    return grid_info_size + direction_size + hit_points_size

if __name__ == "__main__":
    # 환경 생성
    env = make_grid_survivor(show_screen=False)
    state,_ = env.reset()
    
    # 임시 테스트용 state_size (실제 환경에 맞게 수정 필요)
    # 에이전트 생성
    agent = A2CAgent(calculate_state_size(state))
    
    # 학습 설정
    num_episodes = 20000
    save_interval = 10000
    
    # 학습 실행
    reward_history = train(env, agent, num_episodes, save_interval)
    
    # 최종 모델 저장
    agent.save('final')
    
    print("Training completed!")