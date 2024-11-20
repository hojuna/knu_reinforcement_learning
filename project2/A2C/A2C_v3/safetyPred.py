import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque

from knu_rl_env.grid_survivor import make_grid_survivor

class SafetyClassifier(nn.Module):
    def __init__(self, hidden_size=64):
        super().__init__()
        
        # 입력: 5개 셀 * 3방향 = 15차원
        self.net = nn.Sequential(
            nn.Linear(15, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.net(x)

class SafetyPredictor:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = SafetyClassifier().to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.BCELoss()
        
        # 경험 리플레이 버퍼
        self.buffer = deque(maxlen=10000)
        
        # 셀 타입을 인덱스로 매핑
        self.cell_types = {'E':0, 'W':1, 'B':2, 'H':3, 'K':4, 
                          'AL':5, 'AR':5, 'AU':5, 'AD':5}  # 에이전트는 모두 5로



        
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
            'UP': [(-2, 0), (-1, -1), (-1, 1)],
            'DOWN': [(2, 0), (1, -1), (1, 1)],
            'RIGHT': [(0, 2), (-1, 1), (1, 1)],
            'LEFT': [(0, -2), (-1, -1), (1, -1)]
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
    
    def process_grid_state(self, grid, pos, direction):
        """전방 5개 셀의 상태를 추출"""
        x, y = pos
        features = []
        
        # 방향에 따른 offset 매핑
        direction_offsets = {
            'UP': [(-1,0), (-1,-1), (-1,1)],    # 위
            'DOWN': [(1,0), (1,-1), (1,1)],     # 아래
            'LEFT': [(0,-1), (-1,-1), (1,-1)],  # 왼쪽
            'RIGHT': [(0,1), (-1,1), (1,1)]     # 오른쪽
        }
        
        offsets = direction_offsets[direction]
        
        # 각 방향의 셀 상태 추출
        for dx, dy in offsets:
            nx, ny = x + dx, y + dy
            # 격자 범위 체크
            if 0 <= nx < len(grid) and 0 <= ny < len(grid[0]):
                cell_type = self.cell_types.get(grid[nx][ny], 0)
            else:
                cell_type = 1  # 격자 밖은 벽으로 처리
            
            # 원-핫 인코딩
            cell_features = [0] * 5  # 5가지 셀 타입
            cell_features[cell_type] = 1
            features.extend(cell_features)
        
        return torch.FloatTensor(features).to(self.device)
    
    def add_experience(self, grid, pos, direction, is_safe):
        """경험 추가"""
        state = self.process_grid_state(grid, pos, direction)
        self.buffer.append((state, is_safe))
    
    def train_batch(self, batch_size=32):
        """모델 학습"""
        if len(self.buffer) < batch_size:
            return
        
        # 배치 샘플링
        indices = np.random.choice(len(self.buffer), batch_size)
        states, labels = zip(*[self.buffer[i] for i in indices])
        
        # 텐서 변환
        states = torch.stack(states)
        labels = torch.FloatTensor(labels).to(self.device)
        
        # 예측 및 손실 계산
        self.optimizer.zero_grad()
        predictions = self.model(states)
        loss = self.criterion(predictions.squeeze(), labels)
        
        # 역전파 및 최적화
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def predict(self, grid, pos, direction, threshold=0.5):
        """안전 여부 예측"""
        state = self.process_grid_state(grid, pos, direction)
        with torch.no_grad():
            prediction = self.model(state.unsqueeze(0))
        return prediction.item() > threshold
    
    def save(self, path):
        """모델 저장"""
        torch.save({
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict()
        }, path)
    
    def load(self, path):
        """모델 로드"""
        checkpoint = torch.load(path, weights_only=True)
        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])

# 학습 데이터 수집 및 모델 학습
def train_safety_predictor(env, n_episodes=1000):
    predictor = SafetyPredictor()
    
    for episode in range(n_episodes):
        state, _ = env.reset()
        done = False
        
        while not done:
            grid = state['grid']
            pos, direction = predictor.extract_agent_info(grid)
            
            # 현재 상태가 안전한지 판단
            next_pos = predictor._get_next_position(pos, direction)
            is_safe = predictor._is_valid_move(next_pos, grid)
            
            # 경험 추가
            predictor.add_experience(grid, pos, direction, is_safe)
            
            # 배치 학습
            if len(predictor.buffer) >= 32:
                loss = predictor.train_batch()
                
            # 환경과 상호작용 (랜덤 행동)
            action = np.random.randint(3)
            state, _, done, _, _ = env.step(action)
        
        # 에피소드 종료 시 중간 결과 출력
        if episode % 100 == 0:
            print(f"Episode {episode} completed")
    
    return predictor

# 사용 예시
if __name__ == "__main__":
    env = make_grid_survivor(show_screen=False)
    
    # 모델 학습
    predictor = train_safety_predictor(env)
    
    # 모델 저장
    predictor.save("safety_model.pth")
    
    # 테스트
    state, _ = env.reset()
    grid = state['grid']
    pos, direction = predictor.extract_agent_info(grid)
    is_safe = predictor.predict(grid, pos, direction)
    print(f"Is current move safe? {is_safe}")