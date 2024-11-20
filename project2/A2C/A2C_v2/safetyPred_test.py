import torch
import numpy as np
from knu_rl_env.grid_survivor import make_grid_survivor
import random
from safetyPred import SafetyPredictor
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
class SafeRandomAgent:
    def __init__(self, safety_model_path):
        self.safety_predictor = SafetyPredictor()
        self.safety_predictor.load(safety_model_path)
        
    def act(self, state):
        """안전한 행동들 중에서 랜덤하게 선택"""
        grid = state['grid']
        pos, direction = self.safety_predictor.extract_agent_info(grid)
        
        # 각 행동의 안전성 확인
        safe_actions = []
        
        # 현재 방향에서 전진했을 때의 안전성
        forward_safety = self.safety_predictor.predict(grid, pos, direction)
        if forward_safety:
            safe_actions.append(2)  # FORWARD
            
        # 회전은 항상 안전 (LEFT, RIGHT)
        safe_actions.extend([0, 1])  
        
        # 안전한 행동들 중에서 랜덤 선택
        return random.choice(safe_actions)

def test_safe_random_agent(episodes=100):
    """안전한 랜덤 에이전트 테스트"""
    env = make_grid_survivor(show_screen=False)  # 시각화 켜기
    agent = SafeRandomAgent(f"/home/comoz/main_project/knu_reinforcement_learning/safety_model.pth")
    
    results = []
    
    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        steps = 0
        remaining_bees = np.sum(state['grid'] == 'B')
        
        while not done:
            action = agent.act(state)
            next_state, reward, done, _, _ = env.step(action)
            
            state = next_state
            steps += 1
            episode_reward += reward
            
            if steps >= 1200:  # max step
                break
        
        final_remaining_bees = np.sum(state['grid'] == 'B')
        hp = state['hit_points']
        
        results.append({
            'episode': episode,
            'steps': steps,
            'reward': episode_reward,
            'saved_bees': remaining_bees - final_remaining_bees,
            'remaining_bees': final_remaining_bees,
            'final_hp': hp
        })
        
        print(f"Episode {episode}:")
        print(f"  Steps: {steps}")
        print(f"  Saved Bees: {remaining_bees - final_remaining_bees}")
        print(f"  Remaining Bees: {final_remaining_bees}")
        print(f"  Final HP: {hp}")
        print("------------------------")
    
    # 전체 통계 출력
    total_episodes = len(results)
    avg_steps = sum(r['steps'] for r in results) / total_episodes
    avg_saved_bees = sum(r['saved_bees'] for r in results) / total_episodes
    avg_hp = sum(r['final_hp'] for r in results) / total_episodes
    
    print("\nOverall Statistics:")
    print(f"Average Steps: {avg_steps:.2f}")
    print(f"Average Saved Bees: {avg_saved_bees:.2f}")
    print(f"Average Final HP: {avg_hp:.2f}")
    
    # 최고의 에피소드 찾기
    best_episode = max(results, key=lambda x: (x['saved_bees'], -x['steps'], x['final_hp']))
    print("\nBest Episode:")
    print(f"Episode: {best_episode['episode']}")
    print(f"Steps: {best_episode['steps']}")
    print(f"Saved Bees: {best_episode['saved_bees']}")
    print(f"Final HP: {best_episode['final_hp']}")
    
    return results


def test_safety_model_accuracy(safety_model_path, n_episodes=10):
   """안전 예측 모델의 정확도 테스트"""
   predictor = SafetyPredictor()
   predictor.load(safety_model_path)
   
   env = make_grid_survivor(show_screen=False)
   predictions = []
   ground_truths = []
   
   for episode in range(n_episodes):
       state, _ = env.reset()
       done = False
       
       while not done:
           # 현재 상태에서의 예측값과 실제값 수집
           grid = state['grid']
           pos, direction = predictor.extract_agent_info(grid)
           
           if pos is not None:
               # 모델 예측
               pred = predictor.predict(grid, pos, direction)
               
               # 실제 안전 여부
               next_pos = predictor._get_next_position(pos, direction)
               actual_safe = not (grid[next_pos[0]][next_pos[1]] in ['W', 'K'])
               
               predictions.append(int(pred))
               ground_truths.append(int(actual_safe))
           
           # 랜덤 행동으로 다음 상태로
           action = np.random.randint(3)
           state, _, done, _, _ = env.step(action)
           
   # 성능 메트릭 계산
   accuracy = accuracy_score(ground_truths, predictions)
   precision = precision_score(ground_truths, predictions)
   recall = recall_score(ground_truths, predictions)
   conf_matrix = confusion_matrix(ground_truths, predictions)
   
   print(f"\nSafety Model Test Results (over {len(predictions)} samples):")
   print(f"Accuracy: {accuracy:.4f}")
   print(f"Precision: {precision:.4f}")
   print(f"Recall: {recall:.4f}")
   print("\nConfusion Matrix:")
   print(conf_matrix)
   
   return accuracy, precision, recall, conf_matrix


if __name__ == "__main__":
    # results = test_safe_random_agent(episodes=100)
    test_safety_model_accuracy(f"/home/comoz/main_project/knu_reinforcement_learning/safety_model.pth", n_episodes=10)