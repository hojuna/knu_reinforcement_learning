# 경북대학교 강화학습 프로젝트

이 저장소는 경북대학교(KNU) 강화학습 수업의 프로젝트들을 포함하고 있습니다. 다양한 강화학습 알고리즘을 구현하고 적용한 결과를 담고 있습니다.

## 📚 목차

- [프로젝트 개요](#프로젝트-개요)
- [프로젝트 구조](#프로젝트-구조)
- [설치 방법](#설치-방법)
- [프로젝트 1: Grid Adventure](#프로젝트-1-grid-adventure)
- [프로젝트 2: Grid Survivor](#프로젝트-2-grid-survivor)
- [사용된 알고리즘](#사용된-알고리즘)
- [결과](#결과)

## 프로젝트 개요

본 저장소는 강화학습의 다양한 알고리즘을 학습하고 실제 환경에 적용하는 것을 목표로 합니다. 두 가지 주요 프로젝트를 통해 Q-Learning, DQN, A2C, PPO 등의 알고리즘을 구현하였습니다.

## 프로젝트 구조

```
knu_reinforcement_learning/
├── project1/               # Grid Adventure 프로젝트
│   └── KNU/               # 최종 제출 코드
│       ├── knu_agent.py   # Q-Learning 에이전트 구현
│       ├── knu_result_q-table.pkl  # 학습된 Q-테이블
│       └── requirements.txt
├── project2/              # Grid Survivor 프로젝트
│   ├── A2C/              # Advantage Actor-Critic 구현
│   ├── DQN/              # Deep Q-Network 구현
│   └── PPO/              # Proximal Policy Optimization 구현
├── checkpoints/          # 모델 체크포인트
└── README.md
```

## 설치 방법

### 필수 요구사항

- Python 3.8 이상
- pip (Python 패키지 관리자)

### 설치 단계

1. 저장소 클론

```bash
git clone https://github.com/hojuna/knu_reinforcement_learning.git
cd knu_reinforcement_learning
```

2. 의존성 패키지 설치

```bash
cd project1/KNU
pip install -r requirements.txt
```

주요 패키지:
- `gymnasium`: OpenAI Gym의 후속 버전, RL 환경 제공
- `knu-rl-env`: KNU 강화학습 환경 패키지
- `numpy`: 수치 계산
- `pygame`: 게임 환경 렌더링
- `matplotlib`, `seaborn`: 결과 시각화

## 프로젝트 1: Grid Adventure

### 과제 설명

- **과제 문서**: [Google Docs](https://docs.google.com/document/d/11gPfnmHj1430d15T5cIFbpdtQmMyzKy7N0WXPyc12mE/edit?usp=sharing)
- **최종 제출 코드**: [project1/KNU](https://github.com/hojuna/knu_reinforcement_learning/tree/main/project1/KNU)

### 알고리즘

**Q-Learning (표 기반 강화학습)**

Grid Adventure 환경에서 에이전트가 최적의 경로를 학습하도록 Q-Learning 알고리즘을 구현하였습니다.

주요 특징:
- 상태 공간: 26x26 그리드
- 행동 공간: 좌회전, 우회전, 전진, 줍기, 놓기, 열기 (6가지 행동)
- 방향 상태: 4가지 방향 (위, 오른쪽, 아래, 왼쪽)
- 문 상태: 3가지 문 상태 관리 (잠김, 열림, 닫힘)
- 학습률(α): 0.1
- 할인율(γ): 0.95
- 탐색률(ε): 0.1 (점진적 감소)

### 실행 방법

```bash
cd project1/KNU
python knu_agent.py
```

### 결과

✅ **성과**: 가장 빠른 최적의 경로를 학습시켜 만점을 획득

⚠️ **개선 사항**: 코드 자체의 완성도는 객관적으로 개선의 여지가 있음

## 프로젝트 2: Grid Survivor

Grid Survivor 환경에서 다양한 심층 강화학습 알고리즘을 구현하고 비교 실험을 진행하였습니다.

### 구현된 알고리즘

#### 1. A2C (Advantage Actor-Critic)

여러 버전의 A2C 구현을 통해 성능을 개선해나갔습니다:
- A2C_v1 ~ A2C_v15: 반복적인 개선 과정
- 주요 개선사항:
  - Huber Loss 적용
  - RMSprop 옵티마이저 사용
  - 병렬 환경 학습

디렉토리: `project2/A2C/`

#### 2. DQN (Deep Q-Network)

다양한 DQN 변형을 구현하였습니다:
- 기본 DQN with CNN
- Dueling DQN
- Experience Replay Buffer

주요 파일:
- `DQN_cnn_agent.py`: CNN 기반 DQN 에이전트
- `DQN_network.py`: 신경망 구조
- `DQN_memory.py`: 경험 재생 메모리

학습된 모델:
- `dueling_dqn_grid_survivor_final.pth`
- `dueling_dqn_checkpoint_episode_1000.pth`

디렉토리: `project2/DQN/`

#### 3. PPO (Proximal Policy Optimization)

정책 기반 강화학습 알고리즘인 PPO를 구현하였습니다.

디렉토리: `project2/PPO/`

### 실행 방법

각 알고리즘별 실행:

```bash
# A2C 실행
cd project2/A2C/A2C_v15
python a2c_Rms_huber.py

# DQN 실행
cd project2/DQN/dqn_cnn_2
python DQN_cnn_run.py

# PPO 실행
cd project2/PPO/PPO_v1
python PPO_run.py
```

## 사용된 알고리즘

### 1. Q-Learning

표 기반 강화학습 알고리즘으로, 상태-행동 쌍에 대한 Q-값을 테이블로 저장합니다.

**업데이트 식**:
```
Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
```

### 2. DQN (Deep Q-Network)

신경망을 사용하여 Q-함수를 근사하는 알고리즘입니다.

주요 기법:
- Experience Replay: 과거 경험을 재사용
- Target Network: 학습 안정화
- CNN: 이미지 기반 상태 처리

### 3. A2C (Advantage Actor-Critic)

Actor-Critic 구조를 사용하는 정책 기반 알고리즘입니다.

구성요소:
- Actor: 정책 네트워크 (행동 선택)
- Critic: 가치 네트워크 (상태 평가)
- Advantage 함수: 행동의 상대적 가치 평가

### 4. PPO (Proximal Policy Optimization)

정책 업데이트를 제한하여 안정적인 학습을 수행하는 알고리즘입니다.

특징:
- Clipped Surrogate Objective
- 정책 업데이트의 안정성 보장
- 샘플 효율성 개선

## 결과

### Project 1 성과

- ✅ 최적 경로 학습 성공
- ✅ 과제 만점 획득
- 📝 코드 리팩토링 필요

### Project 2 진행 상황

다양한 알고리즘 구현 완료:
- A2C: 15개 버전의 반복 개선
- DQN: CNN 기반 및 Dueling DQN 구현
- PPO: 기본 구현 완료

학습된 모델 체크포인트 저장:
- `safety_model.pth`
- `dueling_dqn_grid_survivor_final.pth`

## 참고 자료

- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/)
- Sutton & Barto - Reinforcement Learning: An Introduction

## 라이선스

이 프로젝트는 교육 목적으로 작성되었습니다.

## 연락처

프로젝트 관련 문의사항이 있으시면 이슈를 등록해주세요.
