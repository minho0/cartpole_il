import numpy as np
import torch
import gymnasium as gym
import argparse

# --- 학습 스크립트와 동일한 모델 및 정규화 함수 정의 ---
# NOTE: 유지보수를 위해 train.py와 evaluate.py가 공통으로 사용하는
#       이러한 부분은 별도의 파일(예: model.py)로 분리하는 것이 좋습니다.

SCALE = np.array([2.4, 3.0, 0.2095, 3.5], dtype=np.float32)

def normalize_obs(X_np: np.ndarray):
    """관측값을 SCALE로 나누고 [-2, 2] 범위로 클리핑합니다."""
    X = X_np / SCALE
    return np.clip(X, -2.0, 2.0).astype(np.float32)

class MLP(torch.nn.Module):
    def __init__(self, in_dim=4, hid=32, out_dim=2):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hid), torch.nn.ReLU(),
            torch.nn.Linear(hid, hid), torch.nn.ReLU(),
            torch.nn.Linear(hid, out_dim)
        )
    def forward(self, x): return self.net(x)
# -----------------------------------------------------------

def evaluate(ckpt_path, num_episodes=500, seed=0, render=False):
    """학습된 모델을 CartPole 환경에서 평가합니다."""
    
    # 1. 환경 생성
    # render_mode="human"으로 설정 시 화면에 게임 창이 나타남
    env = gym.make("CartPole-v1", render_mode="human" if render else None)

    # 2. 모델 불러오기
    model = MLP()
    model.load_state_dict(torch.load(ckpt_path))
    model.eval()  # 평가 모드로 설정
    print(f"Model loaded from: {ckpt_path}")

    # 3. 평가 실행
    returns = []
    rng = np.random.default_rng(seed)
    for i in range(num_episodes):
        obs, _ = env.reset(seed=int(rng.integers(0, 1_000_000)))
        done = False
        total_reward = 0
        
        while not done:
            # 3.1. 관측값 정규화 및 텐서 변환
            obs_normalized = normalize_obs(obs)
            obs_tensor = torch.from_numpy(obs_normalized).unsqueeze(0) # 배치 차원 추가

            # 3.2. 행동 결정 (추론)
            with torch.no_grad(): # 기울기 계산 비활성화
                logits = model(obs_tensor)
                action = torch.argmax(logits, dim=1).item()
            
            # 3.3. 환경에서 행동 수행
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated

        returns.append(total_reward)
        if (i + 1) % 50 == 0:
            print(f"Episode {i+1}/{num_episodes} finished. Return: {total_reward}")

    env.close()

    # 4. 결과 계산 및 출력
    returns = np.array(returns)
    avg_return = np.mean(returns)
    std_return = np.std(returns)
    # 성공 기준: CartPole-v1의 최대 스텝인 500 달성
    success_rate = (returns == 500).mean() * 100

    print("\n--- Evaluation Results ---")
    print(f"| {'항목':<26} | {'Episodes':<8} | {'Avg. Return':<11} | {'Std.':<4} | {'Success Rate':<14} |")
    print(f"| {'-'*26} | {'-'*8} | {'-'*11} | {'-'*4} | {'-'*14} |")
    print(f"| {'Large Dataset, Single Model':<26} | {num_episodes:<8} | {avg_return:<11.1f} | {std_return:<4.1f} | {success_rate:<14.1f} % |")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="data/checkpoint_s0.pt", help="모델 체크포인트 경로")
    parser.add_argument("--episodes", type=int, default=500, help="평가를 진행할 에피소드 수")
    parser.add_argument("--seed", type=int, default=0, help="평가 시 사용할 랜덤 시드")
    parser.add_argument("--render", action="store_true", help="화면에 환경을 렌더링할지 여부")
    args = parser.parse_args()

    evaluate(ckpt_path=args.ckpt, num_episodes=args.episodes, seed=args.seed, render=args.render)