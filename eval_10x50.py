import numpy as np
import torch
import gymnasium as gym
import argparse

# --- 학습 스크립트와 동일한 모델 및 정규화 함수 정의 ---
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

# <--- 변경: 결과 테이블에 표시될 실험 이름을 인자로 받도록 추가
def evaluate(ckpt_path, exp_name, num_episodes=500, seed=0, render=False):
    """학습된 모델을 CartPole 환경에서 평가합니다."""
    
    env = gym.make("CartPole-v1", render_mode="human" if render else None)

    model = MLP()
    model.load_state_dict(torch.load(ckpt_path))
    model.eval()
    print(f"Model loaded from: {ckpt_path}")

    returns = []
    rng = np.random.default_rng(seed)
    for i in range(num_episodes):
        obs, _ = env.reset(seed=int(rng.integers(0, 1_000_000)))
        done = False
        total_reward = 0
        
        while not done:
            obs_normalized = normalize_obs(obs)
            obs_tensor = torch.from_numpy(obs_normalized).unsqueeze(0)

            with torch.no_grad():
                logits = model(obs_tensor)
                action = torch.argmax(logits, dim=1).item()
            
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated

        returns.append(total_reward)
        if (i + 1) % 50 == 0:
            print(f"Episode {i+1}/{num_episodes} finished. Return: {total_reward}")

    env.close()

    returns = np.array(returns)
    avg_return = np.mean(returns)
    std_return = np.std(returns)
    success_rate = (returns == 500).mean() * 100

    print("\n--- Evaluation Results ---")
    print(f"| {'항목':<26} | {'Episodes':<8} | {'Avg. Return':<11} | {'Std.':<4} | {'Success Rate':<14} |")
    print(f"| {'-'*26} | {'-'*8} | {'-'*11} | {'-'*4} | {'-'*14} |")
    # <--- 변경: 하드코딩된 이름 대신 exp_name 인자 사용
    print(f"| {exp_name:<26} | {num_episodes:<8} | {avg_return:<11.1f} | {std_return:<4.1f} | {success_rate:<14.1f} % |")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # <--- 변경: 기본 체크포인트 경로 수정
    parser.add_argument("--ckpt", type=str, default="data/checkpoint_10x50.pt", help="모델 체크포인트 경로")
    # <--- 변경: 결과 테이블에 표시될 이름 추가 및 기본값 설정
    parser.add_argument("--name", type=str, default="Small Dataset (10x50)", help="결과 테이블에 표시될 실험 이름")
    parser.add_argument("--episodes", type=int, default=500, help="평가를 진행할 에피소드 수")
    parser.add_argument("--seed", type=int, default=0, help="평가 시 사용할 랜덤 시드")
    parser.add_argument("--render", action="store_true", help="화면에 환경을 렌더링할지 여부")
    args = parser.parse_args()

    evaluate(
        ckpt_path=args.ckpt,
        exp_name=args.name,  # <--- 변경: 이름 인자 전달
        num_episodes=args.episodes,
        seed=args.seed,
        render=args.render
    )