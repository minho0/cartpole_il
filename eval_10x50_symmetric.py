# evaluate_ensemble_tta.py
import numpy as np
import torch
import gymnasium as gym
import argparse
from typing import List

# --- 학습 스크립트와 동일한 정규화 ---
SCALE = np.array([2.4, 3.0, 0.2095, 3.5], dtype=np.float32)
def normalize_obs(x_np: np.ndarray):
    x = x_np / SCALE
    return np.clip(x, -2.0, 2.0).astype(np.float32)

# --- 학습 스크립트와 동일한 모델 ---
class MLP(torch.nn.Module):
    def __init__(self, in_dim=4, hid=32, out_dim=2):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hid), torch.nn.ReLU(),
            torch.nn.Linear(hid, hid), torch.nn.ReLU(),
            torch.nn.Linear(hid, out_dim)
        )
    def forward(self, x): return self.net(x)

def load_models(seeds: List[int], ckpt_tpl: str, hid: int, device: str) -> List[torch.nn.Module]:
    models = []
    for s in seeds:
        path = ckpt_tpl.format(seed=s)
        m = MLP(hid=hid).to(device)
        m.load_state_dict(torch.load(path, map_location=device))
        m.eval()
        print(f"[load] {path}")
        models.append(m)
    return models

@torch.no_grad()
def tta_logits(models: List[torch.nn.Module], obs_tensor: torch.Tensor, use_tta: bool) -> torch.Tensor:
    """
    models: [M] list
    obs_tensor: [1, 4] normalized, on device
    returns: [1, 2] averaged logits
    """
    # ensemble 평균
    logits_sum = torch.zeros((1, 2), device=obs_tensor.device, dtype=torch.float32)
    for m in models:
        lo = m(obs_tensor)  # [1,2]

        if use_tta:
            # 좌우 반전: 입력 부호 반전 -> 행동도 좌우 뒤집힘(클래스 0<->1)
            lf = m(-obs_tensor)
            # 클래스 순서를 뒤집어서(0<->1) 원래 의미계로 맞춘 후 평균
            lf_swapped = torch.flip(lf, dims=[1])  # [1,2]
            lo = (lo + lf_swapped) * 0.5

        logits_sum += lo

    return logits_sum / len(models)

def evaluate_ensemble_tta(exp_name: str, seeds: List[int], ckpt_tpl: str,
                          num_episodes: int, seed: int, render: bool,
                          hid: int, device: str, success_at: int, use_tta: bool):
    torch.set_grad_enabled(False)

    env = gym.make("CartPole-v1", render_mode="human" if render else None)
    models = load_models(seeds, ckpt_tpl, hid=hid, device=device)

    returns = []
    rng = np.random.default_rng(seed)

    for i in range(num_episodes):
        obs, _ = env.reset(seed=int(rng.integers(0, 1_000_000)))
        done = False
        total_reward = 0.0

        while not done:
            obs_norm = normalize_obs(obs)
            obs_tensor = torch.from_numpy(obs_norm).to(device=device, dtype=torch.float32).unsqueeze(0)

            logits = tta_logits(models, obs_tensor, use_tta=use_tta)
            action = int(torch.argmax(logits, dim=1).item())

            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)
            done = terminated or truncated

        returns.append(total_reward)
        if (i + 1) % 50 == 0:
            print(f"Episode {i+1}/{num_episodes} finished. Return: {total_reward:.0f}")

    env.close()

    # 통계
    returns = np.asarray(returns, dtype=np.float32)
    avg_return = float(returns.mean())
    std_return = float(returns.std())
    med_return = float(np.median(returns))
    success_rate = float((returns >= success_at).mean() * 100.0)

    print("\n--- Evaluation Results ---")
    print(f"| {'항목':<26} | {'Episodes':<8} | {'Avg. Return':<11} | {'Median':<6} | {'Std.':<4} | {'Success Rate':<14} |")
    print(f"| {'-'*26} | {'-'*8} | {'-'*11} | {'-'*6} | {'-'*4} | {'-'*14} |")
    print(f"| {exp_name:<26} | {num_episodes:<8} | {avg_return:<11.1f} | {med_return:<6.1f} | {std_return:<4.1f} | {success_rate:<14.1f} % |")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--name", type=str, default="Ensemble+TTA", help="결과 테이블 표기 이름")
    p.add_argument("--seeds", type=str, default="0,1,2", help="앙상블에 사용할 시드 목록 (콤마 구분)")
    p.add_argument("--ckpt_tpl", type=str, default="data/checkpoint_eff_s{seed}.pt",
                   help="체크포인트 경로 템플릿. 예: data/checkpoint_eff_s{seed}.pt")
    p.add_argument("--episodes", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--render", action="store_true")
    p.add_argument("--hid", type=int, default=32, help="학습과 동일해야 함")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--success_at", type=int, default=500, help="성공판정 스텝 수")
    p.add_argument("--no_tta", action="store_true", help="TTA 비활성화(기본은 사용)")
    args = p.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip() != ""]
    evaluate_ensemble_tta(
        exp_name=args.name,
        seeds=seeds,
        ckpt_tpl=args.ckpt_tpl,
        num_episodes=args.episodes,
        seed=args.seed,
        render=args.render,
        hid=args.hid,
        device=args.device,
        success_at=args.success_at,
        use_tta=(not args.no_tta)
    )
