# collect_pid_10x50.py
import argparse
import numpy as np
import gymnasium as gym

def pid_expert(obs, Kp=50.0, Kd=5.0, Kv=1.0):
    """
    간단한 '전문가' 정책:
    obs = [x, x_dot, theta, theta_dot]
    u > 0 이면 오른쪽(1), 아니면 왼쪽(0)
    """
    x, x_dot, theta, theta_dot = obs
    u = Kp * theta + Kd * theta_dot + Kv * x_dot
    return 1 if u > 0 else 0

def collect(num_episodes=10, seed=0, max_steps=500, out="data/demo_pid_10x500.npz", render=False):
    env = gym.make("CartPole-v1", render_mode="human" if render else None)
    rng = np.random.default_rng(seed)

    observations, actions, ep_lengths = [], [], []
    for ep in range(num_episodes):
        obs, _ = env.reset(seed=int(rng.integers(0, 1_000_000)))
        steps = 0
        for t in range(max_steps):
            act = pid_expert(obs)
            observations.append(obs.copy())
            actions.append(act)
            obs, reward, terminated, truncated, _ = env.step(act)
            steps += 1
            if terminated or truncated:
                break
        ep_lengths.append(steps)
        print(f"[EP {ep+1:02d}] steps={steps}")

    env.close()

    observations = np.asarray(observations, dtype=np.float32)
    actions = np.asarray(actions, dtype=np.int64)
    ep_lengths = np.asarray(ep_lengths, dtype=np.int32)

    # 요약 통계
    total = len(actions)
    a0 = int((actions == 0).sum())
    a1 = total - a0
    print(f"Action distribution: 0={a0/total*100:.1f}%, 1={a1/total*100:.1f}%")

    np.savez(out, observations=observations, actions=actions, ep_lengths=ep_lengths)
    print(f"Saved demos to {out} | transitions={total} | episodes={num_episodes}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_steps", type=int, default=500)
    ap.add_argument("--out", type=str, default="data/demo_pid_10x500.npz")
    ap.add_argument("--render", action="store_true")
    args = ap.parse_args()
    collect(num_episodes=args.episodes, seed=args.seed, max_steps=args.max_steps, out=args.out, render=args.render)
