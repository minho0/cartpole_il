# mlp_ce_adam_train_50x10.py
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
import matplotlib.pyplot as plt
import os

# === 상태 정규화 함수 ===
SCALE = np.array([2.4, 3.0, 0.2095, 3.5], dtype=np.float32)

def normalize_obs(X_np: np.ndarray):
    """관측값을 SCALE로 나누고 [-2, 2] 범위로 클리핑합니다."""
    X = X_np / SCALE
    return np.clip(X, -2.0, 2.0).astype(np.float32)

# 1) 데이터 로드 및 정규화
def load_npz_and_normalize(path):
    d = np.load(path)
    X = normalize_obs(d["observations"])
    y = d["actions"].astype(np.int64)
    return torch.from_numpy(X), torch.from_numpy(y)

# 2) 네트워크: MLP (4 -> 32 -> 32 -> 2)
class MLP(nn.Module):
    def __init__(self, in_dim=4, hid=32, out_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid), nn.ReLU(),
            nn.Linear(hid, hid), nn.ReLU(),
            nn.Linear(hid, out_dim)
        )
    def forward(self, x): return self.net(x)

# <--- 변경: 파일 이름 관리를 위해 exp_name 인자 추가
def train(path, exp_name, epochs=50, batch=256, lr=1e-3, seed=0, device="cpu"):
    torch.manual_seed(seed); np.random.seed(seed)
    
    X, y = load_npz_and_normalize(path)
    X, y = X.to(device), y.to(device)

    model = MLP().to(device)
    loss_fn = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=lr)

    epoch_losses = []

    N = X.shape[0]
    idx = np.arange(N)
    for ep in range(1, epochs+1):
        np.random.shuffle(idx)
        total_loss = 0.0
        for i in range(0, N, batch):
            b = idx[i:i+batch]
            logits = model(X[b])
            loss = loss_fn(logits, y[b])
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item() * len(b)
        
        avg_loss = total_loss / N
        epoch_losses.append(avg_loss)
        print(f"[{exp_name} | seed {seed} | ep {ep:02d}] loss={avg_loss:.4f}")

    # <--- 변경: 파일 경로에 exp_name 포함
    save_path = f"data/checkpoint_{exp_name}_s{seed}.pt"
    torch.save(model.state_dict(), save_path)
    print(f"saved: {save_path}")

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), epoch_losses, marker='o', linestyle='-')
    # <--- 변경: 그래프 제목에 exp_name 포함
    plt.title(f'Training Loss Curve - {exp_name} (seed={seed})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    # <--- 변경: 그래프 파일 경로에 exp_name 포함
    plot_path = f"data/training_loss_curve_{exp_name}_s{seed}.png"
    plt.savefig(plot_path)
    print(f"Loss curve saved to: {plot_path}")

if __name__ == "__main__":
    # 'data' 폴더가 없으면 생성
    os.makedirs("data", exist_ok=True)
    
    # <--- 변경: 50x10 데이터셋으로 학습 실행
    # NOTE: 'data/demo_pid_50x10.npz' 파일이 존재해야 합니다.
    train(
        path="data/demo_pid_10x50.npz", 
        exp_name="10x50", 
        seed=0
    )