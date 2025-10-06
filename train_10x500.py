# mlp_ce_adam_train_with_plot.py
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
import matplotlib.pyplot as plt

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

def train(path="cartpole_demos.npz", epochs=50, batch=256, lr=1e-3, seed=0, device="cpu"):
    torch.manual_seed(seed); np.random.seed(seed)
    
    X, y = load_npz_and_normalize(path)
    X, y = X.to(device), y.to(device)

    model = MLP().to(device)
    loss_fn = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=lr)

    # === loss 저장을 위한 리스트 추가 ===
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
        epoch_losses.append(avg_loss) # 매 epoch의 평균 loss 저장
        print(f"[seed {seed} | ep {ep:02d}] loss={avg_loss:.4f}")

    save_path = f"data/checkpoint_10x500.pt"
    torch.save(model.state_dict(), save_path)
    print(f"saved: {save_path}")

    # === 학습 종료 후 loss curve 그리기 ===
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), epoch_losses, marker='o', linestyle='-')
    plt.title(f'Training Loss Curve (seed={seed})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    # 그래프를 이미지 파일로 저장
    plot_path = f"data/training_loss_curve_10x500.png"
    plt.savefig(plot_path)
    print(f"Loss curve saved to: {plot_path}")
    # plt.show() # 주석 해제 시 그래프를 화면에 바로 표시

if __name__ == "__main__":
    train(path="data/demo_pid_10x500.npz", seed=0)