# train_il.py
import os, argparse, numpy as np
import torch, torch.nn as nn, torch.optim as optim
import matplotlib.pyplot as plt   # ← 추가

# === 정규화 ===
SCALE = np.array([2.4, 3.0, 0.2095, 3.5], dtype=np.float32)
def normalize_obs(X_np: np.ndarray):
    X = X_np / SCALE
    return np.clip(X, -2.0, 2.0).astype(np.float32)

# === 데이터 로드 (에피소드 단위 split) ===
def load_npz_split_by_episode(path, val_eps=2, seed=0):
    assert os.path.exists(path), f"NPZ not found: {path}"
    d = np.load(path)
    X = d["observations"].astype(np.float32)
    y = d["actions"].astype(np.int64)
    ep_len = d["ep_lengths"].astype(np.int32)
    E = len(ep_len); assert E >= max(1, val_eps)
    rng = np.random.default_rng(seed)
    order = rng.permutation(E); val_idx = set(order[:val_eps])
    xs_tr, ys_tr, xs_val, ys_val = [], [], [], []
    s = 0
    for e, L in enumerate(ep_len):
        segX, segY = X[s:s+L], y[s:s+L]; s += L
        (xs_val if e in val_idx else xs_tr).append(segX)
        (ys_val if e in val_idx else ys_tr).append(segY)
    Xtr = normalize_obs(np.concatenate(xs_tr, 0))
    ytr = np.concatenate(ys_tr, 0)
    Xva = normalize_obs(np.concatenate(xs_val, 0))
    yva = np.concatenate(ys_val, 0)
    print(f"[load] episodes={E}  train={len(ytr)}  val={len(yva)}")
    return (torch.from_numpy(Xtr), torch.from_numpy(ytr),
            torch.from_numpy(Xva), torch.from_numpy(yva))

# === 대칭 증강 (좌↔우) ===
def augment_sign_flip(X: torch.Tensor, y: torch.Tensor):
    Xf = X.clone()
    Xf[:, [0,1,2,3]] *= -1  # x, x_dot, theta, theta_dot 모두 부호 반전
    yf = 1 - y
    X_aug = torch.cat([X, Xf], dim=0)
    y_aug = torch.cat([y, yf], dim=0)
    return X_aug, y_aug

# === 어려운 상태 가중치 ===
def make_sample_weight(X: torch.Tensor, alpha=1.0):
    th = X[:, 2]; thd = X[:, 3]
    w = 1.0 + alpha * (th.abs() + 0.5 * thd.abs())
    return w.clamp(1.0, 3.0)

# === 모델 ===
class MLP(nn.Module):
    def __init__(self, in_dim=4, hid=32, out_dim=2, dropout=0.0):
        super().__init__()
        layers = [nn.Linear(in_dim, hid), nn.ReLU()]
        if dropout > 0: layers.append(nn.Dropout(p=dropout))
        layers += [nn.Linear(hid, hid), nn.ReLU()]
        if dropout > 0: layers.append(nn.Dropout(p=dropout))
        layers += [nn.Linear(hid, out_dim)]
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

def compute_class_weight(y):
    y_np = y.cpu().numpy()
    N = len(y_np); n0 = int((y_np == 0).sum()); n1 = N - n0
    w0 = N / (2 * max(n0, 1)); w1 = N / (2 * max(n1, 1))
    print(f"[class] action dist: 0={n0/N*100:.1f}%, 1={n1/N*100:.1f}%  -> weight=({w0:.3f},{w1:.3f})")
    return torch.tensor([w0, w1], dtype=torch.float32)

def train_one_seed(seed, args):
    torch.manual_seed(seed); np.random.seed(seed)
    Xtr, ytr, Xva, yva = load_npz_split_by_episode(args.data, val_eps=args.val_eps, seed=seed)
    device = torch.device(args.device)
    Xtr, ytr, Xva, yva = Xtr.to(device), ytr.to(device), Xva.to(device), yva.to(device)

    # 대칭 증강
    if args.augment:
        Xtr, ytr = augment_sign_flip(Xtr, ytr)
        print(f"[aug] sign-flip -> train={len(ytr)}")

    # 중요도 가중치
    sample_w_all = make_sample_weight(Xtr, alpha=args.alpha)
    model = MLP(hid=args.hid, dropout=args.dropout).to(device)
    class_w = compute_class_weight(ytr).to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-3)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=2, verbose=True, min_lr=1e-5)

    N = Xtr.shape[0]; idx = np.arange(N)
    best_val, best = float("inf"), None

    # === loss 저장용 리스트 추가 ===
    train_losses = []
    val_losses = []

    for ep in range(1, args.epochs+1):
        np.random.shuffle(idx)
        model.train(); total = 0.0
        for i in range(0, N, args.batch):
            b = idx[i:i+args.batch]
            logits = model(Xtr[b])
            ce = nn.functional.cross_entropy(logits, ytr[b], reduction='none',
                                             label_smoothing=0.05, weight=class_w)
            if args.imp_weight:
                loss = (ce * sample_w_all[b]).mean()
            else:
                loss = ce.mean()
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(b)
        tr_loss = total / N
        train_losses.append(tr_loss)  # ← 저장

        model.eval()
        with torch.no_grad():
            va_logits = model(Xva)
            va_loss = nn.functional.cross_entropy(va_logits, yva,
                                                  label_smoothing=0.05, weight=class_w)
        val_losses.append(va_loss.item())  # ← 저장

        sched.step(va_loss)
        if ep > args.epochs - 10 and opt.param_groups[0]['lr'] > 3e-4:
            for g in opt.param_groups: g['lr'] = max(g['lr'] * 0.5, 3e-4)

        print(f"[seed {seed} | {ep:02d}] train={tr_loss:.4f}  val={va_loss:.4f}  lr={opt.param_groups[0]['lr']:.2e}")
        if va_loss < best_val:
            best_val = va_loss
            best = {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}

    os.makedirs("data", exist_ok=True)
    ckpt = f"data/checkpoint_eff_s{seed}.pt"
    if best is not None:
        model.load_state_dict(best)
    torch.save(model.state_dict(), ckpt)
    print(f"[save] {ckpt}  (best val={best_val:.4f})")

    # === 학습 종료 후 loss curve 그리기 ===
    plt.figure(figsize=(10,5))
    plt.plot(range(1, args.epochs+1), train_losses, label='Train Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training Loss Curve (seed={seed})')
    plt.grid(True)
    plt.legend()
    plot_path = f"data/loss_curve_10x50_eff_s{seed}.png"
    plt.savefig(plot_path)
    print(f"[plot] train loss curve saved to {plot_path}")
    # plt.show()  # 필요시 주석 해제

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data/demo_pid_10x50.npz")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hid", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--val_eps", type=int, default=2)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--seeds", type=str, default="0,1,2")
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--imp_weight", action="store_true")
    ap.add_argument("--alpha", type=float, default=1.0)
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    for s in [int(x) for x in args.seeds.split(",") if x.strip()!=""]:
        train_one_seed(s, args)
