###############################################
# STARE + FRACTIONAL CALCULUS + TIMESTAMP
# PROPER TRAIN / VAL / TEST SPLIT
###############################################

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from math import gamma

torch.set_default_dtype(torch.float32)

# ============================================================
# 1. FRACTIONAL CALCULUS
# ============================================================

def gl_weights(alpha, r):
    return [gamma(alpha+1)/(gamma(k+1)*gamma(alpha-k+1)) for k in range(1, r+1)]

def fractional_derivative(series, dt, alpha=0.8, r=20):
    series = np.asarray(series, dtype=np.float32)
    series = np.nan_to_num(series, nan=np.nanmedian(series))
    out = np.zeros_like(series)

    if dt <= 0:
        dt = 1.0

    C = gl_weights(alpha, r)
    for t in range(r, len(series)):
        s = sum(((-1)**k)*C[k-1]*series[t-k] for k in range(1, r+1))
        out[t] = s / (dt**alpha)

    return out


# ============================================================
# 2. ROI TOKENIZER
# ============================================================

def coords_to_roi_ids(x, y, n_cols=8, n_rows=6):
    x = np.asarray(x)
    y = np.asarray(y)
    x_n = (x - x.min()) / (x.max() - x.min() + 1e-8)
    y_n = (y - y.min()) / (y.max() - y.min() + 1e-8)
    col = np.clip((x_n * n_cols).astype(int), 0, n_cols-1)
    row = np.clip((y_n * n_rows).astype(int), 0, n_rows-1)
    return (row * n_cols + col).astype(int), n_cols*n_rows


# ============================================================
# 3. DATASET (TEMPORAL SAFE)
# ============================================================

class GazeFCDataset(Dataset):
    """
    SAFE dataset for STARE + Fractional Calculus

    Returns:
      roi_seq : (seq_len,)
      fc_seq  : (seq_len, 4) -> [Dax, Day, alpha, t_norm]
      fut_xy  : (H, 2)
    """

    def __init__(self, csv_path, seq_len=30, future_steps=5,
                 alpha=0.8, r=20, n_cols=8, n_rows=6):

        self.seq_len = seq_len
        self.future_steps = future_steps

        # ----------------------------------------------------
        # Load + clean
        # ----------------------------------------------------
        df = pd.read_csv(csv_path)[["timestamp", "x", "y"]].copy()

        for c in ["timestamp", "x", "y"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        df = df.replace([np.inf, -np.inf], np.nan)
        df["timestamp"] = df["timestamp"].ffill().bfill()
        df["x"] = df["x"].fillna(df["x"].median())
        df["y"] = df["y"].fillna(df["y"].median())

        df = df.sort_values("timestamp").reset_index(drop=True)

        t_raw = df["timestamp"].values.astype(np.float32)
        x_raw = df["x"].values.astype(np.float32)
        y_raw = df["y"].values.astype(np.float32)

        # ----------------------------------------------------
        # SAFE NORMALIZATION
        # ----------------------------------------------------
        self.x_min, self.x_max = float(x_raw.min()), float(x_raw.max())
        self.y_min, self.y_max = float(y_raw.min()), float(y_raw.max())
        self.t_min, self.t_max = float(t_raw.min()), float(t_raw.max())

        x_range = self.x_max - self.x_min
        y_range = self.y_max - self.y_min
        t_range = self.t_max - self.t_min

        if x_range < 1e-6:
            x = np.zeros_like(x_raw)
        else:
            x = (x_raw - self.x_min) / (x_range + 1e-8)

        if y_range < 1e-6:
            y = np.zeros_like(y_raw)
        else:
            y = (y_raw - self.y_min) / (y_range + 1e-8)

        if t_range < 1e-6:
            t = np.zeros_like(t_raw)
        else:
            t = (t_raw - self.t_min) / (t_range + 1e-8)

        # ----------------------------------------------------
        # SAFE dt ESTIMATION
        # ----------------------------------------------------
        diffs = np.diff(t_raw)
        diffs = diffs[diffs > 0]

        if len(diffs) == 0:
            dt = 1.0
        else:
            dt = float(np.median(diffs))

        if dt <= 0 or np.isnan(dt) or np.isinf(dt):
            dt = 1.0

        # ----------------------------------------------------
        # FRACTIONAL DERIVATIVES (SAFE)
        # ----------------------------------------------------
        Dax = fractional_derivative(x, dt, alpha=alpha, r=r)
        Day = fractional_derivative(y, dt, alpha=alpha, r=r)

        Dax = np.nan_to_num(Dax, nan=0.0, posinf=0.0, neginf=0.0)
        Day = np.nan_to_num(Day, nan=0.0, posinf=0.0, neginf=0.0)

        # ----------------------------------------------------
        # ROI TOKENIZATION (SAFE)
        # ----------------------------------------------------
        roi_ids, roi_vocab = coords_to_roi_ids(
            x_raw, y_raw, n_cols=n_cols, n_rows=n_rows
        )
        self.roi_vocab = int(roi_vocab)

        # ----------------------------------------------------
        # FC FEATURE STACK
        # ----------------------------------------------------
        alpha_vec = np.full_like(x, alpha, dtype=np.float32)
        fc_feats = np.stack([Dax, Day, alpha_vec, t], axis=-1)

        # Normalize derivative channels only
        for ch in [0, 1]:
            mu = fc_feats[:, ch].mean()
            std = fc_feats[:, ch].std()
            if std < 1e-6:
                fc_feats[:, ch] = 0.0
            else:
                fc_feats[:, ch] = (fc_feats[:, ch] - mu) / (std + 1e-6)

        fc_feats = np.nan_to_num(fc_feats, nan=0.0, posinf=0.0, neginf=0.0)

        # ----------------------------------------------------
        # BUILD SEQUENCES (TEMPORALLY SAFE)
        # ----------------------------------------------------
        self.roi_seqs = []
        self.fc_seqs = []
        self.future_xy = []

        T = len(x)

        for end in range(seq_len, T - future_steps):
            start = end - seq_len

            self.roi_seqs.append(roi_ids[start:end])
            self.fc_seqs.append(fc_feats[start:end])

            fx = x[end:end + future_steps]
            fy = y[end:end + future_steps]
            self.future_xy.append(np.stack([fx, fy], axis=-1))

        self.roi_seqs = np.array(self.roi_seqs, dtype=np.int64)
        self.fc_seqs = np.array(self.fc_seqs, dtype=np.float32)
        self.future_xy = np.array(self.future_xy, dtype=np.float32)

        print(
            f"\n[SAFE DATASET BUILT] "
            f"samples={len(self.roi_seqs)}, "
            f"seq_len={seq_len}, future_steps={future_steps}"
        )

    def __len__(self):
        return len(self.roi_seqs)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.roi_seqs[idx], dtype=torch.long),
            torch.tensor(self.fc_seqs[idx], dtype=torch.float32),
            torch.tensor(self.future_xy[idx], dtype=torch.float32),
        )



# ============================================================
# 4. MODEL
# ============================================================

class STARE_FC(nn.Module):
    def __init__(self, roi_vocab, model_dim=128, future_steps=5):
        super().__init__()
        self.future_steps = future_steps
        self.roi_emb = nn.Embedding(roi_vocab, 64)
        self.fc_emb = nn.Linear(4, 64)

        self.fuse = nn.Sequential(
            nn.Linear(128, model_dim),
            nn.LayerNorm(model_dim),
            nn.ReLU()
        )

        enc = nn.TransformerEncoderLayer(model_dim, 4, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, 3)
        self.head = nn.Linear(model_dim, future_steps*2)

    def forward(self, roi, fc):
        x = torch.cat([self.roi_emb(roi), self.fc_emb(fc)], dim=-1)
        x = self.encoder(self.fuse(x))
        out = self.head(x[:,-1])
        return out.view(-1, self.future_steps, 2)


# ============================================================
# 5. TRAIN / VAL / TEST
# ============================================================

def train_and_evaluate(csv_path):
    dataset = GazeFCDataset(csv_path)
    N = len(dataset)

    n_train = int(0.7*N)
    n_val   = int(0.15*N)

    train_set = torch.utils.data.Subset(dataset, range(0, n_train))
    val_set   = torch.utils.data.Subset(dataset, range(n_train, n_train+n_val))
    test_set  = torch.utils.data.Subset(dataset, range(n_train+n_val, N))

    train_loader = DataLoader(train_set, batch_size=32, shuffle=False)
    val_loader   = DataLoader(val_set, batch_size=32, shuffle=False)
    test_loader  = DataLoader(test_set, batch_size=32, shuffle=False)

    model = STARE_FC(dataset.roi_vocab)
    opt = torch.optim.Adam(model.parameters(), 1e-4)
    loss_fn = nn.MSELoss()

    print("\n🔥 Training (Leakage-Free)\n")

    for epoch in range(10):
        model.train()
        train_loss = 0

        for r,f,y in train_loader:
            pred = model(r,f)
            loss = loss_fn(pred,y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item()

        model.eval()
        val_loss = sum(loss_fn(model(r,f),y).item() for r,f,y in val_loader)

        print(f"Epoch {epoch+1} | Train={train_loss/len(train_loader):.4f} | Val={val_loss/len(val_loader):.4f}")

    # -------- TEST EVALUATION --------
    model.eval()
    errors = []

    for r,f,y in test_loader:
        p = model(r,f)
        err = torch.sqrt(((p-y)**2).sum(dim=-1))
        errors.append(err.mean().item())

    print("\n📊 TEST PERFORMANCE")
    print(f"Mean Euclidean Error: {np.mean(errors):.4f} (normalized units)")

    return model


# ============================================================
# 6. MAIN
# ============================================================

if __name__ == "__main__":
    train_and_evaluate("D:\\Fractional Calculus\\Working\\Dataset\\cluster_0_mean.csv")
