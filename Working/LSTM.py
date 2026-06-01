###############################################
#  STARE + FRACTIONAL CALCULUS + TIMESTAMP
#  with LSTM DECODER for Multi-step Gaze Prediction
###############################################

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from math import gamma
import matplotlib.pyplot as plt
import os

torch.set_default_dtype(torch.float32)


############################################################
# 1. FRACTIONAL CALCULUS UTILITIES
############################################################

def gl_weights(alpha, r):
    return [gamma(alpha+1)/(gamma(k+1)*gamma(alpha-k+1)) for k in range(1, r+1)]


def fractional_derivative(series, dt, alpha=0.8, r=20):
    series = np.asarray(series, dtype=np.float32)
    series = np.nan_to_num(
        series,
        nan=np.nanmedian(series),
        posinf=np.nanmedian(series),
        neginf=np.nanmedian(series),
    )

    T = len(series)
    fd = np.zeros(T, dtype=np.float32)

    if dt <= 0 or np.isnan(dt) or np.isinf(dt):
        dt = 1.0

    C = gl_weights(alpha, r)

    for t in range(r, T):
        s = 0.0
        for k in range(1, r + 1):
            s += ((-1) ** k) * C[k - 1] * series[t - k]

        val = s / (dt ** alpha)
        if np.isnan(val) or np.isinf(val):
            val = 0.0

        fd[t] = val

    return fd


############################################################
# 2. ROI TOKENIZER
############################################################

def coords_to_roi_ids(x, y, n_cols=8, n_rows=6):
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)

    x = np.nan_to_num(x, nan=np.nanmedian(x), posinf=np.nanmedian(x), neginf=np.nanmedian(x))
    y = np.nan_to_num(y, nan=np.nanmedian(y), posinf=np.nanmedian(y), neginf=np.nanmedian(y))

    x_min, x_max = float(x.min()), float(x.max())
    y_min, y_max = float(y.min()), float(y.max())

    x_range = x_max - x_min
    y_range = y_max - y_min

    x_norm = np.zeros_like(x) if x_range < 1e-8 else (x - x_min) / (x_range + 1e-8)
    y_norm = np.zeros_like(y) if y_range < 1e-8 else (y - y_min) / (y_range + 1e-8)

    col = np.clip((x_norm * n_cols).astype(int), 0, n_cols - 1)
    row = np.clip((y_norm * n_rows).astype(int), 0, n_rows - 1)

    roi_ids = row * n_cols + col
    return roi_ids.astype(np.int64), n_cols * n_rows


############################################################
# 3. DATASET CLASS WITH GAZE & TIMESTAMP NORMALIZATION
############################################################

class GazeFCDataset(Dataset):
    """
    Outputs:
       roi_seq   : (seq_len,)
       fc_feats  : (seq_len, 4)  → [Dαx, Dαy, α, t_norm]
       future_xy : (future_steps, 2)  → normalized gaze in [0,1]
    """

    def __init__(self, csv_path, seq_len=30, future_steps=5,
                 alpha=0.8, r=20, n_cols=8, n_rows=6):

        df = pd.read_csv(csv_path)[["timestamp", "x", "y"]].copy()

        # Clean numeric
        for col in ["timestamp", "x", "y"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)

        df["timestamp"].ffill(inplace=True)
        df["timestamp"].bfill(inplace=True)
        df["x"].fillna(df["x"].median(), inplace=True)
        df["y"].fillna(df["y"].median(), inplace=True)

        df = df.sort_values("timestamp").reset_index(drop=True)

        # RAW
        t_raw = df["timestamp"].values.astype(np.float32)
        x_raw = df["x"].values.astype(np.float32)
        y_raw = df["y"].values.astype(np.float32)

        # Save min/max for de-normalization
        self.x_min, self.x_max = float(x_raw.min()), float(x_raw.max())
        self.y_min, self.y_max = float(y_raw.min()), float(y_raw.max())

        # Normalize gaze to [0,1]
        x = (x_raw - self.x_min) / (self.x_max - self.x_min + 1e-6)
        y = (y_raw - self.y_min) / (self.y_max - self.y_min + 1e-6)

        # Normalize timestamp to [0,1]
        self.t_min, self.t_max = float(t_raw.min()), float(t_raw.max())
        t = (t_raw - self.t_min) / (self.t_max - self.t_min + 1e-6)

        # dt estimation on raw timestamps
        diffs = np.diff(t_raw)
        diffs = diffs[diffs > 0]
        dt = float(np.median(diffs)) if len(diffs) > 0 else 1.0
        if dt <= 0 or np.isnan(dt):
            dt = 1.0

        # Fractional derivatives on normalized gaze
        Dax = fractional_derivative(x, dt, alpha=alpha, r=r)
        Day = fractional_derivative(y, dt, alpha=alpha, r=r)

        # ROI IDs from raw gaze
        roi_ids, roi_vocab = coords_to_roi_ids(x_raw, y_raw, n_cols, n_rows)
        self.roi_vocab = int(roi_vocab)

        # FC features: [Dαx, Dαy, α, t_norm]
        alpha_vec = np.full_like(x, alpha)
        fc_feats = np.stack([Dax, Day, alpha_vec, t], axis=-1)
        fc_feats = np.nan_to_num(fc_feats)

        # Normalize FD channels (0 & 1)
        for ch in [0, 1]:
            mu, sigma = fc_feats[:, ch].mean(), fc_feats[:, ch].std() + 1e-6
            fc_feats[:, ch] = (fc_feats[:, ch] - mu) / sigma

        # Build sequences
        self.seq_len = seq_len
        self.future_steps = future_steps
        self.roi_seqs = []
        self.fc_seqs = []
        self.future_xy = []

        T = len(x)
        for end in range(seq_len, T - future_steps):
            start = end - seq_len

            roi_seq = roi_ids[start:end]
            fc_seq = fc_feats[start:end]

            fx = x[end:end + future_steps]
            fy = y[end:end + future_steps]
            fut_xy = np.stack([fx, fy], axis=-1)  # (H,2)

            self.roi_seqs.append(roi_seq)
            self.fc_seqs.append(fc_seq)
            self.future_xy.append(fut_xy)

        self.roi_seqs = np.array(self.roi_seqs, dtype=np.int64)
        self.fc_seqs = np.array(self.fc_seqs, dtype=np.float32)
        self.future_xy = np.array(self.future_xy, dtype=np.float32)

        print(f"\n[Dataset Built] {self.roi_seqs.shape[0]} samples | seq_len={seq_len}, future_steps={future_steps}")

    def __len__(self):
        return len(self.roi_seqs)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.roi_seqs[idx], dtype=torch.long),
            torch.tensor(self.fc_seqs[idx], dtype=torch.float32),
            torch.tensor(self.future_xy[idx], dtype=torch.float32),
        )


############################################################
# 4. STARE + FC ENCODER + LSTM DECODER
############################################################

class STARE_FC_LSTMHybrid(nn.Module):
    """
    Encoder: Transformer over (ROI + FC)
    Decoder: LSTM over future gaze points (teacher forcing in training)
    """

    def __init__(self, roi_vocab, roi_dim=64, fc_dim=4,
                 model_dim=128, num_layers=4, future_steps=5):
        super().__init__()

        self.future_steps = future_steps
        self.model_dim = model_dim

        # Encoder parts
        self.roi_embed = nn.Embedding(roi_vocab, roi_dim)
        self.fc_embed = nn.Linear(fc_dim, roi_dim)

        self.combine = nn.Sequential(
            nn.Linear(roi_dim * 2, model_dim),
            nn.LayerNorm(model_dim),
            nn.ReLU(),
        )

        enc_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=4,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # Decoder: LSTM over future gaze (2D)
        self.decoder = nn.LSTM(
            input_size=2,
            hidden_size=model_dim,
            num_layers=1,
            batch_first=True,
        )

        self.dec_out = nn.Linear(model_dim, 2)

    def encode(self, roi_seq, fc_feats):
        """
        roi_seq: (B,T)
        fc_feats: (B,T,4)
        returns: context (B,model_dim)
        """
        roi_emb = self.roi_embed(roi_seq)        # (B,T,roi_dim)
        fc_emb = self.fc_embed(fc_feats)         # (B,T,roi_dim)
        fused = torch.cat([roi_emb, fc_emb], dim=-1)
        fused = self.combine(fused)              # (B,T,model_dim)
        enc_out = self.encoder(fused)            # (B,T,model_dim)
        context = enc_out[:, -1, :]              # (B,model_dim) last token
        return context

    def forward(self, roi_seq, fc_feats, fut_xy):
        """
        Training forward with teacher forcing.
        roi_seq: (B,T)
        fc_feats: (B,T,4)
        fut_xy: (B,H,2) normalized future gaze (ground truth)
        returns: (B,H,2) predicted future gaze (normalized)
        """
        B, H, _ = fut_xy.shape
        device = roi_seq.device

        context = self.encode(roi_seq, fc_feats)              # (B,model_dim)

        # Initial hidden state from context
        h0 = context.unsqueeze(0)                             # (1,B,model_dim)
        c0 = torch.zeros_like(h0, device=device)              # (1,B,model_dim)

        # Teacher forcing: shift true sequence by 1, prepend zeros
        # Input to decoder at t=0 is [0,0]; at t>0 is fut_xy[:,t-1,:]
        dec_inputs = torch.zeros(B, H, 2, device=device)      # (B,H,2)
        dec_inputs[:, 1:, :] = fut_xy[:, :-1, :]              # shift right

        dec_out, _ = self.decoder(dec_inputs, (h0, c0))       # (B,H,model_dim)
        pred_xy = self.dec_out(dec_out)                       # (B,H,2)

        return pred_xy

    @torch.no_grad()
    def predict_future(self, roi_seq, fc_feats, future_steps):
        """
        Inference: autoregressive decoder without teacher forcing.
        roi_seq: (1,T)
        fc_feats: (1,T,4)
        returns: (future_steps,2)
        """
        self.eval()
        device = roi_seq.device
        context = self.encode(roi_seq, fc_feats)          # (1,model_dim)
        h = context.unsqueeze(0)                          # (1,1,model_dim)
        c = torch.zeros_like(h, device=device)

        preds = []
        dec_input = torch.zeros(1, 1, 2, device=device)   # start with zeros

        for _ in range(future_steps):
            out, (h, c) = self.decoder(dec_input, (h, c))  # (1,1,model_dim)
            gaze = self.dec_out(out[:, -1, :])             # (1,2)
            preds.append(gaze)
            dec_input = gaze.unsqueeze(1)                  # feed back

        return torch.cat(preds, dim=0)                    # (H,2)


############################################################
# 5. TRAINING LOOP
############################################################

def train_model(csv_path=os.path.abspath(os.path.join(os.path.dirname(__file__), 'Dataset', 'cluster_0_mean.csv')),
                seq_len=30, future_steps=5,
                epochs=10, batch_size=32, lr=1e-4):

    dataset = GazeFCDataset(csv_path, seq_len, future_steps)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = STARE_FC_LSTMHybrid(dataset.roi_vocab, future_steps=future_steps)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    print("\n🔥 Training Started (STARE + FC + LSTM)...\n")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for roi_seq, fc_feats, fut_xy in loader:
            pred_xy = model(roi_seq, fc_feats, fut_xy)   # (B,H,2)

            loss = loss_fn(pred_xy, fut_xy)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} | Loss={epoch_loss/len(loader):.6f}")

    torch.save(model.state_dict(), "stare_fc_lstm_multistep.pth")
    print("\n💾 Model saved → stare_fc_lstm_multistep.pth")

    return model, dataset


############################################################
# 6. VISUALIZATION + ACCURACY
############################################################

def visualize_future_prediction(model, dataset, idx=100):
    model.eval()

    roi_seq, fc_feats, fut_norm = dataset[idx]

    roi_seq = roi_seq.unsqueeze(0)      # (1,T)
    fc_feats = fc_feats.unsqueeze(0)    # (1,T,4)

    H = fut_norm.shape[0]

    # Predict future (normalized) autoregressively
    pred_norm = model.predict_future(roi_seq, fc_feats, future_steps=H).cpu().numpy()  # (H,2)

    # De-normalize predictions
    pred_xy = np.zeros_like(pred_norm)
    pred_xy[:, 0] = pred_norm[:, 0] * (dataset.x_max - dataset.x_min) + dataset.x_min
    pred_xy[:, 1] = pred_norm[:, 1] * (dataset.y_max - dataset.y_min) + dataset.y_min

    # De-normalize ground truth
    fut_real = np.zeros_like(fut_norm)
    fut_real[:, 0] = fut_norm[:, 0].numpy() * (dataset.x_max - dataset.x_min) + dataset.x_min
    fut_real[:, 1] = fut_norm[:, 1].numpy() * (dataset.y_max - dataset.y_min) + dataset.y_min

    print("\n🔥 Predicted Future (Real Scale):\n", pred_xy)
    print("\n🎯 Actual Future (Real Scale):\n", fut_real)

    # Accuracy metrics
    errors = np.sqrt(np.sum((pred_xy - fut_real) ** 2, axis=1))  # (H,)
    max_dist = np.sqrt((dataset.x_max - dataset.x_min) ** 2 +
                       (dataset.y_max - dataset.y_min) ** 2)

    acc_steps = 100 * (1 - errors / max_dist)
    acc_overall = acc_steps.mean()

    print("\n📏 Prediction Accuracy per step (%):", acc_steps)
    print(f"\n🔥 Overall Multi-step Accuracy: {acc_overall:.2f}%")

    # Plot
    plt.figure(figsize=(7, 6))
    plt.plot(fut_real[:, 0], fut_real[:, 1], "-o", label="Actual", color="blue")
    plt.plot(pred_xy[:, 0], pred_xy[:, 1], "-o", label="Predicted", color="red")
    plt.title(f"Future Gaze Prediction (LSTM Decoder)\nAccuracy = {acc_overall:.2f}%")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True)
    plt.show()


############################################################
# 7. MAIN
############################################################

if __name__ == "__main__":
    model, dataset = train_model(
        csv_path=os.path.abspath(os.path.join(os.path.dirname(__file__), 'Dataset', 'cluster_0_mean.csv')),
        seq_len=30,
        future_steps=5,
        epochs=10,
        batch_size=32,
        lr=1e-4,
    )

    visualize_future_prediction(model, dataset, idx=100)
