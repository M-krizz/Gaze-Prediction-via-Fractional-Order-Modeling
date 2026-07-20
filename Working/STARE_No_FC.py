"""
STARE_No_FC.py
--------------
Spatio-Temporal Attention (STARE-style) model to predict the NEXT gaze point (x, y)
from a sequence of past gaze points.

This version:
✅ Saves plots locally
✅ Prints actual vs predicted gaze points
✅ Saves comparison table to CSV
"""

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ============================================================
#                 DATASET
# ============================================================

class GazeSequenceDataset(Dataset):
    def __init__(self, csv_path: str, seq_len: int = 20):
        super().__init__()
        self.seq_len = seq_len

        df = pd.read_csv(csv_path)

        needed = {"participant", "timestamp", "x", "y"}
        if not needed.issubset(df.columns):
            raise ValueError(f"CSV must contain {needed}")

        df = df.dropna(subset=["participant", "timestamp", "x", "y"])

        # Global normalization
        self.x_min, self.x_max = df["x"].min(), df["x"].max()
        self.y_min, self.y_max = df["y"].min(), df["y"].max()

        self.x_range = self.x_max - self.x_min + 1e-8
        self.y_range = self.y_max - self.y_min + 1e-8

        df = df.sort_values(["participant", "timestamp"])

        self.inputs, self.targets = [], []

        for _, g in df.groupby("participant"):
            xs = (g["x"].values - self.x_min) / self.x_range
            ys = (g["y"].values - self.y_min) / self.y_range

            for i in range(len(xs) - seq_len):
                seq = np.stack([xs[i:i+seq_len], ys[i:i+seq_len]], axis=-1)
                tgt = np.array([xs[i+seq_len], ys[i+seq_len]])
                self.inputs.append(seq.astype(np.float32))
                self.targets.append(tgt.astype(np.float32))

        self.inputs = np.array(self.inputs)
        self.targets = np.array(self.targets)

        print(f"[Dataset] Loaded {len(self.inputs)} samples")

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return torch.from_numpy(self.inputs[idx]), torch.from_numpy(self.targets[idx])

    def denormalize(self, x_norm, y_norm) -> Tuple[float, float]:
        x = x_norm * self.x_range + self.x_min
        y = y_norm * self.y_range + self.y_min
        return float(x), float(y)


# ============================================================
#           POSITIONAL ENCODING
# ============================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


# ============================================================
#                 STARE MODEL
# ============================================================

class STAREGazePredictor(nn.Module):
    def __init__(self, d_model=128, nhead=4, num_layers=3, seq_len=20):
        super().__init__()
        self.input_proj = nn.Linear(2, d_model)
        self.pos_enc = PositionalEncoding(d_model, seq_len+10)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers)
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 2)
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.encoder(x)
        return self.fc(x[:, -1])


# ============================================================
#               TRAINING FUNCTIONS
# ============================================================

def train_epoch(model, loader, optimizer, device):
    model.train()
    loss_fn = nn.MSELoss()
    total = 0.0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        total += loss.item() * x.size(0)

    return total / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, dataset, device):
    model.eval()

    rows = []

    for x, y in loader:
        x = x.to(device)
        pred = model(x).cpu().numpy()
        y = y.numpy()

        for i in range(len(y)):
            tx, ty = dataset.denormalize(y[i,0], y[i,1])
            px, py = dataset.denormalize(pred[i,0], pred[i,1])
            err = np.sqrt((tx-px)**2 + (ty-py)**2)
            rows.append([tx, ty, px, py, err])

    return pd.DataFrame(rows, columns=[
        "x_actual", "y_actual", "x_pred", "y_pred", "error"
    ])


# ============================================================
#                 PLOT SAVE HELPER
# ============================================================

def save_plot(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


# ============================================================
#                       MAIN
# ============================================================

if __name__ == "__main__":

    base_dir = os.path.dirname(__file__)
    CSV_PATH = os.path.abspath(
        os.path.join(
            base_dir, "..", "trucate_files", "cluster_means",
            "cluster_0_mean_processed.csv"
        )
    )
    results_dir = os.path.abspath(os.path.join(base_dir, "results"))
    SEQ_LEN = 30
    EPOCHS = 10
    BATCH = 64
    LR = 1e-3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    dataset = GazeSequenceDataset(CSV_PATH, seq_len=SEQ_LEN)
    loader = DataLoader(dataset, batch_size=BATCH, shuffle=True)

    model = STAREGazePredictor(seq_len=SEQ_LEN).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # Training
    for ep in range(1, EPOCHS+1):
        loss = train_epoch(model, loader, optimizer, device)
        print(f"Epoch {ep}/{EPOCHS} | Loss: {loss:.6f}")

    # Evaluation
    results = evaluate(model, loader, dataset, device)

    print("\n===== ACTUAL vs PREDICTED (FIRST 10) =====")
    print(results.head(10).to_string(index=False))

    print("\n===== ERROR STATS =====")
    print(f"Mean Error   : {results['error'].mean():.4f}")
    print(f"Median Error : {results['error'].median():.4f}")
    print(f"Max Error    : {results['error'].max():.4f}")

    # Save table
    os.makedirs(os.path.join(results_dir, "tables"), exist_ok=True)
    results.to_csv(os.path.join(results_dir, "tables", "stare_no_fc_actual_vs_predicted.csv"), index=False)

    # Plot trajectory (sample)
    plt.figure(figsize=(6,6))
    plt.plot(results["x_actual"], results["y_actual"], label="Actual", alpha=0.7)
    plt.plot(results["x_pred"], results["y_pred"], label="Predicted", alpha=0.7)
    plt.legend()
    plt.title("STARE (No FC): Actual vs Predicted Trajectory")

    save_plot(os.path.join(results_dir, "plots", "trajectories", "stare_no_fc_trajectory.png"))

    print("\n✅ Results saved to results/")
