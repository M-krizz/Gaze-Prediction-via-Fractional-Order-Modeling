import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import gamma
from collections import Counter
import os

# ============================================================
#  GRÜNWALD–LETNIKOV FRACTIONAL CALCULUS PREDICTOR COMPONENTS
# ============================================================

def gl_weights(alpha, r):
    return [gamma(alpha+1)/(gamma(k+1)*gamma(alpha-k+1)) for k in range(1, r+1)]


def fractional_predict(series, t, dt, alpha, r):
    v = series[t] - series[t-1]    
    C = gl_weights(alpha, r)

    frac_term = sum(
        ((-1)**k) * C[k-1] * series[t+1-k]
        for k in range(1, r+1)
    )

    return series[t] + v - (1/(dt**alpha)) * frac_term


def golden_section_search(func, lo=0.0, hi=1.0, tol=1e-3, max_iter=30):
    gr = (np.sqrt(5) + 1) / 2
    c = hi - (hi - lo) / gr
    d = lo + (hi - lo) / gr

    for _ in range(max_iter):
        if abs(hi - lo) < tol:
            break
        if func(c) < func(d):
            hi = d
        else:
            lo = c
        c = hi - (hi - lo) / gr
        d = lo + (hi - lo) / gr

    return (hi + lo) / 2


def adaptive_fc_predict(series, dt, r=20):
    T = len(series)
    preds = [np.nan] * T
    alphas = [0.85] * T

    for t in range(r, T-1):

        def err_fn(alpha):
            return abs(series[t+1] - fractional_predict(series, t, dt, alpha, r))

        alpha_opt = golden_section_search(err_fn, 0.0, 1.0)
        alphas[t] = alpha_opt
        preds[t+1] = fractional_predict(series, t, dt, alpha_opt, r)

    return np.array(preds), np.array(alphas)


# ============================================================
#                 SAFE, ROBUST DATA LOADER
# ============================================================

def load_dataset(path):
    df = pd.read_csv(path, usecols=["timestamp", "x", "y"]).copy()
    df = df.sort_values("timestamp").dropna().drop_duplicates("timestamp").reset_index(drop=True)

    t = df["timestamp"].values.astype(float)
    t = t - t[0]

    if len(t) < 2:
        raise ValueError("Not enough timestamp values for prediction.")

    diffs = np.diff(t)
    diffs = diffs[diffs > 0]

    if len(diffs) == 0:
        raise ValueError("All timestamps identical — cannot compute dt")

    try:
        dt = Counter(np.round(diffs, 2)).most_common(1)[0][0]
    except:
        dt = np.median(diffs)

    if dt <= 0 or np.isnan(dt):
        dt = np.median(diffs)

    n_samples = int(round(t[-1] / dt)) + 1
    n_samples = max(n_samples, 2)

    t_uniform = np.linspace(0, t[-1], n_samples)
    x_uniform = np.interp(t_uniform, t, df["x"].values)
    y_uniform = np.interp(t_uniform, t, df["y"].values)

    return t_uniform, x_uniform, y_uniform, dt


# ============================================================
#               PLOT SAVE HELPER
# ============================================================

def save_plot(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


# ============================================================
#                       MAIN SCRIPT
# ============================================================

if __name__ == "__main__":

    dataset_path = "Dataset/cluster_3_mean.csv"   # safer path style

    t, x, y, dt = load_dataset(dataset_path)

    x_pred, x_alpha = adaptive_fc_predict(x, dt, r=20)
    y_pred, y_alpha = adaptive_fc_predict(y, dt, r=20)

    # ERROR (MSE)
    mse_x = np.nanmean((x - x_pred)**2)
    mse_y = np.nanmean((y - y_pred)**2)

    # NORMALIZED ACCURACY
    var_x = np.nanvar(x)
    var_y = np.nanvar(y)

    acc_x = max(0, 100 * (1 - mse_x / var_x))
    acc_y = max(0, 100 * (1 - mse_y / var_y))

    overall_acc = (acc_x + acc_y) / 2

    print("\n========== FRACTIONAL CALCULUS PREDICTION RESULTS ==========")
    print(f"MSE X: {mse_x:.4f} | Accuracy X: {acc_x:.2f}%")
    print(f"MSE Y: {mse_y:.4f} | Accuracy Y: {acc_y:.2f}%")
    print(f"Overall Accuracy: {overall_acc:.2f}%")
    print("=============================================================\n")

    # ================= TRAJECTORY PLOT =================
    plt.figure(figsize=(10,5))
    plt.plot(x, y, label="Actual", color="blue")
    plt.plot(x_pred, y_pred, label="Predicted", color="red", linestyle="--")
    plt.legend()
    plt.title("Actual vs Predicted Trajectory (Fractional Calculus)")
    plt.xlabel("X")
    plt.ylabel("Y")

    save_plot("results/plots/trajectories/adaptive_fc_trajectory.png")

    # ================= ALPHA PLOT =================
    plt.figure(figsize=(10,4))
    plt.plot(x_alpha, label="Alpha (X)")
    plt.plot(y_alpha, label="Alpha (Y)")
    plt.legend()
    plt.title("Adaptive Fractional Order α Over Time")

    save_plot("results/plots/alpha/adaptive_fc_alpha.png")

    print("✅ Plots saved in results/plots/")
