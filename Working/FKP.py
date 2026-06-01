import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import gamma
from collections import Counter
import os

# ============================================================
# 1️⃣ Fractional Components (GL Kernel + Adaptive Alpha)
# ============================================================

def gl_weights(alpha, r):
    return [gamma(alpha+1)/(gamma(k+1)*gamma(alpha-k+1)) for k in range(1, r+1)]

def fractional_velocity(series, t, dt, alpha, r):
    C = gl_weights(alpha, r)
    frac_term = sum(((-1)**k) * C[k-1] * series[t+1-k] for k in range(1, r+1))
    return - (1/(dt**alpha)) * frac_term


def golden_section_search(func, lo=0.0, hi=1.0, tol=1e-3, max_iter=30):
    gr = (np.sqrt(5) + 1) / 2
    c = hi - (hi-lo)/gr
    d = lo + (hi-lo)/gr

    for _ in range(max_iter):
        if abs(hi - lo) < tol:
            break
        if func(c) < func(d):
            hi = d
        else:
            lo = c

        c = hi - (hi-lo)/gr
        d = lo + (hi-lo)/gr

    return (hi + lo) / 2


# ============================================================
# 2️⃣ Fractional Kalman Filter (Option A)
# ============================================================

def fractional_kalman_filter(x, y, dt, r=20, R_val=3):

    # State:
    # [x, y, vx, vy]

    n = len(x)

    X = np.zeros((4, n))
    P = np.eye(4) * 100  # initial covariance

    # Measurement matrix
    H = np.array([[1,0,0,0],
                  [0,1,0,0]])

    R = np.eye(2) * R_val

    # Outputs
    pred_x = np.zeros(n)
    pred_y = np.zeros(n)
    alpha_list = np.zeros(n)

    # Initialize
    X[0,0] = x[0]
    X[1,0] = y[0]
    X[2,0] = 0
    X[3,0] = 0

    pred_x[0] = x[0]
    pred_y[0] = y[0]
    alpha_list[0] = 0.85

    for t in range(r, n-1):

        # ===== 1) ADAPTIVE α =====
        def err_alpha(alpha):
            frac_vx = fractional_velocity(x, t, dt, alpha, r)
            pred = x[t] + X[2,t] * dt + frac_vx * dt
            return abs(x[t+1] - pred)

        alpha = golden_section_search(err_alpha, 0.0, 1.0)
        alpha_list[t] = alpha

        # ===== 2) FRACTIONAL VELOCITY UPDATE =====
        frac_vx = fractional_velocity(x, t, dt, alpha, r)
        frac_vy = fractional_velocity(y, t, dt, alpha, r)

        # ===== 3) STATE TRANSITION (Option A) =====
        F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        # Predict velocities with fractional memory
        vx_pred = X[2,t] + frac_vx
        vy_pred = X[3,t] + frac_vy

        # Predict position
        x_pred = X[0,t] + X[2,t]*dt
        y_pred = X[1,t] + X[3,t]*dt

        X_pred = np.array([x_pred, y_pred, vx_pred, vy_pred])

        # ===== 4) AUTO ESTIMATE Q (Process Noise) =====
        Q = np.eye(4) * np.var([x_pred - x[t], y_pred - y[t]])

        # ===== 5) KALMAN PREDICT STEP =====
        P_pred = F @ P @ F.T + Q

        # ===== 6) MEASUREMENT UPDATE =====
        z = np.array([x[t+1], y[t+1]])

        y_tilde = z - (H @ X_pred)               # Innovation
        S = H @ P_pred @ H.T + R                 # Innovation covariance
        K = P_pred @ H.T @ np.linalg.inv(S)      # Kalman gain

        X[:,t+1] = X_pred + K @ y_tilde
        P = (np.eye(4) - K @ H) @ P_pred

        pred_x[t+1] = X[0,t+1]
        pred_y[t+1] = X[1,t+1]

    return pred_x, pred_y, alpha_list


# ============================================================
# 3️⃣ SAFE DATA LOADING
# ============================================================

def load_dataset(path):
    df = pd.read_csv(path, usecols=["timestamp", "x", "y"]).copy()
    df = df.sort_values("timestamp").dropna().drop_duplicates("timestamp").reset_index(drop=True)

    t = df["timestamp"].values.astype(float)
    t = t - t[0]

    diffs = np.diff(t)
    diffs = diffs[diffs > 0]

    dt = Counter(np.round(diffs, 2)).most_common(1)[0][0]

    n_samples = int(round(t[-1] / dt)) + 1

    t_uniform = np.linspace(0, t[-1], n_samples)
    x_uniform = np.interp(t_uniform, t, df["x"].values)
    y_uniform = np.interp(t_uniform, t, df["y"].values)

    return t_uniform, x_uniform, y_uniform, dt


# ============================================================
# 4️⃣ MAIN SCRIPT
# ============================================================

if __name__ == "__main__":

    dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "Dataset", "cluster_3_mean.csv"))

    t, x, y, dt = load_dataset(dataset_path)

    pred_x, pred_y, alpha_vals = fractional_kalman_filter(x, y, dt, r=20, R_val=3)

    # ACCURACY METRICS
    mse_x = np.nanmean((x - pred_x)**2)
    mse_y = np.nanmean((y - pred_y)**2)

    var_x = np.nanvar(x)
    var_y = np.nanvar(y)

    acc_x = max(0, 100 * (1 - mse_x / var_x))
    acc_y = max(0, 100 * (1 - mse_y / var_y))
    overall = (acc_x + acc_y) / 2

    print("\n========== FRACTIONAL KALMAN FILTER RESULTS ==========")
    print(f"MSE X: {mse_x:.4f} | Accuracy X: {acc_x:.2f}%")
    print(f"MSE Y: {mse_y:.4f} | Accuracy Y: {acc_y:.2f}%")
    print(f"Overall Accuracy: {overall:.2f}%")
    print("======================================================\n")

    # PLOTS
    plt.figure(figsize=(10,5))
    plt.plot(x, y, label="Actual", color="blue")
    plt.plot(pred_x, pred_y, label="FKF Predicted", color="red", linestyle="--")
    plt.legend()
    plt.title("Actual vs FKF Predicted Trajectory")
    plt.show()

    plt.figure(figsize=(10,4))
    plt.plot(alpha_vals)
    plt.title("Adaptive Fractional Order α")
    plt.show()
