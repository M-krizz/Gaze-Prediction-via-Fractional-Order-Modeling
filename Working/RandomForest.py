import numpy as np
import pandas as pd
from math import gamma
import os

# =========================
# Fractional Calculus (GL)
# =========================
def gl_weights(alpha, r):
    return [gamma(alpha+1)/(gamma(k+1)*gamma(alpha-k+1)) for k in range(1, r+1)]

def fractional_derivative(series, dt, alpha=0.8, r=20):
    series = np.asarray(series, dtype=np.float32)
    series = np.nan_to_num(series, nan=np.nanmedian(series), posinf=np.nanmedian(series), neginf=np.nanmedian(series))

    T = len(series)
    out = np.zeros(T, dtype=np.float32)

    if dt <= 0 or np.isnan(dt) or np.isinf(dt):
        dt = 1.0

    C = gl_weights(alpha, r)
    for t in range(r, T):
        s = 0.0
        for k in range(1, r+1):
            s += ((-1)**k) * C[k-1] * series[t-k]
        val = s / (dt**alpha)
        out[t] = 0.0 if (np.isnan(val) or np.isinf(val)) else val

    return out

# =========================
# ROI tokenization
# =========================
def coords_to_roi_ids(x, y, n_cols=8, n_rows=6):
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    x = np.nan_to_num(x, nan=np.nanmedian(x))
    y = np.nan_to_num(y, nan=np.nanmedian(y))

    xr = float(x.max() - x.min())
    yr = float(y.max() - y.min())

    x_norm = np.zeros_like(x) if xr < 1e-8 else (x - x.min()) / (xr + 1e-8)
    y_norm = np.zeros_like(y) if yr < 1e-8 else (y - y.min()) / (yr + 1e-8)

    col = np.clip((x_norm * n_cols).astype(int), 0, n_cols - 1)
    row = np.clip((y_norm * n_rows).astype(int), 0, n_rows - 1)
    return (row * n_cols + col).astype(np.int64)

# =========================
# Fixation/Saccade detection (I-VT simple)
# =========================
def ivt_fixation_flags(speed, speed_thresh=80.0):
    """
    speed_thresh in pixels/sec (tune it)
    fixation if speed < threshold
    """
    return (speed < speed_thresh).astype(np.int8)

def compute_fixation_duration(flags, dt_series):
    """
    For each sample i, fixation_duration[i] = cumulative duration
    of current fixation segment (else 0 for saccade).
    """
    dur = np.zeros_like(flags, dtype=np.float32)
    cur = 0.0
    for i in range(len(flags)):
        if flags[i] == 1:
            cur += dt_series[i]
            dur[i] = cur
        else:
            cur = 0.0
            dur[i] = 0.0
    return dur

# =========================
# Main feature builder
# =========================
def build_features(csv_path,
                   roi_cols=8, roi_rows=6,
                   speed_thresh=80.0,
                   fc_alpha=0.8, fc_r=20,
                   rolling_window=10):

    df = pd.read_csv(csv_path)

    # Use your known column order; keep what exists
    # Expect at least x,y,timestamp
    needed = ["x", "y", "timestamp"]
    for c in needed:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")

    # Clean & sort
    df = df.copy()
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan)
    df["timestamp"] = df["timestamp"].ffill().bfill()
    df["x"] = df["x"].fillna(df["x"].median())
    df["y"] = df["y"].fillna(df["y"].median())

    df = df.sort_values("timestamp").reset_index(drop=True)

    t = df["timestamp"].values.astype(np.float32)
    x = df["x"].values.astype(np.float32)
    y = df["y"].values.astype(np.float32)

    # dt (seconds) — if your timestamp is in ms, convert:
    # If your timestamps look huge and dt ~ 16/33, it’s probably ms.
    # We’ll auto-detect: if median diff > 10, assume milliseconds.
    diffs = np.diff(t)
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        raise ValueError("Timestamps not increasing.")

    dt_med = float(np.median(diffs))
    if dt_med > 10.0:  # likely ms
        t_sec = (t - t[0]) / 1000.0
    else:              # likely already sec
        t_sec = (t - t[0])

    dt = np.diff(t_sec, prepend=t_sec[0])
    dt[0] = np.median(dt[1:]) if len(dt) > 1 else 1.0
    dt = np.clip(dt, 1e-6, None)

    # Basic kinematics
    dx = np.diff(x, prepend=x[0])
    dy = np.diff(y, prepend=y[0])
    disp = np.sqrt(dx**2 + dy**2)

    speed = disp / dt                      # px/sec
    accel = np.diff(speed, prepend=speed[0]) / dt
    direction = np.arctan2(dy, dx)         # radians

    # ROI features
    roi = coords_to_roi_ids(x, y, n_cols=roi_cols, n_rows=roi_rows)
    roi_prev = np.roll(roi, 1)
    roi_prev[0] = roi[0]
    roi_change = (roi != roi_prev).astype(np.int8)

    # Dwell time in same ROI (cumulative)
    dwell = np.zeros_like(dt, dtype=np.float32)
    cur = 0.0
    for i in range(len(roi)):
        if i == 0 or roi[i] == roi[i-1]:
            cur += dt[i]
        else:
            cur = dt[i]
        dwell[i] = cur

    # Fixation / saccade flags
    fix_flag = ivt_fixation_flags(speed, speed_thresh=speed_thresh)  # 1=fixation
    fix_dur = compute_fixation_duration(fix_flag, dt)

    # Saccade amplitude (only when not fixation)
    sacc_amp = np.where(fix_flag == 0, disp, 0.0).astype(np.float32)

    # Rolling (local context) features
    s = pd.Series(speed)
    a = pd.Series(accel)
    d = pd.Series(disp)

    speed_mean = s.rolling(rolling_window, min_periods=1).mean().values
    speed_std  = s.rolling(rolling_window, min_periods=1).std().fillna(0).values
    accel_mean = a.rolling(rolling_window, min_periods=1).mean().values
    disp_mean  = d.rolling(rolling_window, min_periods=1).mean().values

    # Fractional derivatives (optional but useful)
    # Use a stable dt for FC: median dt
    dt_fc = float(np.median(dt[1:])) if len(dt) > 1 else 1.0
    Dax = fractional_derivative(x, dt_fc, alpha=fc_alpha, r=fc_r)
    Day = fractional_derivative(y, dt_fc, alpha=fc_alpha, r=fc_r)

    # Build feature dataframe
    feat = pd.DataFrame({
        "t_sec": t_sec,
        "x": x, "y": y,
        "dt": dt,
        "dx": dx, "dy": dy,
        "disp": disp,
        "speed": speed,
        "accel": accel,
        "direction": direction,
        "roi": roi,
        "roi_change": roi_change,
        "dwell_time": dwell,
        "fix_flag": fix_flag,
        "fix_dur": fix_dur,
        "sacc_amp": sacc_amp,
        "speed_mean": speed_mean,
        "speed_std": speed_std,
        "accel_mean": accel_mean,
        "disp_mean": disp_mean,
        "Dax_fc": Dax,
        "Day_fc": Day,
        "alpha_fc": fc_alpha
    })

    # Add participant/cluster if present (keeps your context)
    for extra in ["participant", "cluster"]:
        if extra in df.columns:
            feat[extra] = df[extra].values

    # Targets for "next-step" prediction (what RF/SVM will learn later)
    feat["x_next"] = feat["x"].shift(-1)
    feat["y_next"] = feat["y"].shift(-1)
    feat["t_next"] = feat["t_sec"].shift(-1)
    feat["dt_next"] = (feat["t_next"] - feat["t_sec"])

    # Drop last row (no next target)
    feat = feat.iloc[:-1].reset_index(drop=True)

    return feat


if __name__ == "__main__":
    csv_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__), "..", "trucate_files", "cluster_means",
            "cluster_0_mean_processed.csv"
        )
    )

    features = build_features(
        csv_path,
        roi_cols=8, roi_rows=6,
        speed_thresh=80.0,     # tune: 60~120 typical
        fc_alpha=0.8, fc_r=20,
        rolling_window=10
    )

    out_path = "gaze_features.csv"
    features.to_csv(out_path, index=False)

    print("\n✅ Feature Engineering Completed!")
    print("Saved:", out_path)
    print("Feature columns:", list(features.columns))
    print("Rows:", len(features))
    print("\nPreview:")
    print(features.head(5))
