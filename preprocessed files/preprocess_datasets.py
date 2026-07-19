"""Reproducible, non-destructive preprocessing for the gaze project.

Run from any directory with:
    /usr/bin/python3 "preprocessed files/preprocess_datasets.py"

The script deliberately does not fit scalers or clip gaze coordinates. Model-specific
normalization belongs inside a training split, and no screen bounds are documented.
"""

from __future__ import annotations

from math import gamma
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent
RAW_OUT = OUT / "raw"
CLUSTER_OUT = OUT / "cluster_means"
FEATURE_OUT = OUT / "features"
RESULT_OUT = OUT / "results"

RAW_COLUMNS = ["participant", "set", "activity", "x", "y", "timestamp"]
NUMERIC_RAW = ["x", "y", "timestamp"]


def ensure_directories() -> None:
    for directory in (RAW_OUT, CLUSTER_OUT, FEATURE_OUT, RESULT_OUT):
        directory.mkdir(parents=True, exist_ok=True)


def profile(path: Path, role: str, action: str) -> dict:
    df = pd.read_csv(path)
    record = {
        "source_file": str(path.relative_to(ROOT)),
        "role": role,
        "rows": len(df),
        "columns": len(df.columns),
        "missing_cells": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "action": action,
    }
    if "timestamp" in df.columns:
        timestamp = pd.to_numeric(df["timestamp"], errors="coerce")
        record["duplicate_timestamps"] = int(timestamp.duplicated().sum())
        record["timestamp_decreases"] = int((timestamp.diff() < 0).sum())
    else:
        record["duplicate_timestamps"] = ""
        record["timestamp_decreases"] = ""
    return record


def clean_raw_file(path: Path) -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(path)
    if list(df.columns) != RAW_COLUMNS:
        raise ValueError(f"Unexpected schema in {path}: {list(df.columns)}")

    before = len(df)
    df = df.copy()
    for column in ("participant", "set", "activity"):
        df[column] = df[column].astype("string").str.strip().str.upper()
    for column in NUMERIC_RAW:
        df[column] = pd.to_numeric(df[column], errors="raise")

    # These operations are safe invariants. The current raw files need no row removal.
    df = df.drop_duplicates().sort_values("timestamp", kind="stable").reset_index(drop=True)
    duplicate_timestamp_count = int(df["timestamp"].duplicated().sum())
    if duplicate_timestamp_count:
        raise ValueError(f"Ambiguous duplicate timestamps in {path}")
    if df[RAW_COLUMNS].isna().any().any():
        raise ValueError(f"Missing required values in {path}")
    if (df["timestamp"].diff().dropna() <= 0).any():
        raise ValueError(f"Non-increasing timestamps in {path}")

    expected_participant = path.stem.split("_")[0]
    expected_activity = path.stem.split("_", 1)[1]
    if df["participant"].nunique() != 1 or df["participant"].iat[0] != expected_participant:
        raise ValueError(f"Participant value does not match filename in {path}")
    if df["activity"].nunique() != 1 or df["activity"].iat[0] != expected_activity:
        raise ValueError(f"Activity value does not match filename in {path}")

    details = {
        "file": path.name,
        "input_rows": before,
        "output_rows": len(df),
        "removed_duplicate_rows": before - len(df),
        "missing_cells_after": int(df.isna().sum().sum()),
        "timestamp_start": float(df["timestamp"].iat[0]),
        "timestamp_end": float(df["timestamp"].iat[-1]),
        "median_positive_dt": float(df["timestamp"].diff().dropna().median()),
        "x_min": float(df["x"].min()),
        "x_max": float(df["x"].max()),
        "y_min": float(df["y"].min()),
        "y_max": float(df["y"].max()),
        "transformation": "validated types/categories/order; no data values changed",
    }
    return df, details


def load_assignments() -> pd.DataFrame:
    path = ROOT / "Working" / "clustered_players_based_on_gaze.csv"
    assignments = pd.read_csv(path)
    required = {"participant", "cluster"}
    if not required.issubset(assignments.columns):
        raise ValueError(f"Missing assignment fields in {path}")
    assignments = assignments.copy()
    assignments["participant"] = assignments["participant"].astype("string").str.strip().str.upper()
    assignments["cluster"] = pd.to_numeric(assignments["cluster"], errors="raise").astype(int)
    if assignments["participant"].duplicated().any():
        raise ValueError("Participant assignments are not unique")
    return assignments.sort_values(["cluster", "participant"]).reset_index(drop=True)


def build_cluster_means(assignments: pd.DataFrame, raw_by_stem: dict[str, pd.DataFrame]) -> pd.DataFrame:
    membership_rows = []
    for cluster, group in assignments.groupby("cluster", sort=True):
        member_stems = group["participant"].tolist()
        missing = sorted(set(member_stems) - set(raw_by_stem))
        if missing:
            raise ValueError(f"Assignments reference missing files: {missing}")

        members = [raw_by_stem[stem] for stem in member_stems]
        common_start = max(float(df["timestamp"].min()) for df in members)
        common_end = min(float(df["timestamp"].max()) for df in members)
        positive_steps = np.concatenate(
            [df["timestamp"].diff().dropna().to_numpy(dtype=float) for df in members]
        )
        positive_steps = positive_steps[positive_steps > 0]
        dt = float(np.median(positive_steps))
        # Use a strict, uniform grid and include the last point only if it lies on-grid.
        grid = np.arange(common_start, common_end + dt * 0.5, dt, dtype=float)
        grid = grid[grid <= common_end]

        x_stack, y_stack = [], []
        for df in members:
            t = df["timestamp"].to_numpy(dtype=float)
            x_stack.append(np.interp(grid, t, df["x"].to_numpy(dtype=float)))
            y_stack.append(np.interp(grid, t, df["y"].to_numpy(dtype=float)))

        mean_df = pd.DataFrame(
            {
                "x": np.mean(x_stack, axis=0),
                "y": np.mean(y_stack, axis=0),
                "timestamp": grid,
                "participant": f"CLUSTER_{cluster}_MEAN",
                "cluster": int(cluster),
            }
        )
        output_path = CLUSTER_OUT / f"cluster_{cluster}_mean_processed.csv"
        mean_df.to_csv(output_path, index=False)
        membership_rows.append(
            {
                "cluster": int(cluster),
                "member_count": len(member_stems),
                "members": ";".join(member_stems),
                "common_start": common_start,
                "common_end": common_end,
                "uniform_dt": dt,
                "output_rows": len(mean_df),
                "missing_cells": int(mean_df.isna().sum().sum()),
            }
        )
    return pd.DataFrame(membership_rows)


def gl_weights(alpha: float, memory: int) -> np.ndarray:
    return np.asarray(
        [gamma(alpha + 1) / (gamma(k + 1) * gamma(alpha - k + 1)) for k in range(1, memory + 1)],
        dtype=float,
    )


def fractional_derivative(series: np.ndarray, dt: float, alpha: float, memory: int) -> np.ndarray:
    out = np.zeros_like(series, dtype=float)
    weights = gl_weights(alpha, memory)
    for index in range(memory, len(series)):
        history = series[index - np.arange(1, memory + 1)]
        signs = (-1.0) ** np.arange(1, memory + 1)
        out[index] = np.sum(signs * weights * history) / (dt**alpha)
    return out


def build_features(cluster_path: Path, alpha: float = 0.8, memory: int = 20) -> pd.DataFrame:
    df = pd.read_csv(cluster_path).sort_values("timestamp", kind="stable").reset_index(drop=True)
    t_raw = df["timestamp"].to_numpy(dtype=float)
    x = df["x"].to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=float)
    t_sec = (t_raw - t_raw[0]) / 1000.0
    dt = np.diff(t_sec, prepend=np.nan)
    dt[0] = float(np.median(dt[1:]))
    if not np.all(dt > 0):
        raise ValueError("Processed cluster grid must have strictly positive time steps")

    dx = np.diff(x, prepend=x[0])
    dy = np.diff(y, prepend=y[0])
    displacement = np.hypot(dx, dy)
    speed = displacement / dt
    acceleration = np.diff(speed, prepend=speed[0]) / dt
    direction = np.arctan2(dy, dx)

    x_norm = (x - x.min()) / (np.ptp(x) + 1e-8)
    y_norm = (y - y.min()) / (np.ptp(y) + 1e-8)
    col = np.clip((x_norm * 8).astype(int), 0, 7)
    row = np.clip((y_norm * 6).astype(int), 0, 5)
    roi = row * 8 + col
    roi_change = np.r_[0, (roi[1:] != roi[:-1]).astype(np.int8)]

    dwell = np.zeros(len(df), dtype=float)
    for index in range(len(df)):
        dwell[index] = dt[index] + (dwell[index - 1] if index > 0 and roi[index] == roi[index - 1] else 0.0)
    fixation = (speed < 80.0).astype(np.int8)
    fixation_duration = np.zeros(len(df), dtype=float)
    for index in range(len(df)):
        fixation_duration[index] = (
            dt[index] + (fixation_duration[index - 1] if index > 0 else 0.0)
            if fixation[index]
            else 0.0
        )
    saccade_amplitude = np.where(fixation == 0, displacement, 0.0)

    speed_series = pd.Series(speed)
    acceleration_series = pd.Series(acceleration)
    displacement_series = pd.Series(displacement)
    stable_dt = float(np.median(dt))
    feature = pd.DataFrame(
        {
            "t_sec": t_sec,
            "x": x,
            "y": y,
            "dt": dt,
            "dx": dx,
            "dy": dy,
            "disp": displacement,
            "speed": speed,
            "accel": acceleration,
            "direction": direction,
            "roi": roi,
            "roi_change": roi_change,
            "dwell_time": dwell,
            "fix_flag": fixation,
            "fix_dur": fixation_duration,
            "sacc_amp": saccade_amplitude,
            "speed_mean": speed_series.rolling(10, min_periods=1).mean(),
            "speed_std": speed_series.rolling(10, min_periods=1).std().fillna(0.0),
            "accel_mean": acceleration_series.rolling(10, min_periods=1).mean(),
            "disp_mean": displacement_series.rolling(10, min_periods=1).mean(),
            "Dax_fc": fractional_derivative(x, stable_dt, alpha, memory),
            "Day_fc": fractional_derivative(y, stable_dt, alpha, memory),
            "alpha_fc": alpha,
            "participant": df["participant"],
            "cluster": df["cluster"].astype(int),
        }
    )
    feature["x_next"] = feature["x"].shift(-1)
    feature["y_next"] = feature["y"].shift(-1)
    feature["t_next"] = feature["t_sec"].shift(-1)
    feature["dt_next"] = feature["t_next"] - feature["t_sec"]
    feature = feature.iloc[:-1].reset_index(drop=True)
    if not np.isfinite(feature.select_dtypes(include=[np.number]).to_numpy()).all():
        raise ValueError("Engineered features contain non-finite values")
    return feature


def validate_result_tables() -> list[dict]:
    records = []
    source_dir = ROOT / "Working" / "results" / "tables"
    for path in sorted(source_dir.glob("*.csv")):
        df = pd.read_csv(path)
        required = {"x_actual", "y_actual", "x_pred", "y_pred", "error"}
        if not required.issubset(df.columns):
            raise ValueError(f"Unexpected result schema in {path}")
        calculated = np.hypot(df["x_actual"] - df["x_pred"], df["y_actual"] - df["y_pred"])
        mismatch = int((~np.isclose(df["error"], calculated, rtol=1e-6, atol=1e-8, equal_nan=True)).sum())
        # Recompute error deterministically; preserve undefined model parameters such as warm-up alpha.
        df["error"] = calculated
        output_path = RESULT_OUT / path.name.replace(".csv", "_validated.csv")
        df.to_csv(output_path, index=False)
        records.append(
            {
                "file": path.name,
                "rows": len(df),
                "input_missing_cells": int(pd.read_csv(path).isna().sum().sum()),
                "error_mismatches": mismatch,
                "output_missing_cells": int(df.isna().sum().sum()),
                "note": "error recomputed; undefined alpha preserved rather than imputed",
            }
        )
    return records


def main() -> None:
    ensure_directories()
    source_profiles = []
    for path in sorted(ROOT.rglob("*.csv")):
        if OUT in path.parents:
            continue
        if path.parent == ROOT / "DataSet":
            role, action = "raw recording", "validate and preserve values"
        elif path.parent == ROOT / "Working" / "Dataset":
            role, action = "derived cluster trajectory", "replace with common-grid aggregation"
        elif path.parent == ROOT / "Working" / "results" / "tables":
            role, action = "historical evaluation output", "validate error only"
        else:
            role, action = "derived metadata/features", "validate or regenerate downstream artifact"
        source_profiles.append(profile(path, role, action))

    raw_frames, raw_details, raw_by_stem = [], [], {}
    for path in sorted((ROOT / "DataSet").glob("*.csv")):
        cleaned, details = clean_raw_file(path)
        cleaned.to_csv(RAW_OUT / path.name, index=False)
        raw_frames.append(cleaned)
        raw_details.append(details)
        raw_by_stem[path.stem] = cleaned

    combined = pd.concat(raw_frames, ignore_index=True)
    combined.to_csv(OUT / "combined_dataset_processed.csv", index=False)
    pd.DataFrame(raw_details).to_csv(OUT / "raw_preprocessing_summary.csv", index=False)

    assignments = load_assignments()
    assignments.to_csv(OUT / "clustered_players_based_on_gaze_processed.csv", index=False)
    membership = build_cluster_means(assignments, raw_by_stem)
    membership.to_csv(OUT / "cluster_membership_and_grid.csv", index=False)

    louvain_path = ROOT / "Working" / "louvain_community_partition.csv"
    louvain = pd.read_csv(louvain_path).sort_values(["community", "participant"]).reset_index(drop=True)
    louvain.to_csv(OUT / "louvain_community_partition_validated.csv", index=False)

    cluster_zero = CLUSTER_OUT / "cluster_0_mean_processed.csv"
    features = build_features(cluster_zero)
    features.to_csv(FEATURE_OUT / "gaze_features_processed.csv", index=False)

    pd.DataFrame(validate_result_tables()).to_csv(OUT / "result_validation_summary.csv", index=False)
    pd.DataFrame(source_profiles).to_csv(OUT / "source_data_quality_audit.csv", index=False)

    print(f"Processed {len(raw_frames)} raw recordings ({len(combined):,} rows).")
    print(f"Created {len(membership)} common-grid cluster trajectories.")
    print(f"Created {len(features):,} finite feature rows.")
    print(f"Outputs: {OUT}")


if __name__ == "__main__":
    main()
