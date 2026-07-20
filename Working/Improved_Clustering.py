"""Improved participant clustering for gaze-behavior analysis.

The script extracts one rich feature vector per participant, scales the complete
participant-feature matrix once, builds a proper RBF affinity matrix, evaluates
candidate cluster counts, and writes downstream-compatible cluster trajectories.

Default input:
    <repository>/trucate_files/raw/*.csv

Default output:
    <repository>/Working/Improved_Clustering_Output/

Example:
    python Working/Improved_Clustering.py
    python Working/Improved_Clustering.py --output-dir my_cluster_run
"""

from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import SpectralClustering
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.metrics.pairwise import pairwise_distances, rbf_kernel
from sklearn.preprocessing import StandardScaler


RANDOM_SEED = 42
MIN_CLUSTERS = 2
MAX_CLUSTERS = 10
REQUIRED_COLUMNS = {"participant", "x", "y", "timestamp"}

# Older scikit-learn builds paired with newer NumPy can emit spurious overflow
# warnings from their optimized dot-product helper even when the standardized
# input, affinity, labels, and returned metrics are all finite. Limit suppression
# to that internal module so genuine warnings elsewhere remain visible.
warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    module=r"sklearn\.utils\.extmath",
)


@dataclass(frozen=True)
class CandidateResult:
    """Evaluation scores and labels for one candidate cluster count."""

    n_clusters: int
    silhouette: float
    davies_bouldin: float
    calinski_harabasz: float
    labels: np.ndarray


def load_participant_csv(path: Path) -> pd.DataFrame:
    """Load and validate one participant recording without altering gaze values."""

    frame = pd.read_csv(path)
    missing_columns = REQUIRED_COLUMNS.difference(frame.columns)
    if missing_columns:
        raise ValueError(f"{path} is missing required columns: {sorted(missing_columns)}")

    frame = frame.copy()
    for column in ("x", "y", "timestamp"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    # Invalid numeric rows cannot contribute to motion statistics. Exact duplicates
    # and duplicate timestamps are removed deterministically before differentiation.
    frame = (
        frame.dropna(subset=["x", "y", "timestamp"])
        .drop_duplicates()
        .sort_values("timestamp", kind="stable")
        .drop_duplicates(subset="timestamp", keep="first")
        .reset_index(drop=True)
    )
    if len(frame) < 3:
        raise ValueError(f"{path} has fewer than three valid timestamped samples")
    if (frame["timestamp"].diff().dropna() <= 0).any():
        raise ValueError(f"{path} does not have strictly increasing timestamps")

    participants = frame["participant"].astype(str).str.strip().unique()
    if len(participants) != 1:
        raise ValueError(f"{path} contains multiple participant identifiers")
    return frame


def normalized_spatial_entropy(x: np.ndarray, y: np.ndarray) -> float:
    """Measure how broadly gaze occupies a participant-relative 8x6 grid."""

    histogram, _, _ = np.histogram2d(x, y, bins=(8, 6))
    probabilities = histogram.ravel().astype(float)
    probabilities = probabilities[probabilities > 0]
    probabilities /= probabilities.sum()
    entropy = -np.sum(probabilities * np.log(probabilities))
    return float(entropy / np.log(8 * 6)) if len(probabilities) > 1 else 0.0


def extract_participant_features(frame: pd.DataFrame) -> Dict[str, float]:
    """Convert a gaze trajectory into a rich, fixed-length behavior vector."""

    x = frame["x"].to_numpy(dtype=float)
    y = frame["y"].to_numpy(dtype=float)
    timestamp = frame["timestamp"].to_numpy(dtype=float)

    raw_dt = np.diff(timestamp)
    # README/data audit indicates millisecond timestamps (median step about 32).
    time_scale = 1000.0 if np.median(raw_dt) > 10.0 else 1.0
    dt_seconds = raw_dt / time_scale

    dx = np.diff(x)
    dy = np.diff(y)
    step_length = np.hypot(dx, dy)
    speed = np.divide(
        step_length,
        dt_seconds,
        out=np.zeros_like(step_length),
        where=dt_seconds > 0,
    )

    acceleration = np.diff(speed) / dt_seconds[1:]
    acceleration = acceleration[np.isfinite(acceleration)]
    direction = np.unwrap(np.arctan2(dy, dx))
    turning_angle = np.abs(np.diff(direction))

    scanpath_length = float(step_length.sum())
    direct_distance = float(np.hypot(x[-1] - x[0], y[-1] - y[0]))
    duration_seconds = float((timestamp[-1] - timestamp[0]) / time_scale)
    covariance = float(np.cov(x, y, ddof=0)[0, 1])
    correlation = float(np.corrcoef(x, y)[0, 1])
    if not np.isfinite(correlation):
        correlation = 0.0

    features = {
        # Coordinate distribution and robust spread.
        "mean_x": float(np.mean(x)),
        "std_x": float(np.std(x)),
        "min_x": float(np.min(x)),
        "max_x": float(np.max(x)),
        "var_x": float(np.var(x)),
        "median_x": float(np.median(x)),
        "iqr_x": float(np.percentile(x, 75) - np.percentile(x, 25)),
        "mean_y": float(np.mean(y)),
        "std_y": float(np.std(y)),
        "min_y": float(np.min(y)),
        "max_y": float(np.max(y)),
        "var_y": float(np.var(y)),
        "median_y": float(np.median(y)),
        "iqr_y": float(np.percentile(y, 75) - np.percentile(y, 25)),
        # Motion and scanpath behavior.
        "mean_speed": float(np.mean(speed)),
        "std_speed": float(np.std(speed)),
        "max_speed": float(np.max(speed)),
        "median_speed": float(np.median(speed)),
        "p95_speed": float(np.percentile(speed, 95)),
        "total_scanpath_length": scanpath_length,
        "average_step_length": float(np.mean(step_length)),
        "std_step_length": float(np.std(step_length)),
        "max_step_length": float(np.max(step_length)),
        "mean_abs_acceleration": float(np.mean(np.abs(acceleration))) if len(acceleration) else 0.0,
        "mean_turning_angle": float(np.mean(turning_angle)) if len(turning_angle) else 0.0,
        # Sampling and duration characteristics.
        "mean_timestamp_interval": float(np.mean(raw_dt)),
        "std_timestamp_interval": float(np.std(raw_dt)),
        "max_timestamp_interval": float(np.max(raw_dt)),
        "duration_seconds": duration_seconds,
        "sampling_rate_hz": float((len(frame) - 1) / duration_seconds) if duration_seconds > 0 else 0.0,
        # Spatial relationship and additional interpretable behavior descriptors.
        "covariance_xy": covariance,
        "correlation_xy": correlation,
        "bounding_box_area": float(np.ptp(x) * np.ptp(y)),
        "path_efficiency": direct_distance / scanpath_length if scanpath_length > 0 else 0.0,
        "fixation_ratio": float(np.mean(speed < 80.0)),
        "stationary_step_ratio": float(np.mean(step_length < 1.0)),
        "spatial_entropy": normalized_spatial_entropy(x, y),
    }

    if not np.isfinite(np.fromiter(features.values(), dtype=float)).all():
        raise ValueError("Feature extraction produced a non-finite value")
    return features


def build_feature_table(data_directory: Path) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Extract one feature row per CSV after loading files in stable order."""

    paths = sorted(data_directory.glob("*.csv"))
    if len(paths) < 3:
        raise ValueError(f"Need at least three participant CSVs in {data_directory}")

    rows: List[Dict[str, float]] = []
    participant_frames: Dict[str, pd.DataFrame] = {}
    for path in paths:
        frame = load_participant_csv(path)
        # Keep filename stems (for example P01_PLAY) compatible with legacy output.
        participant_id = path.stem
        if participant_id in participant_frames:
            raise ValueError(f"Duplicate participant file identifier: {participant_id}")
        participant_frames[participant_id] = frame
        row = extract_participant_features(frame)
        row["participant"] = participant_id
        rows.append(row)

    feature_table = pd.DataFrame(rows).sort_values("participant").reset_index(drop=True)
    return feature_table, participant_frames


def build_rbf_affinity(scaled_features: np.ndarray) -> Tuple[np.ndarray, float]:
    """Build a Gaussian similarity matrix using a median-distance bandwidth."""

    squared_distances = pairwise_distances(scaled_features, metric="sqeuclidean")
    positive_distances = squared_distances[squared_distances > 0]
    if len(positive_distances) == 0:
        raise ValueError("All participant feature vectors are identical")

    median_squared_distance = float(np.median(positive_distances))
    gamma = 1.0 / (2.0 * median_squared_distance)
    affinity = rbf_kernel(scaled_features, gamma=gamma)
    np.fill_diagonal(affinity, 1.0)
    return affinity, gamma


def spectral_labels(affinity: np.ndarray, n_clusters: int) -> np.ndarray:
    """Fit reproducible Spectral Clustering against a precomputed affinity."""

    model = SpectralClustering(
        n_clusters=n_clusters,
        affinity="precomputed",
        assign_labels="kmeans",
        n_init=100,
        random_state=RANDOM_SEED,
    )
    return model.fit_predict(affinity)


def evaluate_candidates(
    scaled_features: np.ndarray,
    affinity: np.ndarray,
    minimum: int = MIN_CLUSTERS,
    maximum: int = MAX_CLUSTERS,
) -> List[CandidateResult]:
    """Evaluate every feasible k with three complementary quality metrics."""

    n_participants = len(scaled_features)
    upper_bound = min(maximum, n_participants - 1)
    if minimum > upper_bound:
        raise ValueError("Not enough participants for the requested cluster range")

    results: List[CandidateResult] = []
    print("\nEvaluating candidate cluster counts")
    print("=" * 83)
    print(f"{'k':>3} | {'Silhouette':>12} | {'Davies-Bouldin':>16} | {'Calinski-Harabasz':>19}")
    print("-" * 83)

    for n_clusters in range(minimum, upper_bound + 1):
        labels = spectral_labels(affinity, n_clusters)
        unique_labels = np.unique(labels)
        if len(unique_labels) < 2 or len(unique_labels) >= n_participants:
            print(f"{n_clusters:>3} | skipped: clustering produced {len(unique_labels)} usable labels")
            continue

        result = CandidateResult(
            n_clusters=n_clusters,
            silhouette=float(silhouette_score(scaled_features, labels)),
            davies_bouldin=float(davies_bouldin_score(scaled_features, labels)),
            calinski_harabasz=float(calinski_harabasz_score(scaled_features, labels)),
            labels=labels,
        )
        results.append(result)
        print(
            f"{n_clusters:>3} | {result.silhouette:>12.6f} | "
            f"{result.davies_bouldin:>16.6f} | {result.calinski_harabasz:>19.6f}"
        )

    if not results:
        raise RuntimeError("No candidate cluster count produced valid evaluation metrics")
    return results


def select_best_candidate(results: Iterable[CandidateResult]) -> CandidateResult:
    """Select k by equal-weight rank aggregation across all three metrics.

    Silhouette and Calinski-Harabasz are maximized; Davies-Bouldin is minimized.
    Ranking prevents the much larger numeric range of Calinski-Harabasz from
    overwhelming the other metrics. Ties favor silhouette, then DB, CH, and
    finally the smaller cluster count.
    """

    result_list = list(results)
    score_table = pd.DataFrame(
        {
            "n_clusters": [item.n_clusters for item in result_list],
            "silhouette": [item.silhouette for item in result_list],
            "davies_bouldin": [item.davies_bouldin for item in result_list],
            "calinski_harabasz": [item.calinski_harabasz for item in result_list],
        }
    )
    score_table["silhouette_rank"] = score_table["silhouette"].rank(ascending=False, method="min")
    score_table["davies_bouldin_rank"] = score_table["davies_bouldin"].rank(ascending=True, method="min")
    score_table["calinski_harabasz_rank"] = score_table["calinski_harabasz"].rank(
        ascending=False, method="min"
    )
    score_table["rank_sum"] = score_table[
        ["silhouette_rank", "davies_bouldin_rank", "calinski_harabasz_rank"]
    ].sum(axis=1)

    selected_row = score_table.sort_values(
        ["rank_sum", "silhouette", "davies_bouldin", "calinski_harabasz", "n_clusters"],
        ascending=[True, False, True, False, True],
    ).iloc[0]
    selected_k = int(selected_row["n_clusters"])
    return next(item for item in result_list if item.n_clusters == selected_k)


def candidate_metrics_table(results: Iterable[CandidateResult]) -> pd.DataFrame:
    """Return evaluation metrics and selection ranks in a reusable table."""

    table = pd.DataFrame(
        {
            "n_clusters": [item.n_clusters for item in results],
            "silhouette_score": [item.silhouette for item in results],
            "davies_bouldin_index": [item.davies_bouldin for item in results],
            "calinski_harabasz_score": [item.calinski_harabasz for item in results],
        }
    )
    table["silhouette_rank"] = table["silhouette_score"].rank(ascending=False, method="min")
    table["davies_bouldin_rank"] = table["davies_bouldin_index"].rank(
        ascending=True, method="min"
    )
    table["calinski_harabasz_rank"] = table["calinski_harabasz_score"].rank(
        ascending=False, method="min"
    )
    table["aggregate_rank"] = table[
        ["silhouette_rank", "davies_bouldin_rank", "calinski_harabasz_rank"]
    ].sum(axis=1)
    return table.sort_values("n_clusters").reset_index(drop=True)


def canonicalize_cluster_labels(labels: np.ndarray, participants: pd.Series) -> np.ndarray:
    """Give clusters stable IDs ordered by their lexicographically first member."""

    members = {
        int(label): sorted(participants[labels == label].astype(str).tolist())
        for label in np.unique(labels)
    }
    ordered_old_labels = sorted(members, key=lambda label: members[label][0])
    mapping = {old_label: new_label for new_label, old_label in enumerate(ordered_old_labels)}
    return np.asarray([mapping[int(label)] for label in labels], dtype=int)


def create_cluster_mean(
    cluster_id: int,
    participant_ids: Iterable[str],
    participant_frames: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Average members after interpolation onto a shared uniform time grid."""

    ids = list(participant_ids)
    frames = [participant_frames[participant_id] for participant_id in ids]
    common_start = max(float(frame["timestamp"].min()) for frame in frames)
    common_end = min(float(frame["timestamp"].max()) for frame in frames)

    positive_intervals = np.concatenate(
        [frame["timestamp"].diff().dropna().to_numpy(dtype=float) for frame in frames]
    )
    positive_intervals = positive_intervals[positive_intervals > 0]
    interval = float(np.median(positive_intervals))
    if common_end <= common_start or interval <= 0:
        raise ValueError(f"Cluster {cluster_id} has no valid common time grid")

    time_grid = np.arange(common_start, common_end + interval * 0.5, interval)
    time_grid = time_grid[time_grid <= common_end]
    x_values, y_values = [], []
    for frame in frames:
        timestamps = frame["timestamp"].to_numpy(dtype=float)
        x_values.append(np.interp(time_grid, timestamps, frame["x"].to_numpy(dtype=float)))
        y_values.append(np.interp(time_grid, timestamps, frame["y"].to_numpy(dtype=float)))

    return pd.DataFrame(
        {
            "x": np.mean(x_values, axis=0),
            "y": np.mean(y_values, axis=0),
            "timestamp": time_grid,
            "participant": f"CLUSTER_{cluster_id}_MEAN",
            "cluster": cluster_id,
        }
    )


def save_outputs(
    feature_table: pd.DataFrame,
    participant_frames: Dict[str, pd.DataFrame],
    output_directory: Path,
) -> pd.DataFrame:
    """Save legacy-compatible assignments and one mean CSV per cluster."""

    output_directory.mkdir(parents=True, exist_ok=True)
    assignment_path = output_directory / "clustered_players_based_on_gaze.csv"
    feature_table.to_csv(assignment_path, index=False)
    output_records = [
        {
            "output_type": "participant_assignments_and_features",
            "file": assignment_path.name,
            "rows": len(feature_table),
            "columns": len(feature_table.columns),
            "cluster": "",
            "members": feature_table["participant"].nunique(),
            "missing_cells": int(feature_table.isna().sum().sum()),
        }
    ]

    for cluster_id, group in feature_table.groupby("cluster", sort=True):
        participant_ids = group["participant"].tolist()
        cluster_mean = create_cluster_mean(
            int(cluster_id), participant_ids, participant_frames
        )
        output_path = output_directory / f"cluster_{int(cluster_id)}_mean.csv"
        cluster_mean.to_csv(output_path, index=False)
        output_records.append(
            {
                "output_type": "cluster_mean_trajectory",
                "file": output_path.name,
                "rows": len(cluster_mean),
                "columns": len(cluster_mean.columns),
                "cluster": int(cluster_id),
                "members": len(participant_ids),
                "missing_cells": int(cluster_mean.isna().sum().sum()),
            }
        )
    return pd.DataFrame(output_records)


def save_structured_reports(
    output_directory: Path,
    feature_table: pd.DataFrame,
    candidates: List[CandidateResult],
    selected: CandidateResult,
    gamma: float,
    input_directory: Path,
    output_manifest: pd.DataFrame,
) -> None:
    """Save human- and machine-readable reports alongside legacy outputs."""

    reports_directory = output_directory / "reports"
    reports_directory.mkdir(parents=True, exist_ok=True)

    metrics = candidate_metrics_table(candidates)
    metrics["selected"] = metrics["n_clusters"].eq(selected.n_clusters)
    metrics.to_csv(reports_directory / "cluster_evaluation_metrics.csv", index=False)

    membership = feature_table[["participant", "cluster"]].sort_values(
        ["cluster", "participant"]
    )
    membership.to_csv(reports_directory / "cluster_membership.csv", index=False)

    cluster_summary = (
        membership.groupby("cluster", sort=True)["participant"]
        .agg(member_count="size", participants=lambda values: ";".join(values))
        .reset_index()
    )
    cluster_summary.to_csv(reports_directory / "cluster_summary.csv", index=False)
    output_manifest.to_csv(reports_directory / "output_manifest.csv", index=False)

    run_summary = {
        "input_directory": str(input_directory),
        "output_directory": str(output_directory),
        "random_seed": RANDOM_SEED,
        "participants": int(feature_table["participant"].nunique()),
        "gaze_features": int(len(feature_table.columns) - 2),
        "candidate_cluster_range": [MIN_CLUSTERS, MAX_CLUSTERS],
        "rbf_gamma": float(gamma),
        "selection_method": "lowest equal-weight aggregate metric rank",
        "selected_n_clusters": int(selected.n_clusters),
        "selected_scores": {
            "silhouette_score": float(selected.silhouette),
            "davies_bouldin_index": float(selected.davies_bouldin),
            "calinski_harabasz_score": float(selected.calinski_harabasz),
        },
        "cluster_sizes": {
            str(int(row.cluster)): int(row.member_count)
            for row in cluster_summary.itertuples(index=False)
        },
    }
    with (reports_directory / "run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(run_summary, handle, indent=2)
        handle.write("\n")


def parse_arguments() -> argparse.Namespace:
    """Create a small CLI while retaining project-compatible defaults."""

    repository_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=repository_root / "trucate_files" / "raw",
        help="Directory containing one truncated gaze CSV per participant.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "Improved_Clustering_Output",
        help="Directory for assignment and cluster_<id>_mean.csv outputs.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the complete improved clustering workflow."""

    arguments = parse_arguments()
    data_directory = arguments.data_dir.resolve()
    output_directory = arguments.output_dir.resolve()
    print("\n" + "=" * 88)
    print("IMPROVED GAZE-BEHAVIOR CLUSTERING")
    print("=" * 88)
    print("\n[1/5] RUN CONFIGURATION")
    print("-" * 88)
    print(f"Input directory    : {data_directory}")
    print(f"Output directory   : {output_directory}")
    print(f"Random seed        : {RANDOM_SEED}")
    print(f"Candidate k values : {MIN_CLUSTERS} to {MAX_CLUSTERS}")

    feature_table, participant_frames = build_feature_table(data_directory)
    feature_columns = [column for column in feature_table.columns if column != "participant"]

    # Scale once across participants, never separately within a participant.
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(feature_table[feature_columns])
    affinity, gamma = build_rbf_affinity(scaled_features)
    print("\n[2/5] PARTICIPANT REPRESENTATION")
    print("-" * 88)
    print(f"Participants       : {len(feature_table)}")
    print(f"Features extracted : {len(feature_columns)}")
    print(f"Scaling            : one StandardScaler fitted across all participants")
    print(f"Affinity           : Gaussian RBF (gamma={gamma:.8f})")

    print("\n[3/5] CLUSTER-COUNT EVALUATION")
    candidates = evaluate_candidates(scaled_features, affinity)
    selected = select_best_candidate(candidates)
    stable_labels = canonicalize_cluster_labels(selected.labels, feature_table["participant"])
    feature_table["cluster"] = stable_labels

    print("\n[4/5] SELECTED CLUSTERING")
    print("-" * 88)
    print(f"Number of clusters       : {selected.n_clusters}")
    print(f"Silhouette Score         : {selected.silhouette:.6f}")
    print(f"Davies-Bouldin Index     : {selected.davies_bouldin:.6f}")
    print(f"Calinski-Harabasz Score  : {selected.calinski_harabasz:.6f}")
    print("Selection rule           : lowest equal-weight aggregate metric rank")

    print("\nCluster membership")
    for cluster_id, group in feature_table.groupby("cluster", sort=True):
        print(f"Cluster {cluster_id}: {', '.join(group['participant'].tolist())}")

    print("\n[5/5] GENERATED OUTPUTS")
    print("-" * 88)
    output_manifest = save_outputs(feature_table, participant_frames, output_directory)
    save_structured_reports(
        output_directory,
        feature_table,
        candidates,
        selected,
        gamma,
        data_directory,
        output_manifest,
    )
    print(output_manifest.to_string(index=False))
    print(f"\nStructured reports: {output_directory / 'reports'}")
    print("=" * 88)
    print("CLUSTERING COMPLETED SUCCESSFULLY")
    print("=" * 88)


if __name__ == "__main__":
    main()
