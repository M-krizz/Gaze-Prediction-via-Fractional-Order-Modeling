# Gaze-Prediction-via-Fractional-Order-Modeling — Project Report

## Executive Summary
This document is a complete technical handover for the repository *Gaze-Prediction-via-Fractional-Order-Modeling*. It documents datasets, preprocessing, feature engineering, models, clustering, visualizations, outputs, end-to-end workflows, architecture, and recommended improvements. It is intended for team handoff, audit, or academic submission.

## Project Overview
Purpose: explore fractional-order modeling (Grünwald–Letnikov), STARE-style transformers, LSTM decoders, and graph methods for gaze prediction and participant clustering using the provided gaze dataset.

Repository root: repository contains raw datasets ([DataSet](DataSet)), working experiments ([Working](Working)), materials ([Materials]), and generated artifacts such as `combined_dataset.csv` and model checkpoints.

## Repository Structure (high-level)
- DataSet/: raw per-participant CSVs (P01_PLAY.csv … P24_PLAY.csv, P12_READ.csv)
- Working/: experimental scripts and models (LSTM.py, STARE_No_FC.py, RandomForest.py, Adaptive_FC.py, FKP.py, Clustering.py, louvain.py, DBSCAN.py, GCN.py, P01_plot.py, Combine_dataset.py)
- Working/Dataset/: cluster-level aggregated CSVs (cluster_0_mean.csv … cluster_3_mean.csv)
- Working/results/: plots and tables (plots/ and tables/)
- combined_dataset.csv: concatenated raw dataset
- model checkpoints: `stare_fc_lstm_multistep.pth`, etc.

(See per-file details inside the Model Documentation and Dataset Documentation sections.)

---

## Dataset Documentation
This section lists datasets, columns, usage and sizes.

### 1) Raw participant files
- Location: [DataSet](DataSet)
- Files: `P01_PLAY.csv` … `P24_PLAY.csv`, `P12_READ.csv`
- Columns (per file):

| Column | Type | Description |
|---|---:|---|
| participant | string | participant id (e.g., P01) |
| set | string | experiment set (A/B) |
| activity | string | activity label (PLAY/READ) |
| x | numeric | gaze x coordinate (pixels) |
| y | numeric | gaze y coordinate (pixels) |
| timestamp | numeric | time offset (observed units: milliseconds) |

- Data size: total ≈ 209,638 rows across 24 files (wc -l computed). Example counts: `P01_PLAY.csv` = 7696 lines, `P13_PLAY.csv` = 9860 lines, `P20_PLAY.csv` = 10268 lines.
- Missing values: scripts handle NaNs via forward/backfill and median imputation. If any participant file lacks required columns, dataset loaders raise an error.
- Usage: raw input to clustering, feature engineering, sequence datasets, and model training.

### 2) Combined dataset
- File: [combined_dataset.csv](combined_dataset.csv)
- Produced by: [Working/Combine_dataset.py](Working/Combine_dataset.py)
- Columns: same as raw participant CSVs.
- Usage: global analyses and convenience when single-file input is required.

### 3) Cluster-level datasets
- Files: [Working/Dataset/cluster_0_mean.csv](Working/Dataset/cluster_0_mean.csv) … cluster_3_mean.csv
- Produced by: clustering scripts (Clustering.py, louvain.py, DBSCAN.py variants)
- Columns:

| Column | Type | Description |
|---|---:|---|
| x | numeric | mean x per timestep across cluster members |
| y | numeric | mean y |
| timestamp | numeric | averaged timestamp |
| participant | string | base participant id used as row anchor |
| cluster | int | cluster id |

- Notes: cluster CSVs are created by averaging rows across cluster member files; scripts assume alignable lengths or select a base player's timeline — this can create bias and is flagged in Recommendations.

### 4) Feature CSV (tabular)
- File: [Working/gaze_features.csv](Working/gaze_features.csv)
- Produced by: `Working/RandomForest.py` (function `build_features()`)
- Representative columns: `t_sec,x,y,dt,dx,dy,disp,speed,accel,direction,roi,roi_change,dwell_time,fix_flag,fix_dur,sacc_amp,speed_mean,speed_std,accel_mean,disp_mean,Dax_fc,Day_fc,alpha_fc,participant,cluster,x_next,y_next,t_next,dt_next`
- Purpose: training-ready table for classical models (Random Forest, XGBoost, etc.).

### 5) Results & evaluation tables
- Location: [Working/results/tables](Working/results/tables)
- Example files: `stare_no_fc_actual_vs_predicted.csv`, `fkf_actual_vs_predicted.csv` (if produced)
- Columns: `x_actual,y_actual,x_pred,y_pred,error` (script-specific)
- Purpose: evaluation, metric computation, reporting.

### 6) Model checkpoints
- Files: `stare_fc_lstm_multistep.pth`, `stare_fc_multistep.pth`, `stare_fc_model.pth`, `stare_gaze_best.pt` (locations: repo root or Working/)
- Format: PyTorch `state_dict` or checkpoint
- Usage: load with `torch.load()` in inference flows.

Limitations: Where run-time metrics or exact numbers weren't saved in files, I include reported console outputs where available. If further numeric audit is needed, run training/evaluation scripts that will re-generate metrics and artifacts.

---

## Feature Engineering Documentation
This section documents each feature-engineering module and the transformations applied.

### `Working/RandomForest.py` — Main tabular feature builder
- Purpose: compute kinematic, ROI, fixation, rolling and fractional features; create next-step targets.
- Key functions: `gl_weights()`, `fractional_derivative()`, `coords_to_roi_ids()`, `ivt_fixation_flags()`, `compute_fixation_duration()`, `build_features()`.
- Input: cluster CSV or participant CSV (expects columns: `timestamp`, `x`, `y`). Default: `Working/Dataset/cluster_0_mean.csv`.
- Transformations:
  - Numeric cleaning & imputation (timestamp ffill/bfill, x/y median fill)
  - Timestamp unit auto-detection (if median dt > 10 → treat as ms)
  - Kinematics: dx, dy, disp, speed (disp/dt), accel
  - Direction: arctan2(dy,dx)
  - ROI tokenization: grid (default 8 columns × 6 rows) → `roi` id
  - ROI changes, dwell time
  - Fixation detection (I-VT) with speed threshold (default 80 px/s)
  - Rolling-window statistics (mean/std) over `rolling_window` (default 10 samples)
  - Fractional derivatives Dax/Day via GL kernel (default alpha=0.8, r=20)
  - Targets: `x_next`, `y_next`
- Output: `gaze_features.csv` (CSV), rows N-1 (last sample dropped due to missing future target).

### `Working/LSTM.py` — Sequence + FC dataset builder
- Class: `GazeFCDataset`
- Purpose: produce sequence windows for STARE+FC+LSTM: ROI tokens + FC features per timestep + future H steps.
- Transformations:
  - Sort by timestamp, numeric cleaning, interpolate/median replaces missing values, normalize x,y to [0,1], normalize timestamps
  - Fractional derivatives computed on normalized gaze (`fractional_derivative()`)
  - ROI tokenization via `coords_to_roi_ids()`
  - Build `fc_feats` per timestep: `[Dαx, Dαy, α, t_norm]`; normalize FD channels
  - Build sequences: `roi_seqs` (seq_len,), `fc_seqs` (seq_len,4), `future_xy` (future_steps,2)

### `Working/STARE_No_FC.py` — Sequence builder (no FC)
- Class: `GazeSequenceDataset`
- Purpose: prepare simple sequences of normalized x,y for STARE baseline.
- Transformations: global min/max normalization, group-by participant, build sliding windows of length `seq_len`.

### `Working/Clustering.py`, `Working/DBSCAN.py`, `Working/louvain.py` — clustering preproc
- Purpose: compute per-player summary features and group players.
- Transformations: per-player StandardScaler on gaze points → compute mean and variance (mean_x, mean_y, var_x, var_y) → similarity matrix → cluster.
- Output: `clustered_players_based_on_gaze.csv`, `louvain_community_partition.csv`, `cluster_{i}_mean.csv`.

Limitations: cluster-mean aggregation assumes equal-length records or uses min-length base in some scripts; recommends resampling or interpolation for robust aggregation.

---

## Model Documentation
This section documents every model implemented in the repo. Each model has name, file, type, purpose, inputs, training details, outputs and metrics (if available).

### STARE (Transformer Encoder) — Next-step baseline
- File: [Working/STARE_No_FC.py](Working/STARE_No_FC.py)
- Type: Transformer encoder (encoder-only)
- Purpose: predict next gaze point (x,y) from a history sequence.
- Input: sequence (seq_len) of normalized x,y per participant (from `GazeSequenceDataset`), default `Working/Dataset/cluster_0_mean.csv`.
- Training: Adam (`lr=1e-3`), `EPOCHS=10`, `BATCH=64`, loss MSELoss. Outputs saved evaluation table (`Working/results/tables/stare_no_fc_actual_vs_predicted.csv`) and trajectory PNG.
- Metrics: script prints mean/median/max Euclidean errors (no standardized CSV metrics in repo beyond the saved table). See output CSV for example rows.

### STARE + FC + LSTM (Hybrid multi-step)
- File: [Working/LSTM.py](Working/LSTM.py)
- Type: Transformer encoder fused with fractional-calculus features; LSTM decoder for autoregressive multi-step prediction.
- Purpose: multi-step future gaze prediction (H steps).
- Input: `GazeFCDataset` sequences: ROI tokens, fc_feats per timestep, and future_xy targets (default `cluster_0_mean.csv`).
- Training: Adam (`lr=1e-4`), `epochs=10`, `batch_size=32`, MSELoss, gradient clipping (5.0). Model saved to `stare_fc_lstm_multistep.pth`.
- Observed example: training run printed final Loss ~0.00199 and `Overall Multi-step Accuracy ≈ 96.21%` in a prior execution. (If you need reproducible metrics rerun training with the dataset and a fixed seed.)
- Outputs: `stare_fc_lstm_multistep.pth`, optional plots via `visualize_future_prediction()`.

### Adaptive Fractional-Calculus Predictor
- File: [Working/Adaptive_FC.py](Working/Adaptive_FC.py)
- Type: algorithmic predictor using Grünwald–Letnikov fractional derivative and golden-section search for adaptive α.
- Purpose: one-step prediction using fractional memory with adaptive fractional order α.
- Input: resampled uniform time series from a cluster CSV (default `Working/Dataset/cluster_3_mean.csv`).
- Method: for each t, golden-section search finds α minimizing one-step error; `fractional_predict()` computes predicted value.
- Outputs: trajectory PNG and α time series PNG saved to `Working/results/plots`.
- Metrics: MSE per axis and normalized accuracy printed; values vary per dataset run.

### Fractional Kalman Filter (FKF)
- File: [Working/FKP.py](Working/FKP.py)
- Type: Kalman filter augmented with fractional-velocity term and adaptive α via golden-section search.
- Purpose: probabilistic filtering/prediction incorporating fractional memory.
- Input: uniform-resampled cluster CSV (default `Working/Dataset/cluster_3_mean.csv`).
- Method: state vector [x,y,vx,vy], compute fractional velocity terms, Kalman predict/update with adaptive Q estimation.
- Outputs: predicted trajectories and α plot (script displays via matplotlib).

### GCN (experiment)
- File: [Working/GCN.py](Working/GCN.py)
- Type: Graph Convolutional Network using PyG (untrained forward pass in current code)
- Purpose: experiment with graph embeddings for cluster-level nodes.
- Notes: Script runs a forward pass but contains no training loop; saved outputs are CSV summaries `cluster_{cluster}_final.csv`.

### Clustering Methods (KMeans / Spectral / Louvain)
- Files: `Working/DBSCAN.py` (KMeans actually), `Working/Clustering.py` (SpectralClustering), `Working/louvain.py` (Louvain)
- Purpose: group participants by mean/variance gaze features for cluster-level modeling.
- Outputs: `clustered_players_based_on_gaze.csv`, `louvain_community_partition.csv`, `cluster_{i}_mean.csv`.

---

## Clustering Documentation (summary)
Detailed descriptions are included in `Working/Clustering.py`, `Working/DBSCAN.py` and `Working/louvain.py`.

Key points:
- KMeans (in `DBSCAN.py`) uses `n_clusters=4` on pairwise Euclidean similarity of mean_x,mean_y,var_x,var_y.
- Spectral uses precomputed affinity and n_clusters=4.
- Louvain uses graph weights = 1/(1+distance) and determines communities automatically.

Outputs: cluster membership CSVs and cluster mean CSVs; no plots are produced by clustering scripts by default. Recommendation: generate PCA scatter plots colored by cluster.

---

## Visualization Documentation
Existing visualization scripts and generated images (where present):

- `Working/P01_plot.py` — trajectory and KDE heatmap for a participant (uses `plt.show()`; not saved by default).
- `Working/Adaptive_FC.py` — saves:
  - [Working/results/plots/trajectories/adaptive_fc_trajectory.png](Working/results/plots/trajectories/adaptive_fc_trajectory.png)
  - [Working/results/plots/alpha/adaptive_fc_alpha.png](Working/results/plots/alpha/adaptive_fc_alpha.png)
- `Working/STARE_No_FC.py` — saves evaluation table and a trajectory plot:
  - [Working/results/tables/stare_no_fc_actual_vs_predicted.csv](Working/results/tables/stare_no_fc_actual_vs_predicted.csv)
  - [Working/results/plots/trajectories/stare_no_fc_trajectory.png](Working/results/plots/trajectories/stare_no_fc_trajectory.png)
- `Working/LSTM.py` — visualization helper `visualize_future_prediction()` shows and prints multi-step accuracy for a sample (not saved by default).

Interpretation: see the Visualization Documentation earlier in this report. For missing saved images (LSTM, FKF), rerun the helper calls or add `save_plot()` calls.

---

## Output Files Documentation (summary)
All generated outputs are CSV, PNG or PyTorch checkpoint files. Representative list:

| Output File | Generated By | Purpose | Format |
|---|---|---|---|
| combined_dataset.csv | `Working/Combine_dataset.py` | unified raw data | CSV |
| gaze_features.csv | `Working/RandomForest.py` | feature table | CSV |
| stare_fc_lstm_multistep.pth | `Working/LSTM.py` | saved model | PyTorch (.pth) |
| stare_no_fc_actual_vs_predicted.csv | `Working/STARE_No_FC.py` | evaluation table | CSV |
| adaptive_fc_trajectory.png | `Working/Adaptive_FC.py` | visual compare actual/pred | PNG |

Sample output rows and example files are present in the repo (see links above).

---

## End-to-End Workflow (detailed)
1. Raw Data
   - Acquire per-participant CSVs in `DataSet/`.
2. Preprocessing
   - Clean numeric fields, impute missing values, sort by `participant` and `timestamp`.
   - Optional resampling/interpolation to uniform dt for certain models (Adaptive_FC, FKP).
3. Feature Engineering
   - Build per-timestep kinematic features, ROI tokens, fixation flags, rolling statistics.
   - Compute fractional derivatives (GL kernel) for long-memory features.
4. Model Training
   - Sequence models (STARE, STARE+FC+LSTM) trained with Adam/MSELoss; hyperparameters in scripts.
   - Algorithmic models (Adaptive_FC, FKP) run per-sample optimization (golden-section) and filtering; no training required.
5. Evaluation
   - De-normalize predictions; compute per-sample Euclidean error; aggregate metrics saved to `Working/results/tables`.
6. Prediction
   - Load checkpoints, run `predict_future()` or forward pass on new sequences.
7. Visualization
   - Save plots to `Working/results/plots` and tables to `Working/results/tables`.
8. Final Output
   - Trained model files (`.pth`), feature CSVs, cluster CSVs, and plots.

---

## Architecture Summary
- Data layer: raw CSVs under `DataSet/`.
- Preprocessing & features: `Working/RandomForest.py` (tabular) and dataset classes in `Working/LSTM.py` & `Working/STARE_No_FC.py` (sequence builders).
- Modeling layer: Transformer encoder (STARE), LSTM decoder (hybrid), fractional algorithmic predictors (Adaptive_FC, FKP), graph experiments (GCN).
- Output layer: `Working/results/` for tables & plots; top-level model artifacts in `.pth` files.

---

## Improvements & Recommendations (prioritized)
1. Consolidate clustering code and fix naming (`DBSCAN.py` → rename to `KMeans_clustering.py`). Use a single clustering module with configuration flags.
2. Add `requirements.txt` or `pyproject.toml` listing exact package versions and a small `README.md` with run instructions.
3. Standardize evaluation metrics via a shared `metrics.py` (RMSE, MAE, Euclidean error, normalized accuracy) and save metrics JSON after each run.
4. Fix cluster aggregation: resample every participant time series to a common uniform grid (e.g., median dt) before averaging to produce `cluster_mean` files.
5. Add consistent plot-saving behavior across scripts and include a `--save` flag for interactive plots.
6. Optimize fractional derivative computations (vectorize or numba) for large sequences.
7. Add small unit tests for core utilities (`gl_weights`, `fractional_derivative`, `coords_to_roi_ids`) and a CI pipeline.
8. If using GCN embeddings: add training/evaluation loop or use embeddings for downstream unsupervised clustering and save them.

---

## Missing / Unclear Items (limitations)
- Some scripts do not save numerical metrics to files (metrics only printed to console). Recommendation: save metrics per run to `Working/results/tables/metrics_{script}.csv` or JSON.
- GCN script runs a forward pass but lacks a training loop; intent likely experimental.
- Some earlier absolute/Windows paths were patched; verify `Working/Stare_FC/STARE.py` for any remaining hard-coded paths before running on another environment.

---

## Conclusion
This report documents the repository's data, feature engineering, models, clustering, visualizations, generated outputs, end-to-end workflow, architecture and recommended improvements. Next actionable steps I can perform for you:

- Export this report into the repo as `REPO_REPORT.md` (done).
- Create `requirements.txt` and `README.md` with run instructions.
- Implement diagnostic plots for clustering (PCA scatter, per-cluster overlays).
- Consolidate clustering scripts into a single module and refactor repeated utilities.

Please tell me which of the next actionable steps you'd like me to take now.
