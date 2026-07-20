# Dataset Preprocessing Report

## Scope and governing requirements

Preprocessing was performed only after reading the repository `README.md` in full. The downstream workflow requires:

- participant-level gaze recordings for clustering and visualization;
- cluster-level mean trajectories for feature engineering, fractional predictors, and deep sequence models;
- a finite next-step feature table for classical modeling;
- preserved historical evaluation tables for result analysis.

All original files remain untouched. Every generated artifact is contained in `preprocessed files/`, and the complete process is reproducible with:

```bash
/usr/bin/python3 "preprocessed files/preprocess_datasets.py"
```

The interpreter path above is recorded because the active Miniforge Python did not have Pandas installed during this run, while the system Python contained Pandas 2.3.3 and NumPy 2.0.2.

## Preprocessing policy

The following principles were applied:

1. Validate before transforming.
2. Do not remove plausible gaze behavior merely because it is statistically extreme.
3. Do not assume screen dimensions that are absent from the project documentation.
4. Do not globally scale or normalize model inputs; scaling parameters must be learned from a training partition to avoid leakage.
5. Do not encode categories when existing downstream scripts expect their string values.
6. Repair derived cluster trajectories from clean raw recordings instead of imputing corrupt aggregate files.
7. Preserve undefined historical model parameters rather than inventing values.

## Source inventory and audit result

The audit covered all 34 CSV files present before preprocessing:

| Category | Files | Audit result | Action |
|---|---:|---|---|
| Raw participant recordings | 24 | Structurally clean | Validated and copied without value changes |
| Existing cluster means | 4 | Clusters 0 and 2 contain missing values and duplicate timestamps | Rebuilt from raw members on uniform common grids |
| Spectral assignment table | 1 | Complete, but its gaze summary features are nearly constant | Sorted/validated; labels retained only for lineage |
| Louvain partition | 1 | Complete, but every participant is community 0 | Sorted/validated; no fabricated communities |
| Engineered gaze features | 1 | Complete, but extreme speed/acceleration arise from near-zero time steps in faulty aggregate data | Regenerated from repaired cluster 0 trajectory |
| Historical evaluation tables | 3 | Error values consistent; FKF has one undefined alpha | Recomputed error and preserved undefined alpha |

Machine-readable details for every source are in `source_data_quality_audit.csv`.

## Raw participant recordings

### Files

`raw/P01_PLAY.csv` through `raw/P24_PLAY.csv`, with `raw/P12_READ.csv` for participant 12.

### Audit

Across the 24 files:

- 209,614 total rows;
- exactly six expected columns: `participant`, `set`, `activity`, `x`, `y`, `timestamp`;
- no missing cells;
- no duplicate rows;
- no duplicate timestamps within a participant;
- no decreasing or non-increasing timestamps;
- all numeric columns parse successfully;
- participant and activity values match their filenames;
- one consistent set and activity value per file.

Sampling gaps occur, and robust statistical checks flag some x/y values as extreme. These were not treated as errors. Gaze coordinates can legitimately leave the visible screen during blinks, tracker loss, calibration drift, or off-screen looks, and the repository provides neither validity flags nor screen boundaries needed to distinguish these cases safely.

### Transformations

- Enforced numeric parsing for `x`, `y`, and `timestamp`.
- Trimmed and uppercased categorical strings as a consistency safeguard.
- Applied stable timestamp sorting and exact-row deduplication.
- Validated participant/activity against the filename.

The current inputs already satisfied all conditions, so zero rows were removed and all 24 output files compare equal to their sources. Per-file row counts, ranges, time intervals, and actions are recorded in `raw_preprocessing_summary.csv`.

### Deliberately not performed

- No missing-value imputation was necessary.
- No coordinate clipping or outlier removal was defensible.
- No temporal interpolation or smoothing was applied to participant recordings.
- No categorical encoding was needed for the current workflow.
- No scaling or normalization was applied before train/test splitting.

## Combined raw dataset

### Output

`combined_dataset_processed.csv`

### Transformation and rationale

The 24 validated participant files were concatenated in sorted filename order. The output contains 209,614 rows, 24 participants, no missing cells, and no duplicate rows. Participant sequences retain their original within-file chronological order. This file is ready for global exploratory analysis; sequence models should still group by participant before window construction.

## Participant cluster assignment

### Output

`clustered_players_based_on_gaze_processed.csv`

### Transformation and rationale

Participant identifiers were normalized to uppercase, cluster labels were validated as integers, uniqueness was checked, and rows were sorted by cluster and participant. Existing cluster labels were preserved because preprocessing must not silently replace a modeling decision.

### Scientific warning

The four stored gaze summary features are effectively constant: standardized means are approximately zero and standardized variances approximately one. The cluster labels are therefore not reliable evidence of behavioral separation. They are retained only to reproduce the repository's current downstream lineage. A future clustering experiment should derive meaningful raw-coordinate, dispersion, fixation, velocity, or trajectory features and fit a reproducible clustering model.

## Rebuilt cluster-mean trajectories

### Outputs

- `cluster_means/cluster_0_mean_processed.csv`
- `cluster_means/cluster_1_mean_processed.csv`
- `cluster_means/cluster_2_mean_processed.csv`
- `cluster_means/cluster_3_mean_processed.csv`
- `cluster_membership_and_grid.csv`

### Why rebuilding was necessary

The original cluster means add unequal-length Pandas Series by row index. This produced 8,598 missing cells and 2,865 duplicate timestamps in cluster 0, and 4,743 missing cells and 1,580 duplicate timestamps in cluster 2. Filling those aggregate values would conceal an alignment error and could mix unrelated moments.

### Transformation

For each inherited cluster:

1. Load every member from the validated raw copies.
2. Determine the time interval shared by all members: maximum start time through minimum end time.
3. Calculate the median positive sampling interval across all cluster members.
4. Create one strictly increasing uniform timestamp grid over the common interval.
5. Linearly interpolate each participant's raw x and y values onto that grid.
6. Average aligned x values and aligned y values across members.
7. Store an honest synthetic identifier such as `CLUSTER_0_MEAN`, rather than labeling the aggregate as one participant.

### Result

| Cluster | Rows | Uniform interval | Missing cells | Duplicate timestamps |
|---:|---:|---:|---:|---:|
| 0 | 9,294 | 32 ms | 0 | 0 |
| 1 | 9,505 | 30 ms | 0 | 0 |
| 2 | 9,677 | 31 ms | 0 | 0 |
| 3 | 9,366 | 32 ms | 0 | 0 |

These files match the five-column schema expected by the repository's sequence and fractional models. Membership, common time bounds, interval, and output size are recorded in `cluster_membership_and_grid.csv`.

## Engineered gaze features

### Output

`features/gaze_features_processed.csv`

### Why regeneration was necessary

The original feature table is complete, but it was derived from faulty cluster 0 data. Duplicate/near-duplicate aggregate timestamps were clipped to `1e-6` seconds by the original feature code, producing speed up to 51,911,444 and absolute acceleration above 51 trillion. These are numerical artifacts, not credible gaze dynamics.

### Transformation

Features were regenerated from `cluster_0_mean_processed.csv` using the repository's intended definitions:

- milliseconds converted to elapsed seconds;
- `dt`, `dx`, `dy`, displacement, speed, acceleration, and direction;
- min-max coordinate mapping only for 8-by-6 ROI tokenization;
- ROI change and cumulative dwell time;
- I-VT fixation flag with the project's 80 px/s threshold;
- fixation duration and saccade amplitude;
- 10-sample rolling speed, acceleration, and displacement statistics;
- finite-memory GL derivatives with `alpha=0.8` and `r=20`;
- next-step x, y, time, and time-step targets.

Only ROI construction uses whole-series coordinate ranges, matching the existing code. Predictive model scaling is intentionally not included.

### Result

- 9,293 rows and 29 columns;
- no missing or infinite numerical values;
- strictly positive `dt` values of approximately 0.032 seconds;
- maximum speed reduced to approximately 3,253.93 coordinate units/second;
- maximum absolute acceleration approximately 74,548.69 coordinate units/second squared.

The last trajectory row is omitted because no next-step target exists, as required by the project's feature schema.

## Louvain community partition

### Output

`louvain_community_partition_validated.csv`

The table was schema-checked and sorted. All 24 participants remain in community 0. No new communities were inferred because that would be modeling, not preprocessing, and the README documents a graph-construction defect in the original Louvain script.

## Historical evaluation tables

### Outputs

- `results/adaptive_fc_actual_vs_predicted_validated.csv`
- `results/fkf_actual_vs_predicted_validated.csv`
- `results/stare_no_fc_actual_vs_predicted_validated.csv`
- `result_validation_summary.csv`

For every row, Euclidean error was deterministically recomputed as:

```text
sqrt((x_actual - x_pred)^2 + (y_actual - y_pred)^2)
```

All original stored errors already matched this calculation. The FKF table contains one missing `alpha` value in its final row. That parameter is undefined for the saved warm-up/final state and was deliberately not mean-filled, forward-filled, or set to zero. Actual values, predictions, row order, and other fields were preserved.

These outputs remain historical evaluation artifacts. They should not be mixed as one supervised dataset because their coordinate scales and evaluation procedures differ.

## Output structure

```text
preprocessed files/
├── PREPROCESSING_REPORT.md
├── preprocess_datasets.py
├── source_data_quality_audit.csv
├── raw_preprocessing_summary.csv
├── combined_dataset_processed.csv
├── clustered_players_based_on_gaze_processed.csv
├── louvain_community_partition_validated.csv
├── cluster_membership_and_grid.csv
├── result_validation_summary.csv
├── raw/                          # 24 validated participant files
├── cluster_means/                # 4 aligned, uniform aggregate trajectories
├── features/                     # finite next-step feature table
└── results/                      # 3 validated historical result tables
```

## Downstream usage

Use the processed files as follows:

- Clustering/visualization: `raw/*.csv` or `combined_dataset_processed.csv`.
- Feature-based next-step modeling: `features/gaze_features_processed.csv`.
- Adaptive FC, FKF, STARE, and LSTM experiments: select the desired `cluster_means/cluster_<id>_mean_processed.csv`.
- Result analysis only: `results/*_validated.csv`.

The executable scripts under `Working/` were subsequently updated to use equal-length copies under `trucate_files/`. Participant-level utilities load `trucate_files/raw/`; cluster-0 feature and deep-model scripts load `trucate_files/cluster_means/cluster_0_mean_processed.csv`; Adaptive FC and FKF load the corresponding processed cluster 3 file. The non-truncated files described in this report remain the source layer for creating those equal-length copies.

## Final validation

The generated collection passed these checks:

- all raw copies exactly equal their sources;
- combined row count equals the sum of all participant rows;
- every cluster mean has finite x/y values and one strictly positive uniform time step;
- engineered numeric features contain no NaN or infinity;
- target columns align one row ahead;
- all result errors equal their coordinate-derived Euclidean errors;
- original project datasets were not modified.
