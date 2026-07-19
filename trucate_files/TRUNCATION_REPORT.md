# Dataset Truncation Report

## Outcome

All 33 sample-level preprocessed datasets were truncated to exactly **6,284 rows** and saved under `trucate_files/`. The original files under `preprocessed files/` were not modified.

The minimum was established by:

```text
preprocessed files/results/stare_no_fc_actual_vs_predicted_validated.csv
Original rows: 6,284
```

Each larger dataset retains its header and first 6,284 data records. This preserves chronological order for time-series files and implements truncation without resampling, random selection, padding, aggregation, or additional preprocessing.

## Included datasets

- 24 validated participant gaze recordings from `preprocessed files/raw/`;
- 4 uniform cluster-mean trajectories from `preprocessed files/cluster_means/`;
- `combined_dataset_processed.csv`;
- `features/gaze_features_processed.csv`;
- 3 validated historical prediction tables from `preprocessed files/results/`.

The original directory structure and filenames are mirrored inside `trucate_files/`.

## Files intentionally excluded

The following CSVs were analyzed but not truncated because they are metadata or audit reports, not sample-level datasets:

| File | Rows | Reason for exclusion |
|---|---:|---|
| `cluster_membership_and_grid.csv` | 4 | One record per cluster |
| `clustered_players_based_on_gaze_processed.csv` | 24 | One record per participant assignment |
| `louvain_community_partition_validated.csv` | 24 | One record per participant/community |
| `raw_preprocessing_summary.csv` | 24 | One audit record per raw file |
| `result_validation_summary.csv` | 3 | One audit record per result table |
| `source_data_quality_audit.csv` | 34 | One audit record per source CSV |

Including these entity-level reports in the minimum calculation would have truncated the time-series data to only three or four samples and destroyed their downstream utility. These files remain available unchanged under `preprocessed files/`.

## Preservation guarantees

Every truncated dataset was validated against its source. All 33 passed the following checks:

- exactly 6,284 output rows;
- identical column names and column order;
- identical Pandas-inferred data types;
- exact equality with the first 6,284 source rows;
- preserved missing-value pattern for the retained prefix;
- no additional scaling, normalization, encoding, imputation, or outlier treatment.

The final truncated collection contains:

- 207,372 data rows in total (`33 × 6,284`);
- zero missing cells in the retained records;
- approximately 10 MB of CSV data and supporting documentation/code.

The historical FKF source contains one undefined `alpha` in its final row, but that row lies after position 6,284 and is naturally excluded by truncation. It was not imputed or otherwise altered.

## Reproducibility

Run:

```bash
/usr/bin/python3 "trucate_files/truncate_datasets.py"
```

The script:

1. discovers the 33 intended sample-level preprocessed CSVs;
2. counts their data rows without counting headers;
3. calculates the minimum row count;
4. writes the first `minimum` records while mirroring source paths;
5. creates `truncation_manifest.csv` with source size, output size, removed rows, and column count.

## Manifest

See `truncation_manifest.csv` for the complete file-by-file change summary. Its rows describe the truncated datasets and are not themselves part of the equal-length dataset collection.

## Important interpretation note

Equal row counts do not make heterogeneous tables directly mergeable. Raw recordings, engineered features, cluster means, and prediction results have different schemas and roles. The truncated copies are length-matched as requested, but downstream code should continue to use each category for the task documented in the main README and preprocessing report.
