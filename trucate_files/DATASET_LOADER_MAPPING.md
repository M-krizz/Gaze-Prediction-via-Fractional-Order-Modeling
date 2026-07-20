# Dataset Loader Mapping

All executable experiment scripts under `Working/` now load data from `trucate_files/` using paths resolved from each script's location.

| Code file | Truncated input |
|---|---|
| `Working/Combine_dataset.py` | all CSVs in `trucate_files/raw/` |
| `Working/Clustering.py` | all CSVs in `trucate_files/raw/` |
| `Working/DBSCAN.py` | all CSVs in `trucate_files/raw/` |
| `Working/GCN.py` | all CSVs in `trucate_files/raw/` |
| `Working/louvain.py` | all CSVs in `trucate_files/raw/` |
| `Working/P01_plot.py` | `trucate_files/raw/P01_PLAY.csv` |
| `Working/RandomForest.py` | `trucate_files/cluster_means/cluster_0_mean_processed.csv` |
| `Working/STARE_No_FC.py` | `trucate_files/cluster_means/cluster_0_mean_processed.csv` |
| `Working/LSTM.py` | `trucate_files/cluster_means/cluster_0_mean_processed.csv` |
| `Working/Stare_FC/STARE.py` | `trucate_files/cluster_means/cluster_0_mean_processed.csv` |
| `Working/Adaptive_FC.py` | `trucate_files/cluster_means/cluster_3_mean_processed.csv` |
| `Working/FKP.py` | `trucate_files/cluster_means/cluster_3_mean_processed.csv` |

`preprocessed files/preprocess_datasets.py` and `trucate_files/truncate_datasets.py` were intentionally not redirected. They are data-generation utilities: the first must read authoritative source data, and the second must read the full preprocessed data to regenerate the truncated layer. Pointing either at `trucate_files/` would create a circular and lossy pipeline.

Output locations were not changed. Some outputs remain relative to the directory from which a script is launched, as documented in the main README.
