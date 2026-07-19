"""Create equal-length copies of all row-level preprocessed datasets.

Metadata/report CSVs are intentionally excluded because they describe files,
clusters, or participants rather than containing sample-level observations.
"""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "preprocessed files"
OUTPUT = Path(__file__).resolve().parent


def dataset_paths() -> list[Path]:
    paths = []
    paths.extend(sorted((SOURCE / "raw").glob("*.csv")))
    paths.extend(sorted((SOURCE / "cluster_means").glob("*.csv")))
    paths.extend(sorted((SOURCE / "results").glob("*.csv")))
    paths.extend(
        [
            SOURCE / "combined_dataset_processed.csv",
            SOURCE / "features" / "gaze_features_processed.csv",
        ]
    )
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing expected preprocessed datasets: {missing}")
    return paths


def count_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        try:
            next(reader)
        except StopIteration:
            return 0
        return sum(1 for _ in reader)


def truncate_csv(source: Path, destination: Path, row_limit: int) -> tuple[int, int]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("r", encoding="utf-8", newline="") as input_handle:
        reader = csv.reader(input_handle)
        header = next(reader)
        rows = []
        for index, row in enumerate(reader):
            if index >= row_limit:
                break
            rows.append(row)

    if len(rows) != row_limit:
        raise ValueError(f"{source} has only {len(rows)} rows; expected at least {row_limit}")

    with destination.open("w", encoding="utf-8", newline="") as output_handle:
        writer = csv.writer(output_handle)
        writer.writerow(header)
        writer.writerows(rows)
    return len(header), len(rows)


def main() -> None:
    sources = dataset_paths()
    source_counts = {path: count_rows(path) for path in sources}
    minimum_rows = min(source_counts.values())

    manifest_rows = []
    for source in sources:
        relative = source.relative_to(SOURCE)
        destination = OUTPUT / relative
        column_count, output_rows = truncate_csv(source, destination, minimum_rows)
        manifest_rows.append(
            {
                "source_file": str(source.relative_to(ROOT)),
                "output_file": str(destination.relative_to(ROOT)),
                "input_rows": source_counts[source],
                "output_rows": output_rows,
                "rows_removed": source_counts[source] - output_rows,
                "columns": column_count,
            }
        )

    manifest_path = OUTPUT / "truncation_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"Minimum sample row count: {minimum_rows}")
    print(f"Truncated datasets: {len(sources)}")
    print(f"Output directory: {OUTPUT}")


if __name__ == "__main__":
    main()
