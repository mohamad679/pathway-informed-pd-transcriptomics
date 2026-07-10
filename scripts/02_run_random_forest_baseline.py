from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from eval.cv import load_dev_folds
from models.random_forest_baseline import DEFAULT_SEEDS, evaluate_random_forest_cv


PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_PATH = ROOT / "results" / "development" / "random_forest_baseline_cv.csv"
REPORT_PATH = ROOT / "docs" / "phase2_random_forest_baseline.md"
DECISION_LOG_PATH = ROOT / "docs" / "decision_log.md"
METRIC_NAMES = ("auroc", "auprc", "balanced_accuracy", "brier", "ece")

DECISION_LOG_ENTRY = """## 2026-07-10 — Phase 2 development-only Random Forest baseline

- Evaluated a Random Forest baseline using only `dev_X.npy`, `dev_y.npy`, and the predefined development folds.
- Used seeds 11, 23, and 37 with 500 trees, `max_features="sqrt"`, and `class_weight="balanced_subsample"`.
- No external cohort or held-out NDD data was loaded or used.
- No MLP, BINN, pathway masks, or MSigDB logic was implemented.
"""


def append_if_missing(path: Path, block: str) -> None:
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    if block.strip() in existing:
        return
    separator = "" if not existing or existing.endswith("\n\n") else "\n"
    path.write_text(existing + separator + block.rstrip() + "\n", encoding="utf-8")


def write_csv(rows: list[dict[str, float | int | str]]) -> None:
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["model", "seed", "fold", *METRIC_NAMES]
    with RESULTS_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, float | int | str]]) -> dict[str, float]:
    return {
        metric: float(np.mean([float(row[metric]) for row in rows]))
        for metric in METRIC_NAMES
    }


def write_report(rows: list[dict[str, float | int | str]], summary: dict[str, float]) -> None:
    best_row = max(rows, key=lambda row: float(row["auroc"]))
    worst_row = min(rows, key=lambda row: float(row["auroc"]))
    report = f"""# Phase 2 Random Forest development-only baseline

This report contains development-only cross-validation results. It is not an external-validation result or a final performance claim.

## Configuration

- Model: `random_forest`
- RandomForestClassifier: `n_estimators=500`, `max_features="sqrt"`, `class_weight="balanced_subsample"`, `n_jobs=-1`
- Seeds: {list(DEFAULT_SEEDS)}
- Data loaded: `data/processed/dev_X.npy`, `data/processed/dev_y.npy`, and `data/processed/dev_folds.json` only
- No scaling was applied.
- No external cohort or held-out NDD data was loaded or used.

## Mean metrics across all folds and seeds

| AUROC | AUPRC | Balanced accuracy | Brier | ECE |
| ---: | ---: | ---: | ---: | ---: |
| {summary['auroc']:.6f} | {summary['auprc']:.6f} | {summary['balanced_accuracy']:.6f} | {summary['brier']:.6f} | {summary['ece']:.6f} |

## Fold AUROC range

- Best: seed {best_row['seed']}, fold {best_row['fold']}, AUROC {float(best_row['auroc']):.6f}
- Worst: seed {worst_row['seed']}, fold {worst_row['fold']}, AUROC {float(worst_row['auroc']):.6f}
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


def main() -> int:
    X = np.load(PROCESSED_DIR / "dev_X.npy")
    y = np.load(PROCESSED_DIR / "dev_y.npy")
    folds = load_dev_folds(PROCESSED_DIR)
    rows = evaluate_random_forest_cv(X, y, folds, seeds=DEFAULT_SEEDS)
    summary = summarize(rows)

    write_csv(rows)
    write_report(rows, summary)
    append_if_missing(DECISION_LOG_PATH, DECISION_LOG_ENTRY)

    print("Development-only Random Forest cross-validation")
    print("Mean metrics across all folds/seeds:")
    for metric in METRIC_NAMES:
        print(f"  {metric}: {summary[metric]:.6f}")
    best_row = max(rows, key=lambda row: float(row["auroc"]))
    worst_row = min(rows, key=lambda row: float(row["auroc"]))
    print(f"Best fold AUROC: {float(best_row['auroc']):.6f} (seed={best_row['seed']}, fold={best_row['fold']})")
    print(f"Worst fold AUROC: {float(worst_row['auroc']):.6f} (seed={worst_row['seed']}, fold={worst_row['fold']})")
    print("Confirmed: no external or held-out NDD data used.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
