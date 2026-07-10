"""Generate development-only OOF predictions and bootstrap baseline summaries."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from eval.baseline_summary import METRIC_NAMES, compute_model_summary_from_predictions, write_baseline_summary_csv, write_baseline_summary_report
from eval.cv import load_dev_arrays, load_dev_folds
from models.logistic_baseline import DEFAULT_SEEDS, evaluate_logistic_regression_cv_with_predictions
from models.mlp_baseline import evaluate_mlp_cv_with_predictions
from models.random_forest_baseline import DEFAULT_N_ESTIMATORS, evaluate_random_forest_cv_with_predictions


PROCESSED_DIR = ROOT / "data" / "processed"
OOF_PATH = ROOT / "results" / "development" / "baseline_oof_predictions.csv"
SUMMARY_PATH = ROOT / "results" / "development" / "baseline_summary.csv"
REPORT_PATH = ROOT / "docs" / "phase2_baseline_summary.md"
DECISION_LOG_PATH = ROOT / "docs" / "decision_log.md"
PREDICTION_COLUMNS = ["model", "seed", "fold", "sample_index", "y_true", "y_prob"]

DECISION_LOG_ENTRY = """## 2026-07-10 — Phase 2 development-only baseline OOF comparison and bootstrap CI

- Re-ran only the completed Logistic Regression, Random Forest, and unconstrained MLP baselines using `dev_X.npy`, `dev_y.npy`, and predefined development folds.
- Exported one out-of-fold probability per validation sample, seed, and fold to support sample-level uncertainty estimation without changing the committed per-fold baseline metric CSVs.
- For each model and seed, pooled OOF predictions across the five validation folds before computing seed-level metrics; reported means are across those seed-level metrics.
- Used 2,000 deterministic paired, class-stratified bootstrap resamples of pooled seed-level OOF prediction rows within model (`random_state=20260710`) for 95% CIs.
- No external cohort or held-out NDD data was loaded or used. No BINN, pathway masks, or MSigDB logic was implemented or used.
"""


def append_if_missing(path: Path, block: str) -> None:
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    if block.strip() in existing:
        return
    separator = "" if not existing or existing.endswith("\n\n") else "\n"
    path.write_text(existing + separator + block.rstrip() + "\n", encoding="utf-8")


def write_prediction_csv(rows: list[dict[str, float | int | str]]) -> None:
    OOF_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OOF_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PREDICTION_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    X, y = load_dev_arrays(PROCESSED_DIR)
    folds = load_dev_folds(PROCESSED_DIR)
    _, logistic_predictions = evaluate_logistic_regression_cv_with_predictions(X, y, folds, seeds=DEFAULT_SEEDS)
    _, forest_predictions = evaluate_random_forest_cv_with_predictions(X, y, folds, seeds=DEFAULT_SEEDS, n_estimators=DEFAULT_N_ESTIMATORS)
    _, mlp_predictions = evaluate_mlp_cv_with_predictions(X, y, folds, seeds=DEFAULT_SEEDS)
    prediction_rows = logistic_predictions + forest_predictions + mlp_predictions

    write_prediction_csv(prediction_rows)
    summary = compute_model_summary_from_predictions(prediction_rows)
    write_baseline_summary_csv(summary, SUMMARY_PATH)
    write_baseline_summary_report(summary, REPORT_PATH)
    append_if_missing(DECISION_LOG_PATH, DECISION_LOG_ENTRY)

    print("Development-only pooled out-of-fold baseline comparison")
    print("model | AUROC mean [95% CI] | AUPRC mean [95% CI] | balanced_accuracy mean [95% CI] | Brier mean [95% CI] | ECE mean [95% CI]")
    for _, row in summary.sort_values("model").iterrows():
        values = []
        for metric in METRIC_NAMES:
            values.append(f"{row[f'{metric}_mean']:.6f} [{row[f'{metric}_ci_lower']:.6f}, {row[f'{metric}_ci_upper']:.6f}]")
        print(f"{row['model']} | " + " | ".join(values))
    print(f"Best model by mean AUROC: {summary.loc[summary['auroc_mean'].idxmax(), 'model']}")
    print("Confirmed: no external cohort or held-out NDD data used.")
    print("Confirmed: bootstrap used development out-of-fold predictions only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
