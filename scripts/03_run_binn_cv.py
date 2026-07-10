"""Run development-only pathway-constrained BINN cross-validation."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from models.binn_training import load_binn_inputs, run_binn_cv

PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results" / "development"
REPORT_PATH = ROOT / "docs" / "phase3_binn_training.md"
DECISION_LOG_PATH = ROOT / "docs" / "decision_log.md"
METRICS = ("auroc", "auprc", "balanced_accuracy", "brier", "ece")


def append_if_missing(path: Path, block: str) -> None:
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    if block.strip() not in existing:
        path.write_text(existing.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")


def main() -> int:
    X, y, folds, pathway_mask = load_binn_inputs(
        PROCESSED_DIR / "dev_X.npy", PROCESSED_DIR / "dev_y.npy",
        PROCESSED_DIR / "dev_folds.json", PROCESSED_DIR / "pathway_mask.npz",
    )
    metrics_df, oof_df = run_binn_cv(X, y, folds, pathway_mask)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(RESULTS_DIR / "binn_cv.csv", index=False)
    oof_df.to_csv(RESULTS_DIR / "binn_oof_predictions.csv", index=False)
    means = metrics_df.loc[:, METRICS].mean()
    max_masked = float(metrics_df["max_abs_masked_weight_after_training"].max())
    REPORT_PATH.write_text(
        "# Phase 3 BINN development-only training\n\n"
        "This is development-only cross-validation, not final performance or external validation.\n\n"
        "- Data: `dev_X.npy`, `dev_y.npy`, `dev_folds.json`, and `pathway_mask.npz` only.\n"
        "- Seeds: 11, 23, 37; train-only `StandardScaler`; BCEWithLogitsLoss and Adam.\n"
        "- No external cohort or held-out NDD data was loaded or used.\n"
        "- Off-mask weights were hard-zeroed after every optimizer step and after training.\n\n"
        "## Mean fold/seed metrics\n\n| AUROC | AUPRC | Balanced accuracy | Brier | ECE | Max masked weight |\n| ---: | ---: | ---: | ---: | ---: | ---: |\n"
        f"| {means['auroc']:.6f} | {means['auprc']:.6f} | {means['balanced_accuracy']:.6f} | {means['brier']:.6f} | {means['ece']:.6f} | {max_masked:.1f} |\n",
        encoding="utf-8",
    )
    append_if_missing(DECISION_LOG_PATH, """## 2026-07-10 — Phase 3 development-only BINN CV

- Trained the pathway-constrained BINN only on predefined development folds with seeds 11, 23, and 37.
- Scaling was fit on each training partition only; no external cohort or held-out NDD data was used.
- Off-mask weights were re-zeroed after each optimizer step and verified exactly zero after training.
- This is a development-only training gate, not a final or external-validation performance claim.""")
    print("Development-only pathway-constrained BINN cross-validation")
    print(metrics_df[["model", "seed", "fold", *METRICS]].to_string(index=False))
    print("Mean metrics across all folds/seeds:")
    for metric in METRICS:
        print(f"  {metric}: {means[metric]:.6f}")
    print(f"Max masked weight after training: {max_masked:.1f}")
    print("Confirmation no external/NDD data used: yes")
    print("This is development-only and not final/external validation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
