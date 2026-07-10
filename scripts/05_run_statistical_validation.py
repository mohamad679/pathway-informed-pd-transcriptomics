"""Run Phase 5 development-only statistical validation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from eval.bootstrap import bootstrap_metric_cis
from eval.calibration import calibration_summary, reliability_curve
from eval.metrics import compute_binary_metrics
from eval.permutation import empirical_p_value, run_label_permutation_binn_cv
from models.binn_training import load_binn_inputs


PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results" / "development"
DOCS_DIR = ROOT / "docs"
BINN_OOF_PATH = RESULTS_DIR / "binn_oof_predictions.csv"
CALIBRATION_BINS_PATH = RESULTS_DIR / "statistical_validation_calibration_bins.csv"
BOOTSTRAP_CI_PATH = RESULTS_DIR / "statistical_validation_bootstrap_ci.csv"
PERMUTATION_NULL_PATH = RESULTS_DIR / "statistical_validation_permutation_null.csv"
SUMMARY_PATH = RESULTS_DIR / "statistical_validation_summary.json"
REPORT_PATH = DOCS_DIR / "phase5_statistical_validation.md"
DECISION_LOG_PATH = DOCS_DIR / "decision_log.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-permutations", type=int, default=1000)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--permutation-seed", type=int, default=20260710)
    parser.add_argument("--bootstrap-seed", type=int, default=20260710)
    parser.add_argument("--fast-smoke", action="store_true")
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--hidden-dim", type=int)
    parser.add_argument("--batch-size", type=int)
    return parser.parse_args()


def append_if_missing(path: Path, block: str) -> None:
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    if block.strip() not in existing:
        path.write_text(existing.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")


def training_kwargs_from_args(args: argparse.Namespace) -> dict[str, int]:
    kwargs: dict[str, int] = {}
    for arg_name, kwarg_name in (
        ("max_epochs", "max_epochs"),
        ("patience", "patience"),
        ("hidden_dim", "hidden_dim"),
        ("batch_size", "batch_size"),
    ):
        value = getattr(args, arg_name)
        if value is not None:
            kwargs[kwarg_name] = int(value)
    return kwargs


def bootstrap_markdown_table(bootstrap_df: pd.DataFrame) -> str:
    rows = ["| Metric | Estimate | CI lower | CI upper | n bootstrap |"]
    rows.append("| --- | ---: | ---: | ---: | ---: |")
    for row in bootstrap_df.itertuples(index=False):
        rows.append(
            f"| {row.metric} | {row.estimate:.6f} | {row.ci_lower:.6f} | "
            f"{row.ci_upper:.6f} | {int(row.n_bootstrap)} |"
        )
    return "\n".join(rows)


def write_report(summary: dict[str, Any], bootstrap_df: pd.DataFrame) -> None:
    run_label = "Smoke-only run" if summary["fast_smoke"] else "Configured Phase 5 run"
    smoke_note = (
        "\n\n**Smoke-only:** These outputs are runtime checks and are not the final Phase 5 result.\n"
        if summary["fast_smoke"]
        else ""
    )
    bootstrap_table = bootstrap_markdown_table(bootstrap_df)
    REPORT_PATH.write_text(
        "# Phase 5 statistical validation foundation\n\n"
        f"{run_label}.{smoke_note}\n"
        "This report is development-only. It evaluates whether the observed development BINN "
        "OOF result is distinguishable from label-shuffled development results and documents "
        "calibration for the existing development OOF predictions.\n\n"
        "- Inputs: `dev_X.npy`, `dev_y.npy`, `dev_folds.json`, `pathway_mask.npz`, and "
        "`binn_oof_predictions.csv` from development outputs only.\n"
        "- Permutation labels are shuffled only inside development data.\n"
        "- No external cohort or held-out NDD data was loaded or used.\n"
        "- This is not final validation, and it does not freeze a model.\n"
        "- No biological interpretation or final performance claim is made.\n\n"
        "## Summary\n\n"
        f"- Observed pooled AUROC: {summary['observed_auroc']:.6f}\n"
        f"- Permutations: {summary['n_permutations']}\n"
        f"- Null AUROC mean: {summary['null_auroc_mean']:.6f}\n"
        f"- Null AUROC std: {summary['null_auroc_std']:.6f}\n"
        f"- Empirical p-value: {summary['empirical_p_value']:.6f}\n"
        f"- Brier: {summary['brier']:.6f}\n"
        f"- ECE: {summary['ece']:.6f}\n\n"
        "## Bootstrap confidence intervals\n\n"
        f"{bootstrap_table}\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    n_permutations = 2 if args.fast_smoke else args.n_permutations
    n_bootstrap = 50 if args.fast_smoke else args.n_bootstrap
    if n_permutations < 1:
        raise ValueError("n_permutations must be at least 1")
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be at least 1")

    X, y, folds, pathway_mask = load_binn_inputs(
        PROCESSED_DIR / "dev_X.npy",
        PROCESSED_DIR / "dev_y.npy",
        PROCESSED_DIR / "dev_folds.json",
        PROCESSED_DIR / "pathway_mask.npz",
    )
    oof_df = pd.read_csv(BINN_OOF_PATH)
    required_columns = {"y_true", "y_prob"}
    if not required_columns <= set(oof_df.columns):
        raise ValueError(f"{BINN_OOF_PATH} must contain columns {sorted(required_columns)}")

    y_true = oof_df["y_true"].to_numpy(dtype=int)
    y_prob = oof_df["y_prob"].to_numpy(dtype=float)
    observed_metrics = compute_binary_metrics(y_true, y_prob)
    calibration_bins_df = reliability_curve(y_true, y_prob)
    calibration = calibration_summary(y_true, y_prob)
    bootstrap_df = bootstrap_metric_cis(
        y_true, y_prob, n_bootstrap=n_bootstrap, seed=args.bootstrap_seed
    )
    permutation_df = run_label_permutation_binn_cv(
        X,
        y,
        folds,
        pathway_mask,
        n_permutations=n_permutations,
        seed=args.permutation_seed,
        training_kwargs=training_kwargs_from_args(args),
    )
    null_scores = permutation_df["null_auroc"].to_numpy(dtype=float)
    p_value = empirical_p_value(observed_metrics["auroc"], null_scores)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    calibration_bins_df.to_csv(CALIBRATION_BINS_PATH, index=False)
    bootstrap_df.to_csv(BOOTSTRAP_CI_PATH, index=False)
    permutation_df.to_csv(PERMUTATION_NULL_PATH, index=False)

    summary: dict[str, Any] = {
        "phase": 5,
        "scope": "development-only",
        "fast_smoke": bool(args.fast_smoke),
        "observed_auroc": float(observed_metrics["auroc"]),
        "n_permutations": int(n_permutations),
        "permutation_seed": int(args.permutation_seed),
        "null_auroc_mean": float(np.mean(null_scores)),
        "null_auroc_std": float(np.std(null_scores, ddof=0)),
        "empirical_p_value": float(p_value),
        "n_bootstrap": int(n_bootstrap),
        "bootstrap_seed": int(args.bootstrap_seed),
        "brier": float(calibration["brier"]),
        "ece": float(calibration["ece"]),
        "n_calibration_bins": int(calibration["n_bins"]),
        "n_samples": int(calibration["n_samples"]),
        "external_or_ndd_used": False,
        "final_validation": False,
        "model_frozen": False,
        "training_kwargs": training_kwargs_from_args(args),
        "inputs": [
            "data/processed/dev_X.npy",
            "data/processed/dev_y.npy",
            "data/processed/dev_folds.json",
            "data/processed/pathway_mask.npz",
            "results/development/binn_oof_predictions.csv",
        ],
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(summary, bootstrap_df)

    decision_block = f"""## 2026-07-10 - Phase 5 development-only statistical validation

- Ran a configurable development-only calibration, bootstrap CI, and label-permutation foundation.
- Label permutations shuffled labels only within development data and reused the predefined development folds.
- No external cohort or held-out NDD data was loaded or used; this is not final validation and does not freeze a model.
- Run mode: {'fast smoke only, not the final Phase 5 result' if args.fast_smoke else 'configured Phase 5 run'}; permutations={n_permutations}; bootstrap={n_bootstrap}."""
    append_if_missing(DECISION_LOG_PATH, decision_block)

    print(f"Observed AUROC: {observed_metrics['auroc']:.6f}")
    print(f"Permutation count: {n_permutations}")
    print(f"Null mean/std: {summary['null_auroc_mean']:.6f}/{summary['null_auroc_std']:.6f}")
    print(f"Empirical p-value: {p_value:.6f}")
    print(f"Brier: {calibration['brier']:.6f}")
    print(f"ECE: {calibration['ece']:.6f}")
    print("Bootstrap CI table:")
    print(bootstrap_df.to_string(index=False))
    print("Confirmation no external/NDD data used: yes")
    if args.fast_smoke:
        print("Fast-smoke outputs are smoke-only and not the final Phase 5 result.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
