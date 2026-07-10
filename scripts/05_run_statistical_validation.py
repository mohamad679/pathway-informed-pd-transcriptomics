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
    parser.add_argument("--start-permutation-index", type=int, default=1)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--permutation-seed", type=int, default=20260710)
    parser.add_argument("--bootstrap-seed", type=int, default=20260710)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--output-suffix", default="")
    parser.add_argument("--permutation-output-only", action="store_true")
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


def output_path(path: Path, suffix: str) -> Path:
    if not suffix:
        return path
    normalized_suffix = suffix if suffix.startswith("_") else f"_{suffix}"
    return path.with_name(f"{path.stem}{normalized_suffix}{path.suffix}")


def bootstrap_markdown_table(bootstrap_df: pd.DataFrame) -> str:
    rows = ["| Metric | Estimate | CI lower | CI upper | n bootstrap |"]
    rows.append("| --- | ---: | ---: | ---: | ---: |")
    for row in bootstrap_df.itertuples(index=False):
        rows.append(
            f"| {row.metric} | {row.estimate:.6f} | {row.ci_lower:.6f} | "
            f"{row.ci_upper:.6f} | {int(row.n_bootstrap)} |"
        )
    return "\n".join(rows)


def write_report(summary: dict[str, Any], bootstrap_df: pd.DataFrame, report_path: Path) -> None:
    if summary["fast_smoke"]:
        run_label = "Smoke-only run"
    elif summary["run_mode"] == "batch_chunk":
        run_label = "Batch/chunk run"
    else:
        run_label = "Configured Phase 5 run"
    smoke_note = (
        "\n\n**Smoke-only:** These outputs are runtime checks and are not the final Phase 5 result.\n"
        if summary["fast_smoke"]
        else ""
    )
    batch_note = (
        "\n\n**Batch/chunk:** This run covers permutation indices "
        f"{summary['start_permutation_index']} through {summary['end_permutation_index']} "
        "and may include resumed rows from an existing permutation null file.\n"
        if summary["run_mode"] == "batch_chunk"
        else ""
    )
    bootstrap_table = bootstrap_markdown_table(bootstrap_df)
    report_path.write_text(
        "# Phase 5 statistical validation foundation\n\n"
        f"{run_label}.{smoke_note}{batch_note}\n"
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
        f"- Requested permutation range: {summary['start_permutation_index']} to "
        f"{summary['end_permutation_index']}\n"
        f"- Requested permutations this run: {summary['requested_n_permutations']}\n"
        f"- Completed permutations in requested range: {summary['completed_n_permutations']}\n"
        f"- Skipped existing permutations: {summary['skipped_existing_permutations']}\n"
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
    start_permutation_index = int(args.start_permutation_index)
    end_permutation_index = start_permutation_index + n_permutations - 1
    if n_permutations < 1:
        raise ValueError("n_permutations must be at least 1")
    if start_permutation_index < 1:
        raise ValueError("start_permutation_index must be at least 1")
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be at least 1")

    calibration_bins_path = output_path(CALIBRATION_BINS_PATH, args.output_suffix)
    bootstrap_ci_path = output_path(BOOTSTRAP_CI_PATH, args.output_suffix)
    permutation_null_path = output_path(PERMUTATION_NULL_PATH, args.output_suffix)
    summary_path = output_path(SUMMARY_PATH, args.output_suffix)
    report_path = output_path(REPORT_PATH, args.output_suffix)

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
    append_existing = None
    if args.resume is not None:
        if not args.resume.exists():
            raise FileNotFoundError(f"--resume path does not exist: {args.resume}")
        append_existing = pd.read_csv(args.resume)
    existing_indices = (
        set(append_existing["permutation_index"].astype(int).tolist())
        if append_existing is not None and "permutation_index" in append_existing.columns
        else set()
    )
    requested_indices = set(range(start_permutation_index, end_permutation_index + 1))
    skipped_existing_permutations = len(requested_indices & existing_indices)
    permutation_df = run_label_permutation_binn_cv(
        X,
        y,
        folds,
        pathway_mask,
        n_permutations=n_permutations,
        seed=args.permutation_seed,
        training_kwargs=training_kwargs_from_args(args),
        start_permutation_index=start_permutation_index,
        append_existing=append_existing,
    )
    completed_n_permutations = int(
        permutation_df["permutation_index"].astype(int).isin(requested_indices).sum()
    )
    null_scores = permutation_df["null_auroc"].to_numpy(dtype=float)
    p_value = empirical_p_value(observed_metrics["auroc"], null_scores)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    write_calibration_bootstrap = (not args.permutation_output_only) or bool(args.output_suffix)
    if write_calibration_bootstrap:
        calibration_bins_df.to_csv(calibration_bins_path, index=False)
        bootstrap_df.to_csv(bootstrap_ci_path, index=False)
    permutation_df.to_csv(permutation_null_path, index=False)

    is_batch_chunk = (
        start_permutation_index != 1
        or args.resume is not None
        or bool(args.output_suffix)
        or bool(args.permutation_output_only)
    )
    run_mode = "smoke_only" if args.fast_smoke else "batch_chunk" if is_batch_chunk else "configured"

    summary: dict[str, Any] = {
        "phase": 5,
        "scope": "development-only",
        "fast_smoke": bool(args.fast_smoke),
        "run_mode": run_mode,
        "observed_auroc": float(observed_metrics["auroc"]),
        "n_permutations": int(len(permutation_df)),
        "start_permutation_index": int(start_permutation_index),
        "end_permutation_index": int(end_permutation_index),
        "requested_n_permutations": int(n_permutations),
        "completed_n_permutations": int(completed_n_permutations),
        "skipped_existing_permutations": int(skipped_existing_permutations),
        "output_suffix": str(args.output_suffix),
        "permutation_output_only": bool(args.permutation_output_only),
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
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(summary, bootstrap_df, report_path)

    decision_block = f"""## 2026-07-10 - Phase 5 development-only statistical validation

- Ran a configurable development-only calibration, bootstrap CI, and label-permutation foundation.
- Label permutations shuffled labels only within development data and reused the predefined development folds.
- No external cohort or held-out NDD data was loaded or used; this is not final validation and does not freeze a model.
- Run mode: {'fast smoke only, not the final Phase 5 result' if args.fast_smoke else 'batch/chunk run' if is_batch_chunk else 'configured Phase 5 run'}; requested permutations={n_permutations}; completed requested range={completed_n_permutations}; bootstrap={n_bootstrap}."""
    append_if_missing(DECISION_LOG_PATH, decision_block)

    print(f"Observed AUROC: {observed_metrics['auroc']:.6f}")
    print(f"Permutation count: {len(permutation_df)}")
    print(f"Requested permutation range: {start_permutation_index}-{end_permutation_index}")
    print(f"Skipped existing permutations: {skipped_existing_permutations}")
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
