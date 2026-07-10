"""Score the frozen BINN once on external PD/HC and held-out NDD cohorts."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from eval.frozen_scoring import (
    apply_frozen_preprocessing,
    compute_external_metrics,
    load_and_verify_frozen_bundle,
    score_frozen_model,
    summarize_ndd_specificity,
)
from models.frozen_bundle import verify_hash_manifest


REQUIRED_FROZEN_TAG = "frozen-v1"
THRESHOLD = 0.5
FROZEN_DIR = ROOT / "frozen"
HASH_BEFORE_PATH = FROZEN_DIR / "HASH_BEFORE.txt"

PROCESSED_DIR = ROOT / "data" / "processed"
EXTERNAL_X_PATH = PROCESSED_DIR / "ext_X.npy"
EXTERNAL_Y_PATH = PROCESSED_DIR / "ext_y.npy"
EXTERNAL_SAMPLE_IDS_PATH = PROCESSED_DIR / "ext_sample_ids.txt"
NDD_X_PATH = PROCESSED_DIR / "ndd_X.npy"
NDD_SAMPLE_IDS_PATH = PROCESSED_DIR / "ndd_sample_ids.txt"

RESULTS_DIR = ROOT / "results" / "external"
EXTERNAL_PREDICTIONS_PATH = RESULTS_DIR / "external_predictions.csv"
EXTERNAL_METRICS_PATH = RESULTS_DIR / "external_metrics.json"
NDD_PREDICTIONS_PATH = RESULTS_DIR / "ndd_predictions.csv"
NDD_SUMMARY_PATH = RESULTS_DIR / "ndd_specificity_summary.json"
SCORING_AUDIT_PATH = RESULTS_DIR / "scoring_audit.json"
REPORT_PATH = ROOT / "docs" / "phase6_external_validation.md"
DECISION_LOG_PATH = ROOT / "docs" / "decision_log.md"
RESULT_OUTPUTS = (
    EXTERNAL_PREDICTIONS_PATH,
    EXTERNAL_METRICS_PATH,
    NDD_PREDICTIONS_PATH,
    NDD_SUMMARY_PATH,
    SCORING_AUDIT_PATH,
)


def _git_output(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _verify_frozen_git_tag() -> tuple[str, str]:
    frozen_commit = _git_output("rev-parse", "HEAD")
    tags_at_head = set(_git_output("tag", "--points-at", "HEAD").splitlines())
    if REQUIRED_FROZEN_TAG not in tags_at_head:
        raise RuntimeError(
            f"Required tag {REQUIRED_FROZEN_TAG} does not point to current commit {frozen_commit}"
        )
    tag_commit = _git_output("rev-list", "-n", "1", REQUIRED_FROZEN_TAG)
    if tag_commit != frozen_commit:
        raise RuntimeError(
            f"Required tag {REQUIRED_FROZEN_TAG} resolves to {tag_commit}, not {frozen_commit}"
        )
    return frozen_commit, REQUIRED_FROZEN_TAG


def _require_result_outputs_absent() -> None:
    existing = [str(path.relative_to(ROOT)) for path in RESULT_OUTPUTS if path.exists()]
    if existing:
        raise FileExistsError(
            "One-time Phase 6 scoring outputs already exist; refusing to rescore or overwrite: "
            f"{existing}"
        )


def _read_sample_ids(path: Path) -> list[str]:
    sample_ids = path.read_text(encoding="utf-8").splitlines()
    if not sample_ids or any(not sample_id for sample_id in sample_ids):
        raise ValueError(f"Sample ID file must contain only nonempty lines: {path}")
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError(f"Sample IDs must be unique within cohort: {path}")
    return sample_ids


def _relative_paths(paths: tuple[Path, ...]) -> list[str]:
    return [str(path.relative_to(ROOT)) for path in paths]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _append_if_missing(path: Path, block: str) -> None:
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    if block.strip() not in existing:
        path.write_text(existing.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")


def _write_report(
    *,
    frozen_commit: str,
    frozen_tag: str,
    external_metrics: dict[str, Any],
    ndd_summary: dict[str, Any],
) -> None:
    REPORT_PATH.write_text(
        "# Phase 6 frozen-model external validation\n\n"
        "This report records the one-time inference-only Phase 6 execution.\n\n"
        "## Chain of custody\n\n"
        f"- Frozen commit: `{frozen_commit}`\n"
        f"- Frozen tag: `{frozen_tag}`\n"
        "- `frozen-v1` was verified before loading external or NDD inputs.\n"
        "- `HASH_BEFORE.txt` was verified before scoring.\n"
        "- No model retraining, weight update, or preprocessing fitting occurred.\n"
        "- The external cohort was scored once and the NDD cohort was scored once.\n"
        "- The classification threshold was fixed at exactly `0.5`; outcomes were not used for tuning or selection.\n\n"
        "## External PD/HC cohort\n\n"
        f"- Samples: `{external_metrics['n_samples']}` (`{external_metrics['n_hc']}` HC, "
        f"`{external_metrics['n_pd']}` PD)\n"
        f"- AUROC: `{external_metrics['auroc']:.6f}`\n"
        f"- AUPRC: `{external_metrics['auprc']:.6f}`\n"
        f"- Balanced accuracy at 0.5: `{external_metrics['balanced_accuracy']:.6f}`\n"
        f"- Brier score: `{external_metrics['brier']:.6f}`\n"
        f"- ECE: `{external_metrics['ece']:.6f}`\n\n"
        "## Held-out NDD specificity stress test\n\n"
        "The NDD cohort has no binary target label in this analysis. This is a specificity/stress-test "
        "summary only and is not diagnostic validation.\n\n"
        f"- Samples: `{ndd_summary['n_samples']}`\n"
        f"- Mean PD probability: `{ndd_summary['mean_pd_probability']:.6f}`\n"
        f"- Fraction predicted PD at 0.5: `{ndd_summary['fraction_predicted_pd_at_0_5']:.6f}`\n\n"
        "These results are research validation only, not clinical validation, and do not support "
        "clinical, diagnostic, or deployment claims.\n",
        encoding="utf-8",
    )


def main() -> int:
    frozen_commit, frozen_tag = _verify_frozen_git_tag()

    # Chain of custody: this must remain before any external or NDD np.load call.
    hash_before_verified = verify_hash_manifest(FROZEN_DIR, HASH_BEFORE_PATH)
    _require_result_outputs_absent()
    model, scaler_parameters, frozen_metadata = load_and_verify_frozen_bundle(FROZEN_DIR)
    expected_gene_count = len(frozen_metadata["gene_space"])

    required_input_paths = (
        EXTERNAL_X_PATH,
        EXTERNAL_Y_PATH,
        EXTERNAL_SAMPLE_IDS_PATH,
        NDD_X_PATH,
        NDD_SAMPLE_IDS_PATH,
    )
    missing_inputs = [str(path) for path in required_input_paths if not path.exists()]
    if missing_inputs:
        raise FileNotFoundError(f"Missing Phase 6 scoring inputs: {missing_inputs}")

    external_X = np.load(EXTERNAL_X_PATH, allow_pickle=False)
    external_y = np.load(EXTERNAL_Y_PATH, allow_pickle=False)
    external_sample_ids = _read_sample_ids(EXTERNAL_SAMPLE_IDS_PATH)
    ndd_X = np.load(NDD_X_PATH, allow_pickle=False)
    ndd_sample_ids = _read_sample_ids(NDD_SAMPLE_IDS_PATH)

    if external_X.ndim != 2:
        raise ValueError("External X must be two-dimensional")
    if external_y.ndim != 1:
        raise ValueError("External y must be one-dimensional")
    if external_X.shape[0] != external_y.shape[0] or external_X.shape[0] != len(
        external_sample_ids
    ):
        raise ValueError("External rows must match ext_y and ext_sample_ids")
    if external_X.shape[1] != expected_gene_count:
        raise ValueError("External feature count must equal the frozen gene count")
    if not np.all(np.isin(np.unique(external_y), (0, 1))):
        raise ValueError("External labels must contain only HC=0 and PD=1")
    if np.unique(external_y).size != 2:
        raise ValueError("External labels must contain both HC=0 and PD=1")
    if ndd_X.ndim != 2:
        raise ValueError("NDD X must be two-dimensional")
    if ndd_X.shape[0] != len(ndd_sample_ids):
        raise ValueError("NDD rows must match ndd_sample_ids")
    if ndd_X.shape[1] != expected_gene_count:
        raise ValueError("NDD feature count must equal the frozen gene count")

    external_scaled = apply_frozen_preprocessing(
        external_X, scaler_parameters, expected_gene_count
    )
    ndd_scaled = apply_frozen_preprocessing(ndd_X, scaler_parameters, expected_gene_count)

    external_logits, external_probabilities = score_frozen_model(
        model, external_scaled, batch_size=64, device="cpu"
    )
    ndd_logits, ndd_probabilities = score_frozen_model(
        model, ndd_scaled, batch_size=64, device="cpu"
    )

    external_labels = external_y.astype(int, copy=False)
    external_predictions = (external_probabilities >= THRESHOLD).astype(int)
    external_metrics: dict[str, Any] = compute_external_metrics(
        external_labels, external_probabilities
    )
    external_metrics.update(
        {
            "threshold": THRESHOLD,
            "n_samples": int(external_labels.size),
            "n_hc": int(np.sum(external_labels == 0)),
            "n_pd": int(np.sum(external_labels == 1)),
            "frozen_commit": frozen_commit,
            "frozen_tag": frozen_tag,
            "hash_before_verified": bool(hash_before_verified),
            "model_frozen": True,
            "external_scored_once": True,
        }
    )
    ndd_summary = summarize_ndd_specificity(ndd_probabilities, threshold=THRESHOLD)

    scoring_audit = {
        "execution_scope": "one-time frozen-model external PD/HC scoring and held-out NDD specificity stress test",
        "frozen_commit": frozen_commit,
        "frozen_tag": frozen_tag,
        "hash_before_manifest": str(HASH_BEFORE_PATH.relative_to(ROOT)),
        "hash_before_verified": bool(hash_before_verified),
        "external_input_paths": _relative_paths(
            (EXTERNAL_X_PATH, EXTERNAL_Y_PATH, EXTERNAL_SAMPLE_IDS_PATH)
        ),
        "ndd_input_paths": _relative_paths((NDD_X_PATH, NDD_SAMPLE_IDS_PATH)),
        "frozen_payload_modified": False,
        "threshold": THRESHOLD,
        "scaler_refit": False,
        "model_retrained": False,
        "external_metric_used_for_selection": False,
        "external_scored_once": True,
        "ndd_scored_once": True,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "sample_id": external_sample_ids,
            "y_true": external_labels,
            "logit": external_logits,
            "y_prob": external_probabilities,
            "y_pred": external_predictions,
        }
    ).to_csv(EXTERNAL_PREDICTIONS_PATH, index=False)
    _write_json(EXTERNAL_METRICS_PATH, external_metrics)
    pd.DataFrame(
        {
            "sample_id": ndd_sample_ids,
            "logit": ndd_logits,
            "y_prob": ndd_probabilities,
            "predicted_class": np.where(
                ndd_probabilities >= THRESHOLD, "PD", "HC"
            ),
        }
    ).to_csv(NDD_PREDICTIONS_PATH, index=False)
    _write_json(NDD_SUMMARY_PATH, ndd_summary)
    _write_json(SCORING_AUDIT_PATH, scoring_audit)
    _write_report(
        frozen_commit=frozen_commit,
        frozen_tag=frozen_tag,
        external_metrics=external_metrics,
        ndd_summary=ndd_summary,
    )
    _append_if_missing(
        DECISION_LOG_PATH,
        f"""## 2026-07-10 - Phase 6 one-time frozen-model scoring

- Verified `{frozen_tag}` at `{frozen_commit}` and verified `HASH_BEFORE.txt` before loading external or NDD inputs.
- Applied only the frozen preprocessing parameters; no model retraining, weight update, scaler refit, threshold tuning, or outcome-based selection occurred.
- Scored the external PD/HC cohort once and the held-out NDD cohort once at the fixed threshold of 0.5.
- Treated the unlabeled NDD analysis only as a specificity/stress test.
- These results are research validation only, not clinical validation, and support no clinical, diagnostic, or deployment claim.""",
    )

    print(f"frozen commit: {frozen_commit}")
    print(f"frozen tag: {frozen_tag}")
    print("HASH_BEFORE verification PASS")
    print(f"external sample count: {external_metrics['n_samples']}")
    print(f"external HC/PD counts: {external_metrics['n_hc']}/{external_metrics['n_pd']}")
    print(f"external AUROC: {external_metrics['auroc']:.6f}")
    print(f"external AUPRC: {external_metrics['auprc']:.6f}")
    print(f"external balanced accuracy: {external_metrics['balanced_accuracy']:.6f}")
    print(f"external Brier: {external_metrics['brier']:.6f}")
    print(f"external ECE: {external_metrics['ece']:.6f}")
    print(f"NDD sample count: {ndd_summary['n_samples']}")
    print(f"NDD mean PD probability: {ndd_summary['mean_pd_probability']:.6f}")
    print(
        "NDD fraction predicted PD at 0.5: "
        f"{ndd_summary['fraction_predicted_pd_at_0_5']:.6f}"
    )
    print("confirmation model retrained: no")
    print("confirmation scaler refit: no")
    print("confirmation external used for selection: no")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
