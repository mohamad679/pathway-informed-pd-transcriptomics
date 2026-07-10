from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from eval.phase4_gate import (  # noqa: E402
    load_phase4_outputs,
    validate_activation_outputs,
    validate_agreement_outputs,
    validate_ig_outputs,
)
from eval.phase5_gate import (  # noqa: E402
    load_phase5_outputs,
    validate_bootstrap_output,
    validate_calibration_output,
    validate_permutation_output,
    validate_summary,
)
from eval.phase6_gate import (  # noqa: E402
    EXPECTED_FROZEN_COMMIT,
    EXPECTED_FROZEN_TAG,
    load_phase6_outputs,
    validate_external_metrics,
    validate_external_predictions,
    validate_frozen_hash_chain,
    validate_ndd_predictions,
    validate_ndd_summary,
    validate_scoring_audit,
)
from eval.phase7_gate import AUTHORITATIVE_VALUES, is_lfs_pointer  # noqa: E402


REQUIRED_FILES = (
    "README.md",
    "requirements.txt",
    "docs/limitations.md",
    "docs/methods.md",
    "docs/results.md",
    "docs/reproducibility.md",
    "frozen/HASH_BEFORE.txt",
    "frozen/HASH_AFTER.txt",
    "frozen/model_v1.pt",
    "frozen/preprocessing_config.json",
    "frozen/gene_space.txt",
    "frozen/pathway_names.txt",
    "frozen/pathway_mask.npz",
    "frozen/training_metadata.json",
    "results/development/baseline_summary.csv",
    "results/development/binn_cv.csv",
    "results/development/binn_oof_predictions.csv",
    "results/development/pathway_activation_stability.csv",
    "results/development/pathway_ig_stability.csv",
    "results/development/pathway_attribution_global_agreement.csv",
    "results/development/pathway_attribution_seed_fold_agreement.csv",
    "results/development/pathway_rna_processing_tier.csv",
    "results/development/statistical_validation_bootstrap_ci.csv",
    "results/development/statistical_validation_permutation_null.csv",
    "results/development/statistical_validation_summary.json",
    "results/external/external_metrics.json",
    "results/external/external_predictions.csv",
    "results/external/ndd_predictions.csv",
    "results/external/ndd_specificity_summary.json",
    "results/external/scoring_audit.json",
)


def verify_reproducibility(root: Path = ROOT) -> dict[str, Any]:
    missing = [path for path in REQUIRED_FILES if not (root / path).is_file()]
    if missing:
        raise FileNotFoundError(f"missing required committed files: {missing}")

    model_path = root / "frozen" / "model_v1.pt"
    if is_lfs_pointer(model_path):
        raise ValueError("frozen/model_v1.pt is a Git LFS pointer; run git lfs pull")

    hash_chain = validate_frozen_hash_chain(root / "frozen")
    _verify_phase_gate_reports(root)
    _verify_frozen_tag(root)
    _verify_no_raw_processed_required(root)
    _verify_phase4(root)
    phase5_summary = _verify_phase5(root)
    phase6_summary = _verify_phase6(root)
    _verify_authoritative_metrics(root, phase5_summary, phase6_summary)

    return {
        "status": "PASS",
        "required_files": len(REQUIRED_FILES),
        "hash_before_after_equal": hash_chain["hash_before_after_equal"],
        "phase_gates": 6,
        "frozen_tag": EXPECTED_FROZEN_TAG,
        "frozen_commit": EXPECTED_FROZEN_COMMIT,
        "raw_processed_required": False,
        "external_rescored": False,
    }


def _verify_phase_gate_reports(root: Path) -> None:
    missing_pass = []
    for phase in range(1, 7):
        path = root / "docs" / f"phase{phase}_gate.md"
        if not path.is_file() or "PASS" not in path.read_text(encoding="utf-8"):
            missing_pass.append(str(path.relative_to(root)))
    if missing_pass:
        raise ValueError(f"phase gate reports missing PASS: {missing_pass}")


def _verify_frozen_tag(root: Path) -> None:
    git_dir = root / ".git"
    if not git_dir.exists():
        return
    completed = subprocess.run(
        ["git", "tag", "--list", EXPECTED_FROZEN_TAG],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise ValueError("could not inspect local git tags")
    if EXPECTED_FROZEN_TAG not in completed.stdout.splitlines():
        raise ValueError(f"local git tag missing: {EXPECTED_FROZEN_TAG}")


def _verify_no_raw_processed_required(root: Path) -> None:
    forbidden = ("data/" + "raw", "data/" + "processed")
    required_text = "\n".join(REQUIRED_FILES)
    if any(path in required_text for path in forbidden):
        raise ValueError("artifact-only verification must not require raw or processed data")
    for name in ("ext_X", "ext_y", "ndd_X", "dev_X", "dev_y"):
        if name in required_text:
            raise ValueError(f"artifact-only verification must not require {name}")


def _verify_phase4(root: Path) -> None:
    outputs = load_phase4_outputs(root / "results" / "development")
    validate_activation_outputs(outputs["activation_scores"], outputs["activation_stability"])
    validate_ig_outputs(outputs["ig_scores"], outputs["ig_stability"])
    validate_agreement_outputs(
        outputs["seed_fold_agreement"],
        outputs["global_agreement"],
        outputs["rna_processing_tier"],
    )


def _verify_phase5(root: Path) -> dict[str, Any]:
    outputs = load_phase5_outputs(root / "results" / "development")
    validate_permutation_output(outputs["permutation_df"])
    validate_summary(outputs["summary"], outputs["permutation_df"], outputs["source_report_text"])
    validate_bootstrap_output(outputs["bootstrap_df"], outputs["summary"])
    validate_calibration_output(outputs["calibration_df"], outputs["summary"])
    return outputs["summary"]


def _verify_phase6(root: Path) -> dict[str, Any]:
    outputs = load_phase6_outputs(root / "frozen", root / "results" / "external")
    validate_external_predictions(outputs["external_predictions"])
    validate_external_metrics(outputs["external_metrics"])
    validate_ndd_predictions(outputs["ndd_predictions"])
    validate_ndd_summary(outputs["ndd_summary"])
    validate_scoring_audit(outputs["scoring_audit"], outputs["external_metrics"])
    return {
        "external_metrics": outputs["external_metrics"],
        "ndd_summary": outputs["ndd_summary"],
    }


def _verify_authoritative_metrics(
    root: Path,
    phase5_summary: dict[str, Any],
    phase6_summary: dict[str, Any],
) -> None:
    import pandas as pd

    binn_cv = pd.read_csv(root / "results" / "development" / "binn_cv.csv")
    baseline = pd.read_csv(root / "results" / "development" / "baseline_summary.csv")
    global_agreement = pd.read_csv(root / "results" / "development" / "pathway_attribution_global_agreement.csv")
    seed_fold = pd.read_csv(root / "results" / "development" / "pathway_attribution_seed_fold_agreement.csv")
    bootstrap = pd.read_csv(root / "results" / "development" / "statistical_validation_bootstrap_ci.csv").set_index("metric")

    checks = {
        "binn_mean_auroc": float(binn_cv["auroc"].mean()),
        "binn_mean_auprc": float(binn_cv["auprc"].mean()),
        "binn_mean_balanced_accuracy": float(binn_cv["balanced_accuracy"].mean()),
        "binn_mean_brier": float(binn_cv["brier"].mean()),
        "binn_mean_ece": float(binn_cv["ece"].mean()),
        "pooled_auroc": float(phase5_summary["observed_auroc"]),
        "bootstrap_auroc_lower": float(bootstrap.loc["auroc", "ci_lower"]),
        "bootstrap_auroc_upper": float(bootstrap.loc["auroc", "ci_upper"]),
        "null_auroc_mean": float(phase5_summary["null_auroc_mean"]),
        "null_auroc_std": float(phase5_summary["null_auroc_std"]),
        "empirical_p_value": float(phase5_summary["empirical_p_value"]),
        "baseline_logistic_auroc": float(baseline.loc[baseline["model"] == "logistic_regression", "auroc_mean"].iloc[0]),
        "baseline_random_forest_auroc": float(baseline.loc[baseline["model"] == "random_forest", "auroc_mean"].iloc[0]),
        "baseline_mlp_auroc": float(baseline.loc[baseline["model"] == "unconstrained_mlp", "auroc_mean"].iloc[0]),
        "mean_seed_fold_spearman": float(seed_fold["spearman_rank_correlation"].mean()),
        "global_spearman": float(global_agreement.iloc[0]["spearman_rank_correlation"]),
        "global_top20_overlap": int(global_agreement.iloc[0]["top20_overlap"]),
        "activation_top20_rna_count": int(global_agreement.iloc[0]["activation_top20_rna_count"]),
        "ig_top20_rna_count": int(global_agreement.iloc[0]["ig_top20_rna_count"]),
        "external_auroc": float(phase6_summary["external_metrics"]["auroc"]),
        "external_auprc": float(phase6_summary["external_metrics"]["auprc"]),
        "external_balanced_accuracy": float(phase6_summary["external_metrics"]["balanced_accuracy"]),
        "external_brier": float(phase6_summary["external_metrics"]["brier"]),
        "external_ece": float(phase6_summary["external_metrics"]["ece"]),
        "ndd_mean_pd_probability": float(phase6_summary["ndd_summary"]["mean_pd_probability"]),
        "ndd_fraction_predicted_pd": float(phase6_summary["ndd_summary"]["fraction_predicted_pd_at_0_5"]),
    }
    for key, observed in checks.items():
        expected = AUTHORITATIVE_VALUES[key]
        if abs(float(observed) - float(expected)) > 1e-6:
            raise ValueError(f"authoritative metric mismatch for {key}: observed={observed}, expected={expected}")


def main() -> int:
    try:
        summary = verify_reproducibility(ROOT)
    except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        print("FAIL")
        print(f"reproducibility verification failure: {error}")
        return 1
    print("PASS")
    print(f"required files verified: {summary['required_files']}")
    print("Git LFS model binary: present")
    print("HASH_BEFORE == HASH_AFTER: yes")
    print("Phase 1-6 PASS reports: yes")
    print(f"frozen tag/commit: {summary['frozen_tag']} / {summary['frozen_commit']}")
    print("authoritative metrics: match committed artifacts")
    print("artifact-only verification required raw/processed arrays: no")
    print("external/NDD rescoring: no")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
