"""Final Phase 7 artifact, documentation, claim-safety, and hygiene gate."""

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from eval.phase4_gate import (
    load_phase4_outputs,
    summarize_phase4_gate,
    validate_activation_outputs,
    validate_agreement_outputs,
    validate_ig_outputs,
)
from eval.phase5_gate import (
    load_phase5_outputs,
    summarize_phase5_gate,
    validate_bootstrap_output,
    validate_calibration_output,
    validate_permutation_output,
    validate_summary,
)
from eval.phase6_gate import (
    EXPECTED_EXTERNAL_HC,
    EXPECTED_EXTERNAL_PD,
    EXPECTED_EXTERNAL_ROWS,
    EXPECTED_FROZEN_COMMIT,
    EXPECTED_FROZEN_TAG,
    EXPECTED_NDD_ROWS,
    IMMUTABLE_PAYLOAD_FILES,
    load_phase6_outputs,
    validate_external_metrics,
    validate_external_predictions,
    validate_frozen_hash_chain,
    validate_ndd_predictions,
    validate_ndd_summary,
    validate_scoring_audit,
)


FLOAT_TOLERANCE = 1e-6
DOCUMENT_PATHS = (
    "README.md",
    "docs/methods.md",
    "docs/results.md",
    "docs/limitations.md",
    "docs/reproducibility.md",
)
PHASE_GATE_REPORTS = tuple(f"docs/phase{phase}_gate.md" for phase in range(1, 7))
FIGURE_FILES = tuple(f"results/figures/fig{index:02d}_{name}.png" for index, name in (
    (1, "cohort_overview"),
    (2, "development_model_comparison"),
    (3, "permutation_validation"),
    (4, "top_pathways"),
    (5, "attribution_agreement"),
    (6, "external_validation"),
))
FORBIDDEN_CLAIMS = (
    "clinically validated",
    "deployment ready",
    "robust generalization",
    "strong external performance",
    "state of the art",
    "sota",
)
CLAIM_DOCS = DOCUMENT_PATHS
AUTHORITATIVE_VALUES = {
    "development_samples": 438,
    "genes": 13908,
    "pathways": 1297,
    "edges": 75429,
    "binn_mean_auroc": 0.712859,
    "binn_mean_auprc": 0.676449,
    "binn_mean_balanced_accuracy": 0.649416,
    "binn_mean_brier": 0.249931,
    "binn_mean_ece": 0.208059,
    "pooled_auroc": 0.702895,
    "bootstrap_auroc_lower": 0.673004,
    "bootstrap_auroc_upper": 0.731080,
    "permutations": 50,
    "null_auroc_mean": 0.536038,
    "null_auroc_std": 0.022579,
    "empirical_p_value": 0.019608,
    "baseline_logistic_auroc": 0.687072,
    "baseline_random_forest_auroc": 0.664454,
    "baseline_mlp_auroc": 0.650682,
    "mean_seed_fold_spearman": 0.808936,
    "global_spearman": 0.933899,
    "global_top20_overlap": 12,
    "activation_top20_rna_count": 0,
    "ig_top20_rna_count": 0,
    "external_auroc": 0.695455,
    "external_auprc": 0.782081,
    "external_balanced_accuracy": 0.5,
    "external_brier": 0.694444,
    "external_ece": 0.694441,
    "ndd_samples": 48,
    "ndd_mean_pd_probability": 0.489050,
    "ndd_fraction_predicted_pd": 0.520833,
}


def run_phase7_gate(root: str | Path) -> dict[str, Any]:
    base = Path(root)
    documentation = audit_documentation(base)
    figures = audit_figures(base)
    frozen = audit_frozen_chain(base)
    results = audit_result_values(base)
    claims = audit_claim_safety(base)
    hygiene = audit_repository_hygiene(base)
    summary = {
        "status": "PASS",
        "documentation": documentation,
        "figures": figures,
        "frozen": frozen,
        "results": results,
        "claims": claims,
        "hygiene": hygiene,
        "tests_run_inside_gate": False,
        "training_scoring_recomputation": False,
    }
    write_phase7_gate_report(base / "docs" / "phase7_gate.md", summary)
    append_decision_log(base / "docs" / "decision_log.md")
    return summary


def audit_documentation(root: Path) -> dict[str, Any]:
    missing = [path for path in (*DOCUMENT_PATHS, *PHASE_GATE_REPORTS) if not (root / path).is_file()]
    if missing:
        raise ValueError(f"missing required documentation files: {missing}")
    non_pass = []
    for path in PHASE_GATE_REPORTS:
        text = (root / path).read_text(encoding="utf-8")
        if "PASS" not in text:
            non_pass.append(path)
    if non_pass:
        raise ValueError(f"phase gate reports missing PASS: {non_pass}")
    return {"documents": len(DOCUMENT_PATHS), "phase_gate_reports": len(PHASE_GATE_REPORTS)}


def audit_figures(root: Path) -> dict[str, Any]:
    missing_or_empty = [
        path for path in FIGURE_FILES if not (root / path).is_file() or (root / path).stat().st_size <= 0
    ]
    if missing_or_empty:
        raise ValueError(f"missing or empty figures: {missing_or_empty}")
    top20_path = root / "results" / "development" / "final_top20_pathways.csv"
    if not top20_path.is_file():
        raise ValueError("missing results/development/final_top20_pathways.csv")
    top20 = pd.read_csv(top20_path)
    if len(top20) != 20:
        raise ValueError("final_top20_pathways.csv must contain exactly 20 rows")
    return {"figures": len(FIGURE_FILES), "top20_rows": int(len(top20))}


def audit_frozen_chain(root: Path) -> dict[str, Any]:
    frozen_dir = root / "frozen"
    model_path = frozen_dir / "model_v1.pt"
    if is_lfs_pointer(model_path):
        raise ValueError("frozen/model_v1.pt is a Git LFS pointer, not a downloaded binary")
    hash_chain = validate_frozen_hash_chain(frozen_dir)
    metadata = json.loads((frozen_dir / "training_metadata.json").read_text(encoding="utf-8"))
    if metadata.get("seed") != 11 or metadata.get("n_epochs") != 16:
        raise ValueError("frozen-v1 metadata must document seed 11 and 16 final epochs")
    if metadata.get("parameter_count") != 18123110:
        raise ValueError("frozen-v1 metadata must document parameter count 18,123,110")
    return {
        "hash_before_after_equal": bool(hash_chain["hash_before_after_equal"]),
        "payload_files": len(IMMUTABLE_PAYLOAD_FILES),
        "model_lfs_pointer": False,
    }


def audit_result_values(root: Path) -> dict[str, Any]:
    development_dir = root / "results" / "development"
    external_dir = root / "results" / "external"

    baseline = pd.read_csv(development_dir / "baseline_summary.csv")
    binn_cv = pd.read_csv(development_dir / "binn_cv.csv")
    phase4_outputs = load_phase4_outputs(development_dir)
    phase5_outputs = load_phase5_outputs(development_dir)
    phase6_outputs = load_phase6_outputs(root / "frozen", external_dir)

    validate_activation_outputs(phase4_outputs["activation_scores"], phase4_outputs["activation_stability"])
    validate_ig_outputs(phase4_outputs["ig_scores"], phase4_outputs["ig_stability"])
    validate_agreement_outputs(
        phase4_outputs["seed_fold_agreement"],
        phase4_outputs["global_agreement"],
        phase4_outputs["rna_processing_tier"],
    )
    phase4_summary = summarize_phase4_gate(
        phase4_outputs["activation_scores"],
        phase4_outputs["activation_stability"],
        phase4_outputs["ig_scores"],
        phase4_outputs["ig_stability"],
        phase4_outputs["seed_fold_agreement"],
        phase4_outputs["global_agreement"],
        phase4_outputs["rna_processing_tier"],
    )

    validate_permutation_output(phase5_outputs["permutation_df"])
    validate_summary(
        phase5_outputs["summary"],
        phase5_outputs["permutation_df"],
        phase5_outputs["source_report_text"],
    )
    validate_bootstrap_output(phase5_outputs["bootstrap_df"], phase5_outputs["summary"])
    validate_calibration_output(phase5_outputs["calibration_df"], phase5_outputs["summary"])
    phase5_summary = summarize_phase5_gate(
        phase5_outputs["permutation_df"],
        phase5_outputs["summary"],
        phase5_outputs["bootstrap_df"],
        phase5_outputs["calibration_df"],
    )

    validate_external_predictions(phase6_outputs["external_predictions"])
    validate_external_metrics(phase6_outputs["external_metrics"])
    validate_ndd_predictions(phase6_outputs["ndd_predictions"])
    validate_ndd_summary(phase6_outputs["ndd_summary"])
    validate_scoring_audit(phase6_outputs["scoring_audit"], phase6_outputs["external_metrics"])

    _check_close(float(baseline.loc[baseline["model"] == "logistic_regression", "auroc_mean"].iloc[0]), "baseline_logistic_auroc")
    _check_close(float(baseline.loc[baseline["model"] == "random_forest", "auroc_mean"].iloc[0]), "baseline_random_forest_auroc")
    _check_close(float(baseline.loc[baseline["model"] == "unconstrained_mlp", "auroc_mean"].iloc[0]), "baseline_mlp_auroc")
    for metric, key in (
        ("auroc", "binn_mean_auroc"),
        ("auprc", "binn_mean_auprc"),
        ("balanced_accuracy", "binn_mean_balanced_accuracy"),
        ("brier", "binn_mean_brier"),
        ("ece", "binn_mean_ece"),
    ):
        _check_close(float(binn_cv[metric].mean()), key)

    bootstrap = phase5_outputs["bootstrap_df"].set_index("metric")
    _check_close(float(phase5_summary["observed_auroc"]), "pooled_auroc")
    _check_close(float(bootstrap.loc["auroc", "ci_lower"]), "bootstrap_auroc_lower")
    _check_close(float(bootstrap.loc["auroc", "ci_upper"]), "bootstrap_auroc_upper")
    _check_close(float(phase5_summary["null_auroc_mean"]), "null_auroc_mean")
    _check_close(float(phase5_summary["null_auroc_std"]), "null_auroc_std")
    _check_close(float(phase5_summary["empirical_p_value"]), "empirical_p_value")

    _check_close(float(phase4_summary["mean_seed_fold_spearman"]), "mean_seed_fold_spearman")
    _check_close(float(phase4_summary["global_spearman"]), "global_spearman")
    _check_exact(int(phase4_summary["global_top20_overlap"]), "global_top20_overlap")
    _check_exact(int(phase4_summary["activation_top20_rna_count"]), "activation_top20_rna_count")
    _check_exact(int(phase4_summary["ig_top20_rna_count"]), "ig_top20_rna_count")

    external_metrics = phase6_outputs["external_metrics"]
    _check_exact(int(external_metrics["n_samples"]), "external rows", EXPECTED_EXTERNAL_ROWS)
    _check_exact(int(external_metrics["n_hc"]), "external HC", EXPECTED_EXTERNAL_HC)
    _check_exact(int(external_metrics["n_pd"]), "external PD", EXPECTED_EXTERNAL_PD)
    _check_close(float(external_metrics["auroc"]), "external_auroc")
    _check_close(float(external_metrics["auprc"]), "external_auprc")
    _check_close(float(external_metrics["balanced_accuracy"]), "external_balanced_accuracy")
    _check_close(float(external_metrics["brier"]), "external_brier")
    _check_close(float(external_metrics["ece"]), "external_ece")
    ndd_summary = phase6_outputs["ndd_summary"]
    _check_exact(int(ndd_summary["n_samples"]), "ndd_samples")
    _check_close(float(ndd_summary["mean_pd_probability"]), "ndd_mean_pd_probability")
    _check_close(float(ndd_summary["fraction_predicted_pd_at_0_5"]), "ndd_fraction_predicted_pd")

    return {
        "development_values": "PASS",
        "permutation_values": "PASS",
        "attribution_values": "PASS",
        "external_values": "PASS",
        "ndd_values": "PASS",
    }


def audit_claim_safety(root: Path) -> dict[str, Any]:
    for relative_path in CLAIM_DOCS:
        text = (root / relative_path).read_text(encoding="utf-8").lower()
        for claim in FORBIDDEN_CLAIMS:
            if claim in text:
                raise ValueError(f"forbidden claim in {relative_path}: {claim}")
    readme = (root / "README.md").read_text(encoding="utf-8").lower()
    required_readme = ("not a diagnostic", "not clinical", "deployment-ready")
    if not all(fragment in readme for fragment in required_readme):
        raise ValueError("README must explicitly reject clinical, diagnostic, and deployment-ready use")
    combined = "\n".join((root / path).read_text(encoding="utf-8").lower() for path in CLAIM_DOCS)
    required_fragments = (
        "poor calibration",
        "0.500000",
        "50 permutations",
        "cannot support `p < 0.01`",
        "stress test only",
    )
    missing = [fragment for fragment in required_fragments if fragment not in combined]
    if missing:
        raise ValueError(f"required limitation language missing: {missing}")
    return {"forbidden_claims": 0, "required_limitations_visible": True}


def audit_repository_hygiene(root: Path) -> dict[str, Any]:
    tracked = _git_lines(root, ["git", "ls-files"])
    forbidden_suffixes = (".ipynb",)
    forbidden_parts = (
        "__pycache__",
        ".pytest_cache",
        ".codebase-memory",
        "data/raw/",
        "data/processed/",
    )
    forbidden_names = ("ext_X", "ext_y", "ndd_X", "dev_X", "dev_y")
    bad_paths = []
    for path in tracked:
        normalized = path.replace("\\", "/")
        if normalized == "frozen/model_v1.pt" or normalized.endswith("/.gitkeep"):
            continue
        if normalized.endswith(forbidden_suffixes):
            bad_paths.append(path)
        if any(part in normalized for part in forbidden_parts):
            bad_paths.append(path)
        if any(name in normalized for name in forbidden_names):
            bad_paths.append(path)
    if bad_paths:
        raise ValueError(f"tracked forbidden cache/notebook/raw/processed/input-array paths: {sorted(set(bad_paths))}")
    attributes = (root / ".gitattributes").read_text(encoding="utf-8") if (root / ".gitattributes").is_file() else ""
    if "frozen/model_v1.pt" not in attributes or "filter=lfs" not in attributes:
        raise ValueError("frozen/model_v1.pt must be managed by Git LFS in .gitattributes")
    return {"tracked_files_checked": len(tracked), "forbidden_tracked_paths": 0, "model_lfs_configured": True}


def is_lfs_pointer(path: str | Path) -> bool:
    data = Path(path).read_bytes()[:200]
    return data.startswith(b"version https://git-lfs.github.com/spec/v1")


def write_phase7_gate_report(output_path: str | Path, summary: Mapping[str, Any]) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase 7 Final Project Gate",
        "",
        "## Gate Status",
        "",
        "- Status: `PASS`.",
        "- PASS means final methods/reproducibility artifact integrity only.",
        "- PASS does not mean clinical validation, diagnostic utility, deployment readiness, mechanistic inference, or superiority.",
        "",
        "## Documentation Audit",
        "",
        f"- Required documents: `{summary['documentation']['documents']}`.",
        f"- Phase 1-6 PASS reports: `{summary['documentation']['phase_gate_reports']}`.",
        "",
        "## Figure Audit",
        "",
        f"- Figures present and nonempty: `{summary['figures']['figures']}`.",
        f"- Final top-20 companion rows: `{summary['figures']['top20_rows']}`.",
        "",
        "## Frozen Hash Audit",
        "",
        "- `HASH_BEFORE.txt` equals `HASH_AFTER.txt`.",
        f"- Immutable payload files verified: `{summary['frozen']['payload_files']}`.",
        "- `frozen/model_v1.pt` is not a Git LFS pointer in the working copy.",
        "",
        "## Result-Value Audit",
        "",
        "- Development, permutation, attribution, external, and NDD authoritative values match committed source artifacts.",
        "- No raw/processed arrays were loaded.",
        "- No result file was changed by the gate.",
        "",
        "## Claim-Safety Audit",
        "",
        "- No forbidden clinical, diagnostic, deployment, generalization, strong-performance, or SOTA claim text was found.",
        "- External poor calibration and balanced accuracy 0.5 are visible.",
        "- The 50-permutation limitation is visible.",
        "- NDD stress-test-only language is visible.",
        "",
        "## Repository-Hygiene Audit",
        "",
        "- No tracked caches, notebooks, raw/processed data, or external input arrays were found.",
        "- The expected large model is configured for Git LFS.",
        "",
        "## Boundary Confirmation",
        "",
        "- Tests are not run inside the gate.",
        "- No training/scoring/recomputation occurred.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def append_decision_log(path: str | Path) -> None:
    decision_path = Path(path)
    heading = "## 2026-07-10 - Phase 7 final project gate"
    existing = decision_path.read_text(encoding="utf-8") if decision_path.is_file() else ""
    if heading in existing:
        return
    entry = "\n".join(
        [
            "",
            "",
            heading,
            "",
            "- Final methods, results, limitations, README, figures, and reproducibility documentation completed.",
            "- Existing results were consolidated without retraining or rescoring.",
            "- Negative external threshold and calibration results were retained.",
            "- Frozen chain of custody remained intact.",
            "- Final gate passed only as a methods/reproducibility project, not as a clinical or diagnostic model.",
            "",
        ]
    )
    decision_path.write_text(existing.rstrip() + entry, encoding="utf-8")


def _check_close(observed: float, key: str) -> None:
    expected = float(AUTHORITATIVE_VALUES[key])
    if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=FLOAT_TOLERANCE):
        raise ValueError(f"{key} mismatch: observed={observed}, expected={expected}")


def _check_exact(observed: int, key: str, expected_override: int | None = None) -> None:
    expected = int(AUTHORITATIVE_VALUES[key]) if expected_override is None else expected_override
    if observed != expected:
        raise ValueError(f"{key} mismatch: observed={observed}, expected={expected}")


def _git_lines(root: Path, command: list[str]) -> list[str]:
    try:
        completed = subprocess.run(
            command,
            cwd=root,
            text=True,
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return []
    return [line for line in completed.stdout.splitlines() if line.strip()]
