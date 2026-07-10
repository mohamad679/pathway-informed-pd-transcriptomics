from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from eval.phase4_gate import (  # noqa: E402
    load_phase4_outputs,
    summarize_phase4_gate,
    validate_activation_outputs,
    validate_agreement_outputs,
    validate_ig_outputs,
    write_phase4_gate_report,
)


RESULTS_DIR = ROOT / "results" / "development"
REPORT_PATH = ROOT / "docs" / "phase4_gate.md"
DECISION_LOG_PATH = ROOT / "docs" / "decision_log.md"
DECISION_HEADING = "## 2026-07-10 - Phase 4 final gate audit"


def append_decision_log(summary: dict[str, float | int]) -> None:
    entry = "\n".join(
        [
            "",
            "",
            DECISION_HEADING,
            "",
            "- Audited only existing Phase 4 development activation, Integrated Gradients, and agreement artifacts.",
            "- Confirmed exact seed/fold/pathway coverage, rank coverage, finite scores, stability fields, and agreement bounds.",
            f"- Mean seed/fold Spearman: `{summary['mean_seed_fold_spearman']:.6f}`; global Spearman: `{summary['global_spearman']:.6f}`; global top-20 overlap: `{summary['global_top20_overlap']}`.",
            "- No training/retraining, activation attribution, Integrated Gradients attribution, final validation, attention, or permutation testing was run inside this gate.",
            "- No external cohort or held-out NDD data was used.",
            "- This is development-only artifact integrity auditing, not a biological claim and not final performance.",
            "",
        ]
    )
    existing = DECISION_LOG_PATH.read_text(encoding="utf-8") if DECISION_LOG_PATH.exists() else ""
    if DECISION_HEADING not in existing:
        DECISION_LOG_PATH.write_text(existing.rstrip() + entry, encoding="utf-8")


def main() -> int:
    try:
        outputs = load_phase4_outputs(RESULTS_DIR)
        validate_activation_outputs(outputs["activation_scores"], outputs["activation_stability"])
        validate_ig_outputs(outputs["ig_scores"], outputs["ig_stability"])
        validate_agreement_outputs(
            outputs["seed_fold_agreement"],
            outputs["global_agreement"],
            outputs["rna_processing_tier"],
        )
        summary = summarize_phase4_gate(
            outputs["activation_scores"],
            outputs["activation_stability"],
            outputs["ig_scores"],
            outputs["ig_stability"],
            outputs["seed_fold_agreement"],
            outputs["global_agreement"],
            outputs["rna_processing_tier"],
        )
        write_phase4_gate_report(REPORT_PATH, summary)
        append_decision_log(summary)
    except (FileNotFoundError, ValueError) as error:
        print("FAIL")
        print(f"Phase 4 gate failure: {error}")
        return 1

    print("PASS")
    print(f"activation score rows: {summary['activation_score_rows']}")
    print(f"IG score rows: {summary['ig_score_rows']}")
    print(f"pathway count: {summary['pathway_count']}")
    print(f"mean seed/fold Spearman: {summary['mean_seed_fold_spearman']:.6f}")
    print(f"global Spearman: {summary['global_spearman']:.6f}")
    print(f"global top20 overlap: {summary['global_top20_overlap']}")
    print(f"RNA-processing tier rows: {summary['rna_processing_tier_rows']}")
    print("confirmation no external/NDD data used: yes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
