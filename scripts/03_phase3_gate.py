from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from eval.phase3_gate import (  # noqa: E402
    load_phase3_outputs,
    summarize_phase3_gate,
    validate_binn_cv,
    validate_binn_oof,
    validate_mask_integrity,
    validate_pathway_mask,
    write_phase3_gate_report,
)


PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results" / "development"
REPORT_PATH = ROOT / "docs" / "phase3_gate.md"
DECISION_LOG_PATH = ROOT / "docs" / "decision_log.md"
DECISION_HEADING = "## 2026-07-10 — Phase 3 final gate audit"


def append_decision_log() -> None:
    entry = "\n".join(
        [
            "",
            "",
            DECISION_HEADING,
            "",
            "- Audited only existing Phase 3 development BINN outputs and the saved pathway mask.",
            "- Confirmed fixed seed/fold coverage, finite bounded metrics and OOF probabilities, and the development AUROC sanity check.",
            "- Confirmed off-mask weights remained exactly zero and reported weight counts match the saved mask.",
            "- No external cohort or held-out NDD result was used.",
            "- No model was trained and BINN CV was not rerun inside this gate.",
            "- This is development-only and not final validation; the gate concerns stable training and mask integrity, not superiority.",
            "",
        ]
    )
    existing = DECISION_LOG_PATH.read_text(encoding="utf-8") if DECISION_LOG_PATH.exists() else ""
    if DECISION_HEADING not in existing:
        DECISION_LOG_PATH.write_text(existing.rstrip() + entry, encoding="utf-8")


def main() -> int:
    try:
        outputs = load_phase3_outputs(PROCESSED_DIR, RESULTS_DIR)
        mask = outputs["mask"]
        metrics_df = outputs["metrics"]
        oof_df = outputs["oof"]
        validate_pathway_mask(mask, outputs["pathway_names"], outputs["gene_space"])
        validate_binn_cv(metrics_df)
        validate_binn_oof(oof_df)
        validate_mask_integrity(metrics_df, mask)
        summary = summarize_phase3_gate(metrics_df, mask, oof_df)
        write_phase3_gate_report(REPORT_PATH, summary)
        append_decision_log()
    except (FileNotFoundError, ValueError) as error:
        print("FAIL")
        print(f"Phase 3 gate failure: {error}")
        return 1

    print("PASS")
    print(f"mask shape: {summary['mask_shape']}")
    print(f"mask nnz: {summary['mask_nnz']}")
    print(f"mean AUROC: {summary['mean_auroc']:.6f}")
    print(f"mean AUPRC: {summary['mean_auprc']:.6f}")
    print(f"mean balanced accuracy: {summary['mean_balanced_accuracy']:.6f}")
    print(f"mean Brier: {summary['mean_brier']:.6f}")
    print(f"mean ECE: {summary['mean_ece']:.6f}")
    print(f"max masked weight: {summary['max_masked_weight']:.1f}")
    print(f"OOF rows: {summary['oof_rows']}")
    print("confirmation no external/NDD data used: yes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
