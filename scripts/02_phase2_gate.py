from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from eval.phase2_gate import (  # noqa: E402
    load_phase2_outputs,
    validate_oof_prediction_integrity,
    validate_required_models,
    validate_sanity_gate,
    write_phase2_gate_report,
)


RESULTS_DIR = ROOT / "results" / "development"
REPORT_PATH = ROOT / "docs" / "phase2_gate.md"
DECISION_LOG_PATH = ROOT / "docs" / "decision_log.md"


def append_decision_log(best_model: str) -> None:
    entry = "\n".join(
        [
            "",
            "",
            "## 2026-07-10 — Phase 2 final gate audit",
            "",
            "- Audited only the existing Phase 2 development baseline output CSVs.",
            "- Confirmed the exact required model set, summary metric mean/CI columns, three-seed OOF coverage, finite bounded probabilities, and the AUROC sanity gate.",
            f"- Recorded `{best_model}` as the best development-only baseline by mean OOF AUROC.",
            "- No external cohort or held-out NDD data was loaded or used.",
            "- No modeling, BINN, pathway masks, or MSigDB logic was implemented or used.",
            "- This gate result is not an external-validation result or final performance claim.",
            "",
        ]
    )
    existing = DECISION_LOG_PATH.read_text(encoding="utf-8") if DECISION_LOG_PATH.exists() else ""
    if "## 2026-07-10 — Phase 2 final gate audit" not in existing:
        DECISION_LOG_PATH.write_text(existing.rstrip() + entry, encoding="utf-8")


def main() -> int:
    try:
        outputs = load_phase2_outputs(RESULTS_DIR)
        summary_df = outputs["summary"]
        validate_required_models(summary_df)
        validate_oof_prediction_integrity(outputs["oof"])
        best_model = validate_sanity_gate(summary_df)
        write_phase2_gate_report(REPORT_PATH, summary_df=summary_df, best_model=best_model)
        append_decision_log(best_model)
    except (FileNotFoundError, ValueError) as error:
        print("FAIL")
        print(f"Phase 2 gate failure: {error}")
        return 1

    print("PASS")
    print("model AUROC means")
    for row in summary_df.sort_values("model").itertuples(index=False):
        print(f"{row.model}: {float(row.auroc_mean):.6f}")
    print(f"best model by AUROC: {best_model}")
    print("confirmation no external/NDD data used: yes")
    print("confirmation Phase 2 gate passed: yes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
