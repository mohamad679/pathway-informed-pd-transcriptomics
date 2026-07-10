from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from eval.phase5_gate import (  # noqa: E402
    load_phase5_outputs,
    summarize_phase5_gate,
    validate_bootstrap_output,
    validate_calibration_output,
    validate_permutation_output,
    validate_summary,
    write_phase5_gate_report,
)


RESULTS_DIR = ROOT / "results" / "development"
REPORT_PATH = ROOT / "docs" / "phase5_gate.md"
DECISION_LOG_PATH = ROOT / "docs" / "decision_log.md"
DECISION_HEADING = "## 2026-07-10 - Phase 5 final statistical-validation gate"


def append_decision_log(summary: dict[str, float | int | bool]) -> None:
    entry = "\n".join(
        [
            "",
            "",
            DECISION_HEADING,
            "",
            "- Audited only existing Phase 5 development statistical-validation artifacts.",
            "- Confirmed exactly 50 unique permutation rows covering indices 1 through 50.",
            f"- Observed AUROC: `{summary['observed_auroc']:.6f}`; null AUROC mean/std: `{summary['null_auroc_mean']:.6f}` / `{summary['null_auroc_std']:.6f}`; empirical p-value: `{summary['empirical_p_value']:.6f}`.",
            "- Confirmed bootstrap CI integrity with 2,000 resamples per metric and 15-bin calibration sample accounting.",
            "- The production limitation remains: 50 permutations only, minimum attainable p-value is `1/51 = 0.019608`, and this result cannot support `p < 0.01`.",
            "- No training, retraining, permutation rerun, bootstrap rerun, model freezing, external cohort, or held-out NDD data was used inside this gate.",
            "- This is development-only statistical-artifact integrity auditing, not final validation and not a biological claim.",
            "",
        ]
    )
    existing = DECISION_LOG_PATH.read_text(encoding="utf-8") if DECISION_LOG_PATH.exists() else ""
    if DECISION_HEADING not in existing:
        DECISION_LOG_PATH.write_text(existing.rstrip() + entry, encoding="utf-8")


def main() -> int:
    try:
        outputs = load_phase5_outputs(RESULTS_DIR)
        permutation_df = outputs["permutation_df"]
        summary = outputs["summary"]
        bootstrap_df = outputs["bootstrap_df"]
        calibration_df = outputs["calibration_df"]

        validate_permutation_output(permutation_df)
        validate_summary(summary, permutation_df, outputs["source_report_text"])
        validate_bootstrap_output(bootstrap_df, summary)
        validate_calibration_output(calibration_df, summary)
        gate_summary = summarize_phase5_gate(
            permutation_df,
            summary,
            bootstrap_df,
            calibration_df,
        )
        write_phase5_gate_report(REPORT_PATH, gate_summary)
        append_decision_log(gate_summary)
    except (FileNotFoundError, KeyError, TypeError, ValueError) as error:
        print("FAIL")
        print(f"Phase 5 gate failure: {error}")
        return 1

    print("PASS")
    print(f"permutation rows: {gate_summary['permutation_rows']}")
    print("exact permutation coverage: yes")
    print(f"observed AUROC: {gate_summary['observed_auroc']:.6f}")
    print(
        "null mean/std: "
        f"{gate_summary['null_auroc_mean']:.6f} / {gate_summary['null_auroc_std']:.6f}"
    )
    print(f"empirical p-value: {gate_summary['empirical_p_value']:.6f}")
    print(f"bootstrap resamples: {gate_summary['bootstrap_resamples']}")
    print(f"Brier: {gate_summary['brier']:.6f}")
    print(f"ECE: {gate_summary['ece']:.6f}")
    print("confirmation no external/NDD data used: yes")
    print("confirmation no training/permutation/bootstrap rerun inside gate: yes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
