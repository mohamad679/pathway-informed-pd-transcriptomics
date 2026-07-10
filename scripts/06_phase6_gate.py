from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from eval.phase6_gate import (  # noqa: E402
    load_phase6_outputs,
    summarize_phase6_gate,
    validate_external_metrics,
    validate_external_predictions,
    validate_frozen_hash_chain,
    validate_ndd_predictions,
    validate_ndd_summary,
    validate_scoring_audit,
    write_phase6_gate_report,
)


FROZEN_DIR = ROOT / "frozen"
EXTERNAL_RESULTS_DIR = ROOT / "results" / "external"
REPORT_PATH = ROOT / "docs" / "phase6_gate.md"
DECISION_LOG_PATH = ROOT / "docs" / "decision_log.md"
DECISION_HEADING = "## 2026-07-10 - Phase 6 final frozen external-validation gate"


def append_decision_log(summary: Mapping[str, Any]) -> None:
    entry = "\n".join(
        [
            "",
            "",
            DECISION_HEADING,
            "",
            "- Audited only the existing frozen manifests, immutable payload hashes, external/NDD result artifacts, and one-time scoring audit.",
            f"- Confirmed `HASH_BEFORE.txt` equals `HASH_AFTER.txt` exactly and both manifests verify the frozen `{summary['frozen_tag']}` payload at `{summary['frozen_commit']}`.",
            f"- External artifact: `{summary['external_rows']}` rows (`{summary['external_hc']}` HC, `{summary['external_pd']}` PD); AUROC `{summary['auroc']:.6f}`; AUPRC `{summary['auprc']:.6f}`; balanced accuracy `{summary['balanced_accuracy']:.6f}`; Brier `{summary['brier']:.6f}`; ECE `{summary['ece']:.6f}`.",
            f"- NDD artifact: `{summary['ndd_rows']}` rows; fraction predicted PD at 0.5 `{summary['ndd_fraction_predicted_pd']:.6f}`; this remains a specificity/stress test only.",
            "- No retraining, preprocessing refit, inference/rescoring, threshold tuning, or external-metric model selection occurred inside the gate.",
            "- PASS means chain-of-custody and artifact integrity only; it does not mean clinical validation, deployment readiness, or acceptable calibration and supports no clinical, diagnostic, deployment, or biological claim.",
            "",
        ]
    )
    existing = DECISION_LOG_PATH.read_text(encoding="utf-8") if DECISION_LOG_PATH.exists() else ""
    if DECISION_HEADING not in existing:
        DECISION_LOG_PATH.write_text(existing.rstrip() + entry, encoding="utf-8")


def main() -> int:
    try:
        outputs = load_phase6_outputs(FROZEN_DIR, EXTERNAL_RESULTS_DIR)
        hash_chain = validate_frozen_hash_chain(FROZEN_DIR)
        external_predictions = outputs["external_predictions"]
        external_metrics = outputs["external_metrics"]
        ndd_predictions = outputs["ndd_predictions"]
        ndd_summary = outputs["ndd_summary"]
        scoring_audit = outputs["scoring_audit"]

        validate_external_predictions(external_predictions)
        validate_external_metrics(external_metrics, external_predictions)
        validate_ndd_predictions(ndd_predictions)
        validate_ndd_summary(ndd_summary, ndd_predictions)
        validate_scoring_audit(scoring_audit, external_metrics)
        gate_summary = summarize_phase6_gate(
            hash_chain,
            external_predictions,
            external_metrics,
            ndd_predictions,
            ndd_summary,
            scoring_audit,
        )
        write_phase6_gate_report(REPORT_PATH, gate_summary)
        append_decision_log(gate_summary)
    except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        print("FAIL")
        print(f"Phase 6 gate failure: {error}")
        return 1

    print("PASS")
    print("HASH_BEFORE vs HASH_AFTER: identical")
    print(f"frozen commit/tag: {gate_summary['frozen_commit']} / {gate_summary['frozen_tag']}")
    print(
        "external rows and class counts: "
        f"{gate_summary['external_rows']} / HC={gate_summary['external_hc']} / "
        f"PD={gate_summary['external_pd']}"
    )
    print(f"external AUROC: {gate_summary['auroc']:.6f}")
    print(f"external AUPRC: {gate_summary['auprc']:.6f}")
    print(f"external balanced accuracy: {gate_summary['balanced_accuracy']:.6f}")
    print(f"external Brier: {gate_summary['brier']:.6f}")
    print(f"external ECE: {gate_summary['ece']:.6f}")
    print(f"NDD rows: {gate_summary['ndd_rows']}")
    print(f"NDD fraction predicted PD: {gate_summary['ndd_fraction_predicted_pd']:.6f}")
    print("confirmation no retraining/refit/rescoring inside gate: yes")
    print("confirmation not clinical validation: yes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
