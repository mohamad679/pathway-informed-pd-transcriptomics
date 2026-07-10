"""Summarize development-only agreement between activation and IG attributions."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from interpret.attribution_agreement import (
    compute_global_stability_agreement,
    compute_seed_fold_agreement,
    load_attribution_outputs,
    summarize_rna_processing_tier,
    validate_attribution_scores,
    validate_attribution_stability,
    write_agreement_report,
)


RESULTS_DIR = ROOT / "results" / "development"
SEED_FOLD_AGREEMENT_PATH = RESULTS_DIR / "pathway_attribution_seed_fold_agreement.csv"
GLOBAL_AGREEMENT_PATH = RESULTS_DIR / "pathway_attribution_global_agreement.csv"
RNA_PROCESSING_TIER_PATH = RESULTS_DIR / "pathway_rna_processing_tier.csv"
REPORT_PATH = ROOT / "docs" / "phase4_attribution_agreement.md"
DECISION_LOG_PATH = ROOT / "docs" / "decision_log.md"


def append_if_missing(path: Path, block: str) -> None:
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    if block.strip() in existing:
        return
    separator = "" if not existing or existing.endswith("\n\n") else "\n"
    path.write_text(existing + separator + block.rstrip() + "\n", encoding="utf-8")


def main() -> int:
    outputs = load_attribution_outputs(RESULTS_DIR)
    validate_attribution_scores(outputs.activation_scores, "activation")
    validate_attribution_stability(outputs.activation_stability, "activation")
    validate_attribution_scores(outputs.ig_scores, "integrated_gradients")
    validate_attribution_stability(outputs.ig_stability, "integrated_gradients")

    seed_fold_agreement = compute_seed_fold_agreement(outputs.activation_scores, outputs.ig_scores, k=20)
    global_agreement = compute_global_stability_agreement(
        outputs.activation_stability, outputs.ig_stability, k=20
    )
    rna_processing_tier = summarize_rna_processing_tier(
        outputs.activation_stability, outputs.ig_stability
    )

    seed_fold_agreement.to_csv(SEED_FOLD_AGREEMENT_PATH, index=False)
    global_agreement.to_csv(GLOBAL_AGREEMENT_PATH, index=False)
    rna_processing_tier.to_csv(RNA_PROCESSING_TIER_PATH, index=False)
    write_agreement_report(seed_fold_agreement, global_agreement, rna_processing_tier, REPORT_PATH)
    append_if_missing(
        DECISION_LOG_PATH,
        """## 2026-07-10 - Phase 4 activation-vs-Integrated-Gradients agreement

- Compared existing development-only activation and Integrated Gradients attribution CSVs without retraining models.
- Wrote seed/fold agreement, global stability agreement, and RNA-processing-flagged tier summaries from development results only.
- No external cohort or held-out NDD data is loaded or used; this is not final validation and not a biological claim.
- Attention, permutation testing, final validation, and later-phase logic are excluded.
""",
    )

    global_row = global_agreement.iloc[0]
    print(
        f"mean seed/fold Spearman: "
        f"{seed_fold_agreement['spearman_rank_correlation'].mean():.6f}"
    )
    print(
        f"min/max seed/fold Spearman: "
        f"{seed_fold_agreement['spearman_rank_correlation'].min():.6f}/"
        f"{seed_fold_agreement['spearman_rank_correlation'].max():.6f}"
    )
    print(f"mean top20 overlap: {seed_fold_agreement['top20_overlap'].mean():.6f}")
    print(f"global Spearman: {global_row.spearman_rank_correlation:.6f}")
    print(f"global top20 overlap: {int(global_row.top20_overlap)}")
    print(f"activation top20 RNA-processing count: {int(global_row.activation_top20_rna_count)}")
    print(f"IG top20 RNA-processing count: {int(global_row.ig_top20_rna_count)}")
    print("confirmation no external/NDD data used: yes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
