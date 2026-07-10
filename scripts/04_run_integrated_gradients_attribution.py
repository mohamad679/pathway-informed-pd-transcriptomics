"""Run development-only, pathway-level BINN Integrated Gradients attribution."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from interpret.integrated_gradients_attribution import run_integrated_gradients_attribution_cv
from interpret.pathway_attribution import load_gene_names, load_pathway_names, load_rna_processing_keywords
from models.binn_training import DEFAULT_SEEDS, load_binn_inputs


PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results" / "development"
SCORES_PATH = RESULTS_DIR / "pathway_ig_scores.csv"
STABILITY_PATH = RESULTS_DIR / "pathway_ig_stability.csv"
REPORT_PATH = ROOT / "docs" / "phase4_integrated_gradients_attribution.md"
DECISION_LOG_PATH = ROOT / "docs" / "decision_log.md"
CONFIG_PATH = ROOT / "config" / "pathways.yaml"
N_STEPS = 16


def append_if_missing(path: Path, block: str) -> None:
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    if block.strip() in existing:
        return
    separator = "" if not existing or existing.endswith("\n\n") else "\n"
    path.write_text(existing + separator + block.rstrip() + "\n", encoding="utf-8")


def main() -> int:
    X, y, folds, pathway_mask = load_binn_inputs(
        PROCESSED_DIR / "dev_X.npy",
        PROCESSED_DIR / "dev_y.npy",
        PROCESSED_DIR / "dev_folds.json",
        PROCESSED_DIR / "pathway_mask.npz",
    )
    pathway_names = load_pathway_names(PROCESSED_DIR / "pathway_names.txt")
    gene_names = load_gene_names(PROCESSED_DIR / "gene_space.txt")
    if len(pathway_names) != pathway_mask.shape[0] or len(gene_names) != X.shape[1]:
        raise ValueError("Development gene/pathway name files do not match the fixed development arrays.")

    scores_df, stability_df, audit = run_integrated_gradients_attribution_cv(
        X, y, folds, pathway_mask, pathway_names, seeds=DEFAULT_SEEDS,
        rna_processing_keywords=load_rna_processing_keywords(CONFIG_PATH), n_steps=N_STEPS,
    )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    scores_df.to_csv(SCORES_PATH, index=False)
    stability_df.to_csv(STABILITY_PATH, index=False)

    top20 = stability_df.head(20)
    top20_rows = "\n".join(
        f"| {row.pathway_name} | {row.mean_rank:.3f} | {str(bool(row.is_rna_processing)).lower()} |"
        for row in top20.itertuples(index=False)
    )
    REPORT_PATH.write_text(
        f"""# Phase 4 pathway-level Integrated Gradients attribution

Generated at: {datetime.now(timezone.utc).isoformat()}

This is development-only interpretation retraining. Phase 3 fold models were not checkpointed, so the same predefined development folds and seeds were retrained solely to calculate out-of-fold interpretation artifacts. This is not model selection and is not final validation. It is not a biological claim.

- Inputs: `dev_X.npy`, `dev_y.npy`, `dev_folds.json`, `pathway_mask.npz`, `pathway_names.txt`, `gene_space.txt`, and `config/pathways.yaml` only.
- Each validation partition was transformed with its fold's train-only scaler.
- This is pathway-level Integrated Gradients on pathway activations, not raw-gene Integrated Gradients.
- Integrated Gradients uses a zero baseline, targets the downstream pathway-to-hidden-to-logit head, and runs with the model in evaluation mode so dropout is disabled.
- No external cohort or held-out NDD data was loaded or used. Attention, permutation testing, and later-phase logic are not included.
- Off-mask weights were enforced at every optimizer step and were zero after each interpretation retraining.

## Audit

| Metric | Value |
| --- | ---: |
| Pathways | {audit['n_pathways']} |
| Seeds | {audit['n_seeds']} |
| Development folds | {audit['n_folds']} |
| IG steps | {audit['n_steps']} |
| RNA-processing pathways | {audit['n_rna_processing_pathways']} |
| Maximum masked weight after interpretation retraining | {audit['max_masked_weight_after_training']:.1f} |
| No external/NDD data used | yes |

## Top 20 pathways by mean IG rank

| Pathway | Mean rank | RNA-processing flag |
| --- | ---: | --- |
{top20_rows}
""",
        encoding="utf-8",
    )
    append_if_missing(
        DECISION_LOG_PATH,
        """## 2026-07-10 — Phase 4 development-only pathway-level Integrated Gradients

- Retrains the existing development-only BINN folds solely because Phase 3 fold checkpoints were not retained.
- Uses train-only scaling, validation-partition pathway activations, and zero-baseline Integrated Gradients through the downstream pathway head; it is not model selection, final validation, or external validation.
- No external cohort or held-out NDD data is loaded or used; no biological claims are made.
- Attention, permutation testing, activation-attribution outputs, and later-phase logic are excluded.
""",
    )

    print("Development-only interpretation retraining: pathway-level Integrated Gradients")
    print(f"Number of pathways: {audit['n_pathways']}")
    print(f"Seeds/folds: {audit['n_seeds']}/{audit['n_folds']}")
    print(f"IG steps: {audit['n_steps']}")
    print("Top 20 IG pathways by mean rank:")
    print(top20[["pathway_name", "mean_rank"]].to_string(index=False))
    print("RNA-processing pathways in top 20:")
    print(top20.loc[top20["is_rna_processing"], "pathway_name"].to_string(index=False))
    print(f"Max masked weight after interpretation retraining: {audit['max_masked_weight_after_training']:.1f}")
    print("Confirmation no external/NDD data used: yes")
    print("This is not final validation and is not a biological claim.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
