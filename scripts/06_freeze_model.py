"""Freeze the pre-external Phase 6 BINN bundle from development inputs only."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from models.final_training import fit_full_development_binn, select_final_epoch_count
from models.frozen_bundle import compute_bundle_hashes, save_frozen_bundle, verify_hash_manifest, write_hash_manifest


PROCESSED_DIR = ROOT / "data" / "processed"
DEVELOPMENT_RESULTS_DIR = ROOT / "results" / "development"
FROZEN_DIR = ROOT / "frozen"
DOCS_DIR = ROOT / "docs"

ALLOWED_INPUTS = (
    PROCESSED_DIR / "dev_X.npy",
    PROCESSED_DIR / "dev_y.npy",
    PROCESSED_DIR / "gene_space.txt",
    PROCESSED_DIR / "pathway_names.txt",
    PROCESSED_DIR / "pathway_mask.npz",
    DEVELOPMENT_RESULTS_DIR / "binn_cv.csv",
)


def _read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def _append_if_missing(path: Path, block: str) -> None:
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    if block.strip() not in existing:
        path.write_text(existing.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")


def _write_phase6_report(
    *,
    path: Path,
    development_sample_count: int,
    gene_count: int,
    pathway_count: int,
    selected_epoch_count: int,
    final_seed: int,
    parameter_count: int,
    max_masked_weight: float,
) -> None:
    path.write_text(
        "# Phase 6 pre-external frozen BINN bundle\n\n"
        "This report documents only pre-external model selection, full-development retraining, "
        "frozen bundle creation, and pre-score hashing.\n\n"
        "- Final epoch count was selected from development CV `best_epoch` values only.\n"
        "- The deterministic rule is the integer median across the 15 seed/fold `best_epoch` values.\n"
        "- The final `StandardScaler` was fit on full development only.\n"
        "- External and NDD files were not loaded.\n"
        "- The external cohort has not yet been scored.\n"
        "- Frozen files must not change after commit/tag; later scoring must verify the pre-score hash manifest first.\n"
        "- This is not external-validation performance and makes no final/external performance claim.\n\n"
        "## Frozen training configuration\n\n"
        f"- Development sample count: `{development_sample_count}`\n"
        f"- Gene count: `{gene_count}`\n"
        f"- Pathway count: `{pathway_count}`\n"
        f"- Selected epoch count: `{selected_epoch_count}`\n"
        f"- Final seed: `{final_seed}`\n"
        f"- Parameter count: `{parameter_count}`\n"
        f"- Max masked weight: `{max_masked_weight:.1f}`\n"
        "- Architecture: `hidden_dim=64`, `dropout=0.25`\n"
        "- Optimizer: Adam with `learning_rate=1e-3`, `weight_decay=1e-4`, `batch_size=64`\n",
        encoding="utf-8",
    )


def main() -> int:
    for path in ALLOWED_INPUTS:
        if not path.exists():
            raise FileNotFoundError(f"Missing allowed development input: {path}")

    X = np.load(PROCESSED_DIR / "dev_X.npy", allow_pickle=False)
    y = np.load(PROCESSED_DIR / "dev_y.npy", allow_pickle=False)
    gene_space = _read_lines(PROCESSED_DIR / "gene_space.txt")
    pathway_names = _read_lines(PROCESSED_DIR / "pathway_names.txt")
    pathway_mask = sparse.load_npz(PROCESSED_DIR / "pathway_mask.npz").toarray().astype(np.float32, copy=False)
    binn_cv_df = pd.read_csv(DEVELOPMENT_RESULTS_DIR / "binn_cv.csv")

    selected_epoch_count = select_final_epoch_count(binn_cv_df)
    final_seed = 11
    model, scaler, training_metadata = fit_full_development_binn(
        X,
        y,
        pathway_mask,
        seed=final_seed,
        hidden_dim=64,
        dropout=0.25,
        n_epochs=selected_epoch_count,
        learning_rate=1e-3,
        weight_decay=1e-4,
        batch_size=64,
        device="cpu",
    )
    save_frozen_bundle(
        frozen_dir=FROZEN_DIR,
        model=model,
        scaler=scaler,
        gene_space=gene_space,
        pathway_names=pathway_names,
        pathway_mask=pathway_mask,
        training_metadata=training_metadata,
        hidden_dim=64,
        dropout=0.25,
        seed=final_seed,
        n_epochs=selected_epoch_count,
    )
    hash_before_path = FROZEN_DIR / "HASH_BEFORE.txt"
    write_hash_manifest(compute_bundle_hashes(FROZEN_DIR), hash_before_path)
    verify_hash_manifest(FROZEN_DIR, hash_before_path)

    parameter_count = int(training_metadata["parameter_count"])
    max_masked_weight = float(training_metadata["max_abs_masked_weight_after_training"])
    _write_phase6_report(
        path=DOCS_DIR / "phase6_freeze.md",
        development_sample_count=int(X.shape[0]),
        gene_count=int(X.shape[1]),
        pathway_count=int(pathway_mask.shape[0]),
        selected_epoch_count=selected_epoch_count,
        final_seed=final_seed,
        parameter_count=parameter_count,
        max_masked_weight=max_masked_weight,
    )
    _append_if_missing(
        DOCS_DIR / "decision_log.md",
        """## 2026-07-10 - Phase 6 pre-external frozen BINN bundle

- Selected the full-development epoch count using only the 15 Phase 3 development CV `best_epoch` values.
- Retrained the fixed-architecture pathway-constrained BINN once on full development data with seed 11.
- Fit the frozen `StandardScaler` on full development only and wrote the immutable frozen payload files plus `HASH_BEFORE.txt`.
- External and NDD files were not loaded, and the external cohort has not yet been scored.
- This is pre-score bundle freezing only, not external-validation performance.""",
    )

    print(f"development sample count: {int(X.shape[0])}")
    print(f"gene count: {int(X.shape[1])}")
    print(f"pathway count: {int(pathway_mask.shape[0])}")
    print(f"selected epoch count: {selected_epoch_count}")
    print(f"final seed: {final_seed}")
    print(f"parameter count: {parameter_count}")
    print(f"max masked weight: {max_masked_weight:.1f}")
    print("HASH_BEFORE verification PASS")
    print("confirmation external/NDD not loaded: yes")
    print("confirmation external cohort not scored: yes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
