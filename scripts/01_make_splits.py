from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data.splits import (
    load_sample_ids,
    make_stratified_folds,
    validate_fold_disjointness,
    validate_no_overlap_between_sets,
    write_folds_json,
    write_split_audit,
)


PROCESSED_DIR = ROOT / "data" / "processed"
DOCS_DIR = ROOT / "docs"

DEV_Y_PATH = PROCESSED_DIR / "dev_y.npy"
DEV_GROUPS_PATH = PROCESSED_DIR / "dev_groups.npy"
DEV_SAMPLE_IDS_PATH = PROCESSED_DIR / "dev_sample_ids.txt"
NDD_SAMPLE_IDS_PATH = PROCESSED_DIR / "ndd_sample_ids.txt"
EXT_SAMPLE_IDS_PATH = PROCESSED_DIR / "ext_sample_ids.txt"
DEV_FOLDS_PATH = PROCESSED_DIR / "dev_folds.json"
SPLIT_AUDIT_PATH = DOCS_DIR / "split_audit.md"
DECISION_LOG_PATH = DOCS_DIR / "decision_log.md"

SUBJECT_IDENTIFIER_NOTE = (
    "Donor IDs were not available in the processed metadata inputs, so GSM accession "
    "was used as the conservative subject identifier for split integrity."
)

DECISION_LOG_ENTRY = """## 2026-07-10 — Phase 1 development split creation

- Created stratified 5-fold cross-validation splits from development PD/HC samples only.
- Held-out NDD samples were checked for overlap but were not used for split creation.
- External cohort samples were checked for overlap but were not used for split creation.
- Donor IDs were not available in the processed metadata inputs, so GSM accession was used as the conservative subject identifier for split integrity.
- No modeling, baselines, training, pathway masks, MSigDB logic, or external-validation model selection artifacts were created.
"""


def append_if_missing(path: Path, block: str) -> None:
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    if block.strip() in existing:
        return
    separator = "" if not existing or existing.endswith("\n\n") else "\n"
    path.write_text(existing + separator + block.rstrip() + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create Phase 1 development-only CV splits.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite data/processed/dev_folds.json if it already exists.",
    )
    return parser.parse_args()


def _class_counts_by_group(group_values: np.ndarray) -> dict[str, int]:
    unique_values, counts = np.unique(group_values, return_counts=True)
    return {str(value): int(count) for value, count in zip(unique_values.tolist(), counts.tolist(), strict=True)}


def main() -> None:
    args = parse_args()
    if DEV_FOLDS_PATH.exists() and not args.force:
        raise SystemExit(
            f"{DEV_FOLDS_PATH.relative_to(ROOT).as_posix()} already exists. "
            "Pass --force to regenerate."
        )

    dev_y = np.load(DEV_Y_PATH, allow_pickle=False)
    dev_groups = np.load(DEV_GROUPS_PATH, allow_pickle=False)
    dev_sample_ids = load_sample_ids(DEV_SAMPLE_IDS_PATH)
    ndd_sample_ids = load_sample_ids(NDD_SAMPLE_IDS_PATH)
    ext_sample_ids = load_sample_ids(EXT_SAMPLE_IDS_PATH)

    if len(dev_y) != len(dev_sample_ids):
        raise ValueError("Development labels and sample IDs must have the same length")
    if len(dev_groups) != len(dev_sample_ids):
        raise ValueError("Development group labels and sample IDs must have the same length")

    overlap_summary = validate_no_overlap_between_sets(dev_sample_ids, ndd_sample_ids, ext_sample_ids)
    folds = make_stratified_folds(dev_y, dev_sample_ids, n_splits=5, random_state=20260710)
    disjointness_summary = validate_fold_disjointness(folds, dev_sample_ids)

    for fold in folds:
        train_indices = np.array(fold["train_indices"], dtype=int)
        validation_indices = np.array(fold["validation_indices"], dtype=int)
        fold["train_group_counts"] = _class_counts_by_group(dev_groups[train_indices])
        fold["validation_group_counts"] = _class_counts_by_group(dev_groups[validation_indices])

    write_folds_json(folds, DEV_FOLDS_PATH)
    write_split_audit(
        folds=folds,
        output_path=SPLIT_AUDIT_PATH,
        dev_y_path=DEV_Y_PATH.relative_to(ROOT),
        dev_sample_ids_path=DEV_SAMPLE_IDS_PATH.relative_to(ROOT),
        ndd_sample_ids_path=NDD_SAMPLE_IDS_PATH.relative_to(ROOT),
        ext_sample_ids_path=EXT_SAMPLE_IDS_PATH.relative_to(ROOT),
        dev_folds_path=DEV_FOLDS_PATH.relative_to(ROOT),
        subject_identifier_note=SUBJECT_IDENTIFIER_NOTE,
    )
    append_if_missing(DECISION_LOG_PATH, DECISION_LOG_ENTRY)

    print(f"number of folds: {disjointness_summary['fold_count']}")
    for fold in folds:
        print(
            f"fold {fold['fold']}: "
            f"train={len(fold['train_sample_ids'])} "
            f"validation={len(fold['validation_sample_ids'])} "
            f"train_class_counts={fold['train_group_counts']} "
            f"validation_class_counts={fold['validation_group_counts']}"
        )
    print("within-fold train/validation overlap: none detected")
    print(
        "dev/ndd/ext overlap: none detected "
        f"(dev={overlap_summary['dev_count']}, ndd={overlap_summary['ndd_count']}, ext={overlap_summary['ext_count']})"
    )


if __name__ == "__main__":
    main()
