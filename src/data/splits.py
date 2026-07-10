from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def load_sample_ids(path: str | Path) -> list[str]:
    sample_id_path = Path(path)
    if not sample_id_path.is_file():
        raise FileNotFoundError(f"Sample ID file not found: {sample_id_path}")
    sample_ids = [line.strip() for line in sample_id_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not sample_ids:
        raise ValueError(f"Sample ID file is empty: {sample_id_path}")
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError(f"Duplicate sample IDs found in {sample_id_path}")
    return sample_ids


def _label_count_map(values: np.ndarray) -> dict[str, int]:
    unique_values, counts = np.unique(values, return_counts=True)
    return {str(value): int(count) for value, count in zip(unique_values.tolist(), counts.tolist(), strict=True)}


def make_stratified_folds(
    y: np.ndarray | list[int] | list[str],
    sample_ids: list[str],
    n_splits: int = 5,
    random_state: int = 20260710,
) -> list[dict[str, object]]:
    y_array = np.asarray(y)
    if y_array.ndim != 1:
        raise ValueError("y must be one-dimensional")
    if len(sample_ids) != len(y_array):
        raise ValueError("sample_ids and y must have the same length")
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("sample_ids must be unique")
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")

    classes, inverse = np.unique(y_array, return_inverse=True)
    for class_index, class_value in enumerate(classes.tolist(), start=0):
        class_size = int(np.sum(inverse == class_index))
        if class_size < n_splits:
            raise ValueError(
                f"Class {class_value!r} has only {class_size} samples, fewer than n_splits={n_splits}"
            )

    rng = np.random.RandomState(random_state)
    fold_to_validation_indices: list[list[int]] = [[] for _ in range(n_splits)]
    for class_index in range(len(classes)):
        class_indices = np.flatnonzero(inverse == class_index).copy()
        rng.shuffle(class_indices)
        for fold_index, chunk in enumerate(np.array_split(class_indices, n_splits), start=0):
            fold_to_validation_indices[fold_index].extend(int(index) for index in chunk.tolist())

    folds: list[dict[str, object]] = []
    for fold_index, validation_indices in enumerate(fold_to_validation_indices, start=1):
        validation_index_array = np.array(sorted(validation_indices), dtype=int)
        train_mask = np.ones(len(sample_ids), dtype=bool)
        train_mask[validation_index_array] = False
        train_index_array = np.flatnonzero(train_mask)
        train_labels = y_array[train_index_array]
        validation_labels = y_array[validation_index_array]
        folds.append(
            {
                "fold": fold_index,
                "train_indices": train_index_array.astype(int).tolist(),
                "validation_indices": validation_index_array.astype(int).tolist(),
                "train_sample_ids": [sample_ids[index] for index in train_index_array.tolist()],
                "validation_sample_ids": [sample_ids[index] for index in validation_index_array.tolist()],
                "train_class_counts": _label_count_map(train_labels),
                "validation_class_counts": _label_count_map(validation_labels),
            }
        )
    return folds


def validate_fold_disjointness(folds: list[dict[str, object]], sample_ids: list[str]) -> dict[str, int]:
    known_sample_ids = set(sample_ids)
    seen_validation_ids: set[str] = set()

    for fold in folds:
        fold_number = int(fold["fold"])
        train_sample_ids = list(fold["train_sample_ids"])
        validation_sample_ids = list(fold["validation_sample_ids"])
        train_indices = [int(index) for index in fold["train_indices"]]
        validation_indices = [int(index) for index in fold["validation_indices"]]

        train_id_set = set(train_sample_ids)
        validation_id_set = set(validation_sample_ids)
        overlap = train_id_set & validation_id_set
        if overlap:
            raise ValueError(f"Fold {fold_number} has train/validation overlap: {sorted(overlap)}")
        if not train_id_set <= known_sample_ids:
            raise ValueError(f"Fold {fold_number} train set contains unknown sample IDs")
        if not validation_id_set <= known_sample_ids:
            raise ValueError(f"Fold {fold_number} validation set contains unknown sample IDs")

        expected_train_ids = [sample_ids[index] for index in train_indices]
        expected_validation_ids = [sample_ids[index] for index in validation_indices]
        if expected_train_ids != train_sample_ids:
            raise ValueError(f"Fold {fold_number} train indices do not match train sample IDs")
        if expected_validation_ids != validation_sample_ids:
            raise ValueError(f"Fold {fold_number} validation indices do not match validation sample IDs")

        validation_overlap = seen_validation_ids & validation_id_set
        if validation_overlap:
            raise ValueError(
                f"Validation sample IDs appear in more than one fold: {sorted(validation_overlap)}"
            )
        seen_validation_ids.update(validation_id_set)

    missing_validation_ids = known_sample_ids - seen_validation_ids
    if missing_validation_ids:
        raise ValueError(
            "Some samples never appear in validation: "
            + ", ".join(sorted(missing_validation_ids))
        )
    extra_validation_ids = seen_validation_ids - known_sample_ids
    if extra_validation_ids:
        raise ValueError(
            "Validation contains unknown samples: "
            + ", ".join(sorted(extra_validation_ids))
        )

    return {
        "fold_count": len(folds),
        "sample_count": len(sample_ids),
        "validation_coverage_count": len(seen_validation_ids),
    }


def validate_no_overlap_between_sets(
    dev_sample_ids: list[str],
    ndd_sample_ids: list[str],
    ext_sample_ids: list[str],
) -> dict[str, int]:
    dev_ndd_overlap = set(dev_sample_ids) & set(ndd_sample_ids)
    dev_ext_overlap = set(dev_sample_ids) & set(ext_sample_ids)
    ndd_ext_overlap = set(ndd_sample_ids) & set(ext_sample_ids)

    if dev_ndd_overlap or dev_ext_overlap or ndd_ext_overlap:
        messages: list[str] = []
        if dev_ndd_overlap:
            messages.append(f"dev/ndd overlap: {sorted(dev_ndd_overlap)}")
        if dev_ext_overlap:
            messages.append(f"dev/ext overlap: {sorted(dev_ext_overlap)}")
        if ndd_ext_overlap:
            messages.append(f"ndd/ext overlap: {sorted(ndd_ext_overlap)}")
        raise ValueError("; ".join(messages))

    return {
        "dev_count": len(dev_sample_ids),
        "ndd_count": len(ndd_sample_ids),
        "ext_count": len(ext_sample_ids),
    }


def write_folds_json(folds: list[dict[str, object]], output_path: str | Path) -> None:
    target_path = Path(output_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(json.dumps(folds, indent=2) + "\n", encoding="utf-8")


def write_split_audit(
    *,
    folds: list[dict[str, object]],
    output_path: str | Path,
    dev_y_path: str | Path,
    dev_sample_ids_path: str | Path,
    ndd_sample_ids_path: str | Path,
    ext_sample_ids_path: str | Path,
    dev_folds_path: str | Path,
    subject_identifier_note: str,
) -> None:
    audit_path = Path(output_path)
    audit_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Split Audit",
        "",
        "This report covers Phase 1 split creation and leakage validation only.",
        "Cross-validation folds were created from the development PD/HC cohort only.",
        "Held-out NDD samples were not used for split creation.",
        "External cohort samples were not used for split creation.",
        subject_identifier_note,
        "",
        "## Inputs",
        "",
        f"- Development labels: `{Path(dev_y_path).as_posix()}`",
        f"- Development sample IDs: `{Path(dev_sample_ids_path).as_posix()}`",
        f"- Held-out NDD sample IDs: `{Path(ndd_sample_ids_path).as_posix()}`",
        f"- External sample IDs: `{Path(ext_sample_ids_path).as_posix()}`",
        "",
        "## Outputs",
        "",
        f"- Fold file: `{Path(dev_folds_path).as_posix()}`",
        f"- Number of folds: `{len(folds)}`",
        "",
        "## Fold Summary",
        "",
    ]

    for fold in folds:
        train_counts = fold.get("train_group_counts", fold["train_class_counts"])
        validation_counts = fold.get("validation_group_counts", fold["validation_class_counts"])
        lines.append(
            f"- Fold {fold['fold']}: train=`{len(fold['train_sample_ids'])}`, "
            f"validation=`{len(fold['validation_sample_ids'])}`, "
            f"train_class_counts=`{train_counts}`, "
            f"validation_class_counts=`{validation_counts}`"
        )

    lines.extend(
        [
            "",
            "## Boundary Confirmation",
            "",
            "- Splits were created from development PD/HC only: `yes`",
            "- Held-out NDD used for split creation: `no`",
            "- External cohort used for split creation: `no`",
            "- Modeling performed: `no`",
            "- Baselines implemented: `no`",
            "- Pathway masks implemented: `no`",
            "- MSigDB logic implemented: `no`",
            "- External validation used for model selection: `no`",
        ]
    )
    audit_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
