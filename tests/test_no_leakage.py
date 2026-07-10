from __future__ import annotations

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data.splits import (
    make_stratified_folds,
    validate_fold_disjointness,
    validate_no_overlap_between_sets,
)


def test_no_train_validation_overlap() -> None:
    sample_ids = [f"GSM{i:05d}" for i in range(20)]
    y = [0] * 10 + [1] * 10

    folds = make_stratified_folds(y, sample_ids, n_splits=5, random_state=17)

    for fold in folds:
        assert set(fold["train_sample_ids"]).isdisjoint(set(fold["validation_sample_ids"]))


def test_validation_sets_are_disjoint_and_cover_all_samples_once() -> None:
    sample_ids = [f"GSM{i:05d}" for i in range(30)]
    y = [0] * 15 + [1] * 15

    folds = make_stratified_folds(y, sample_ids, n_splits=5, random_state=23)
    summary = validate_fold_disjointness(folds, sample_ids)

    validation_ids = []
    for fold in folds:
        validation_ids.extend(fold["validation_sample_ids"])

    assert summary["validation_coverage_count"] == len(sample_ids)
    assert sorted(validation_ids) == sorted(sample_ids)
    assert len(validation_ids) == len(set(validation_ids))


def test_dev_ndd_ext_overlap_detection() -> None:
    dev_sample_ids = ["GSM00001", "GSM00002"]
    ndd_sample_ids = ["GSM00003", "GSM00004"]
    ext_sample_ids = ["GSM00002", "GSM00005"]

    with pytest.raises(ValueError, match="dev/ext overlap"):
        validate_no_overlap_between_sets(dev_sample_ids, ndd_sample_ids, ext_sample_ids)


def test_binary_fold_balance_is_stratified_enough() -> None:
    sample_ids = [f"GSM{i:05d}" for i in range(40)]
    y = [0] * 24 + [1] * 16

    folds = make_stratified_folds(y, sample_ids, n_splits=5, random_state=29)

    validation_zero_counts = [fold["validation_class_counts"]["0"] for fold in folds]
    validation_one_counts = [fold["validation_class_counts"]["1"] for fold in folds]

    assert max(validation_zero_counts) - min(validation_zero_counts) <= 1
    assert max(validation_one_counts) - min(validation_one_counts) <= 1
