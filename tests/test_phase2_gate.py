from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from eval.phase2_gate import (  # noqa: E402
    METRIC_NAMES,
    validate_oof_prediction_integrity,
    validate_required_models,
    validate_sanity_gate,
)


MODELS = ["logistic_regression", "random_forest", "unconstrained_mlp"]


def make_summary(auroc: float = 0.7) -> pd.DataFrame:
    rows = []
    for model in MODELS:
        row: dict[str, float | int | str] = {"model": model, "n_seeds": 3, "n_oof_rows": 1314}
        for metric in METRIC_NAMES:
            row[f"{metric}_mean"] = auroc if metric == "auroc" else 0.5
            row[f"{metric}_ci_lower"] = 0.4
            row[f"{metric}_ci_upper"] = 0.8
        rows.append(row)
    return pd.DataFrame(rows)


def make_oof() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model": model,
                "seed": seed,
                "fold": 1,
                "sample_index": sample_index,
                "y_true": sample_index % 2,
                "y_prob": 0.25,
            }
            for model in MODELS
            for seed in (11, 23, 37)
            for sample_index in range(438)
        ]
    )


def test_required_model_validation() -> None:
    validate_required_models(make_summary())
    invalid = make_summary().iloc[:2].copy()
    with pytest.raises(ValueError, match="exactly"):
        validate_required_models(invalid)


def test_oof_integrity_catches_missing_sample_coverage() -> None:
    oof = make_oof()
    mask = (oof["model"] == "logistic_regression") & (oof["seed"] == 11)
    first_index = oof.index[mask][0]
    oof.loc[first_index, "sample_index"] = 1
    with pytest.raises(ValueError, match="unique development samples"):
        validate_oof_prediction_integrity(oof)


def test_sanity_gate_fails_below_auroc_threshold() -> None:
    with pytest.raises(ValueError, match="requires all means > 0.6"):
        validate_sanity_gate(make_summary(auroc=0.6))


def test_sanity_gate_passes_above_auroc_threshold() -> None:
    assert validate_sanity_gate(make_summary(auroc=0.61)) == "logistic_regression"
