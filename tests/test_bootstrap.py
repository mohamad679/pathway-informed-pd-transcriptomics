from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from eval.bootstrap import bootstrap_metric_cis


def test_bootstrap_output_has_required_metrics_and_columns() -> None:
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 0.9])

    ci = bootstrap_metric_cis(y_true, y_prob, n_bootstrap=25, seed=7)

    assert list(ci.columns) == ["metric", "estimate", "ci_lower", "ci_upper", "n_bootstrap"]
    assert set(ci["metric"]) == {"auroc", "auprc", "balanced_accuracy", "brier", "ece"}
    assert (ci["n_bootstrap"] == 25).all()


def test_bootstrap_is_deterministic_with_fixed_seed() -> None:
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 0.9])

    first = bootstrap_metric_cis(y_true, y_prob, n_bootstrap=25, seed=11)
    second = bootstrap_metric_cis(y_true, y_prob, n_bootstrap=25, seed=11)

    pd.testing.assert_frame_equal(first, second)
