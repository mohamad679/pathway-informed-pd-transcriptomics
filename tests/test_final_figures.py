from __future__ import annotations

import importlib.util
from pathlib import Path
import shutil

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "07_generate_final_figures.py"
EXPECTED_FIGURES = (
    "fig02_development_model_comparison.png",
    "fig03_permutation_validation.png",
    "fig04_top_pathways.png",
    "fig05_attribution_agreement.png",
    "fig06_external_validation.png",
)


def _load_script_module():
    spec = importlib.util.spec_from_file_location("generate_final_figures", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _copy_required_artifacts(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    for relative in (
        "results/development/baseline_summary.csv",
        "results/development/binn_cv.csv",
        "results/development/statistical_validation_bootstrap_ci.csv",
        "results/development/statistical_validation_permutation_null.csv",
        "results/development/statistical_validation_summary.json",
        "results/development/pathway_activation_stability.csv",
        "results/development/pathway_ig_stability.csv",
        "results/development/pathway_attribution_global_agreement.csv",
        "results/external/external_metrics.json",
    ):
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    (root / "results" / "figures").mkdir(parents=True, exist_ok=True)
    return root


def test_figure_script_has_no_training_scoring_or_raw_array_loading() -> None:
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    forbidden = (
        "torch",
        "sklearn",
        "np.load",
        "numpy.load",
        "fit(",
        "predict(",
        "predict_proba",
        "dev_X",
        "dev_y",
        "ext_X",
        "ext_y",
        "ndd_X",
        "data/raw",
        "data/processed",
    )
    assert not [token for token in forbidden if token in source]


def test_generate_final_figures_in_temporary_project(tmp_path: Path) -> None:
    module = _load_script_module()
    temp_root = _copy_required_artifacts(tmp_path)
    outputs = module.generate_final_figures(temp_root)
    assert [path.name for path in outputs] == list(EXPECTED_FIGURES)
    for path in outputs:
        assert path.is_file()
        assert path.stat().st_size > 0
    top20 = pd.read_csv(temp_root / "results" / "development" / "final_top20_pathways.csv")
    assert len(top20) == 20
    assert {"pathway_name", "short_label", "mean_rank"}.issubset(top20.columns)


def test_committed_final_figures_exist_after_generation() -> None:
    for filename in EXPECTED_FIGURES:
        path = ROOT / "results" / "figures" / filename
        assert path.is_file()
        assert path.stat().st_size > 0


def test_top20_companion_csv_has_exactly_20_rows_after_generation() -> None:
    path = ROOT / "results" / "development" / "final_top20_pathways.csv"
    assert path.is_file()
    top20 = pd.read_csv(path)
    assert len(top20) == 20


def test_numeric_annotations_are_sourced_from_committed_values() -> None:
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    assert "empirical p={empirical_p:.6f}" in source
    assert "Global Spearman = {spearman:.6f}" in source
    assert "Balanced accuracy = 0.500000" in source
    assert "Brier = 0.694444; ECE = 0.694441" in source
