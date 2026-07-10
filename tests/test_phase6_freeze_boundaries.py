from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "06_freeze_model.py"


def test_freeze_script_does_not_reference_prohibited_inputs_or_outputs() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    prohibited = (
        "ext_X",
        "ext_y",
        "ndd_X",
        "ext_sample_ids",
        "ndd_sample_ids",
        "results/external",
    )
    for token in prohibited:
        assert token not in source


def test_allowed_development_input_list_is_explicit() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "ALLOWED_INPUTS" in source
    for token in (
        "dev_X.npy",
        "dev_y.npy",
        "gene_space.txt",
        "pathway_names.txt",
        "pathway_mask.npz",
        "binn_cv.csv",
    ):
        assert token in source
