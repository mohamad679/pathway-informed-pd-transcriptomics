from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LIMITATIONS = ROOT / "docs" / "limitations.md"


def _limitations_text() -> str:
    return LIMITATIONS.read_text(encoding="utf-8")


def test_limitations_document_exists() -> None:
    assert LIMITATIONS.exists()


def test_includes_required_numeric_results() -> None:
    text = _limitations_text()
    for token in (
        "438",
        "72",
        "22",
        "50",
        "48",
        "0.702895",
        "0.695455",
        "0.782081",
        "0.500000",
        "0.694444",
        "0.694441",
        "0.489050",
        "0.520833",
    ):
        assert token in text


def test_includes_limited_permutation_resolution_and_p_value_boundary() -> None:
    text = _limitations_text()
    assert "50 permutations" in text
    assert "1/51 = 0.019608" in text
    assert "cannot support p < 0.01" in text


def test_includes_biological_scope_limitations() -> None:
    text = _limitations_text().lower()
    assert "blood is not brain" in text
    assert "static expression is not rna dynamics" in text
    assert "splicing kinetics" in text
    assert "rna velocity" in text
    assert "isoform-level regulation" in text
    assert "pathway attribution describes model usage, not biological causation" in text


def test_includes_external_calibration_and_threshold_limitations() -> None:
    text = _limitations_text().lower()
    assert "severe external miscalibration" in text
    assert "poor external calibration" in text
    assert "balanced accuracy of 0.500000" in text
    assert "frozen 0.5 decision threshold" in text
    assert "no threshold adjustment, recalibration, or external-data tuning" in text


def test_includes_ndd_stress_test_only_language() -> None:
    text = _limitations_text().lower()
    assert "stress-test-only" in text
    assert "specificity/stress test" in text
    assert "not diagnostic validation" in text
    assert "does not establish disease specificity" in text


def test_includes_reproducibility_and_chain_of_custody_limitations() -> None:
    text = _limitations_text()
    assert "frozen-v1" in text
    assert "HASH_BEFORE" in text
    assert "HASH_AFTER" in text
    assert "match" in text
    assert "Git LFS" in text
    assert "Raw and processed GEO-derived data are not committed" in text


def test_explicitly_rejects_forbidden_claim_classes() -> None:
    text = _limitations_text().lower()
    assert "does not support clinical, diagnostic, deployment, causal, or mechanistic claims" in text
    assert "does not provide a diagnostic tool, clinical evidence, disease mechanism, or deployment-ready model" in text


def test_omits_forbidden_promotional_phrases() -> None:
    text = _limitations_text().lower()
    forbidden_phrases = (
        "clinically validated",
        "deployment ready",
        "diagnostic accuracy",
        "robust generalization",
        "strong external performance",
        "state of the art",
        "sota",
    )
    for phrase in forbidden_phrases:
        assert phrase not in text
