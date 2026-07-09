from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data.labels import infer_gse6613_group, infer_gse99039_group


def test_infer_gse99039_pd() -> None:
    sample_metadata = {
        "characteristics_ch1": ["disease label: IPD"],
        "description": ["Whole blood gene expression from IPD patient."],
    }

    assert infer_gse99039_group(sample_metadata) == "PD"


def test_infer_gse99039_hc() -> None:
    sample_metadata = {
        "characteristics_ch1": ["disease label: CONTROL"],
        "description": ["Whole blood gene expression from CONTROL patient."],
    }

    assert infer_gse99039_group(sample_metadata) == "HC"


def test_infer_gse99039_exclude_genetic_unaffected() -> None:
    sample_metadata = {
        "characteristics_ch1": ["disease label: GENETIC_UNAFFECTED"],
        "description": ["Whole blood gene expression from GENETIC_UNAFFECTED patient."],
    }

    assert infer_gse99039_group(sample_metadata) == "EXCLUDE"


def test_infer_gse99039_exclude_gpd() -> None:
    sample_metadata = {
        "characteristics_ch1": ["disease label: GPD"],
        "description": ["Whole blood gene expression from GPD patient."],
    }

    assert infer_gse99039_group(sample_metadata) == "EXCLUDE"


def test_infer_gse99039_ndd() -> None:
    sample_metadata = {
        "characteristics_ch1": ["disease label: PSP"],
        "description": ["Whole blood gene expression from PSP patient."],
    }

    assert infer_gse99039_group(sample_metadata) == "NDD"


def test_infer_gse99039_exclude_atypical_pd() -> None:
    sample_metadata = {
        "characteristics_ch1": ["disease label: ATYPICAL_PD"],
        "description": ["Whole blood gene expression from ATYPICAL_PD patient."],
    }

    assert infer_gse99039_group(sample_metadata) == "EXCLUDE"


def test_infer_gse99039_unknown() -> None:
    sample_metadata = {
        "characteristics_ch1": ["subject_id: synthetic-001"],
        "description": ["Whole blood gene expression from unresolved cohort."],
    }

    assert infer_gse99039_group(sample_metadata) == "UNKNOWN"


def test_infer_gse6613_pd() -> None:
    sample_metadata = {
        "characteristics_ch1": "Parkinson's disease",
        "description": "Whole blood from a patient with Parkinson's disease",
        "title": "Parkinson's disease sample x001",
    }

    assert infer_gse6613_group(sample_metadata) == "PD"


def test_infer_gse6613_hc() -> None:
    sample_metadata = {
        "characteristics_ch1": "healthy control",
        "description": "Whole blood from a healthy control",
        "title": "healthy control sample x004",
    }

    assert infer_gse6613_group(sample_metadata) == "HC"


def test_infer_gse6613_exclude() -> None:
    sample_metadata = {
        "characteristics_ch1": "neurological disease control",
        "description": "Whole blood from a patient with Alzheimer's disease",
        "title": "neurological disease control sample x009",
    }

    assert infer_gse6613_group(sample_metadata) == "EXCLUDE"


def test_infer_gse6613_unknown() -> None:
    sample_metadata = {
        "characteristics_ch1": "unresolved cohort",
        "description": "Whole blood from an unresolved cohort",
        "title": "mystery sample",
    }

    assert infer_gse6613_group(sample_metadata) == "UNKNOWN"
