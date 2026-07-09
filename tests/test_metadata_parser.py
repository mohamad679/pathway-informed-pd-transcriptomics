from __future__ import annotations

import gzip
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data.metadata import (
    iter_series_matrix_metadata_lines,
    parse_series_matrix_metadata,
    summarize_platforms,
)


def write_fixture(path: Path) -> None:
    content = "\n".join(
        [
            '!Series_geo_accession\t"GSETEST"',
            '!Series_platform_id\t"GPLTEST"',
            '!Sample_geo_accession\t"GSM1"\t"GSM2"',
            '!Sample_title\t"sample one"\t"sample two"',
            '!Sample_characteristics_ch1\t"group: alpha"\t"group: beta"',
            "!series_matrix_table_begin",
            "ID_REF\tGSM1\tGSM2",
            "1007_s_at\t1.0\t2.0",
        ]
    )
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(content)


def test_iter_series_matrix_metadata_lines_stops_before_table(tmp_path: Path) -> None:
    matrix_path = tmp_path / "synthetic_series_matrix.txt.gz"
    write_fixture(matrix_path)

    lines = list(iter_series_matrix_metadata_lines(matrix_path))

    assert "!series_matrix_table_begin" not in lines
    assert "1007_s_at\t1.0\t2.0" not in lines
    assert lines[-1] == '!Sample_characteristics_ch1\t"group: alpha"\t"group: beta"'


def test_parse_series_matrix_metadata_extracts_platforms_and_sample_count(
    tmp_path: Path,
) -> None:
    matrix_path = tmp_path / "synthetic_series_matrix.txt.gz"
    write_fixture(matrix_path)

    parsed = parse_series_matrix_metadata(matrix_path)

    assert parsed["sample_count"] == 2
    assert parsed["expression_table_read"] is False
    assert summarize_platforms(parsed) == {"GPLTEST": 2}
