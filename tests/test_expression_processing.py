from __future__ import annotations

import gzip
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data.expression import (
    aggregate_probes_to_genes,
    load_gene_space,
    read_series_matrix_expression,
    within_sample_zscore,
)


def _write_series_matrix(path: Path, content: str) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(content)


def test_expression_reader_stops_at_table_end(tmp_path: Path) -> None:
    matrix_path = tmp_path / "tiny_series_matrix.txt.gz"
    _write_series_matrix(
        matrix_path,
        "\n".join(
            [
                '!Series_title\t"synthetic"',
                "!series_matrix_table_begin",
                '"ID_REF"\t"GSM1"\t"GSM2"',
                '"probe_a"\t1.0\t2.0',
                "!series_matrix_table_end",
                '"probe_b"\tnot\tread',
            ]
        )
        + "\n",
    )

    expression_df = read_series_matrix_expression(matrix_path)

    assert list(expression_df.index) == ["probe_a"]
    assert list(expression_df.columns) == ["GSM1", "GSM2"]
    assert expression_df.loc["probe_a", "GSM1"] == 1.0


def test_aggregation_median_across_probes() -> None:
    expression_df = pd.DataFrame(
        {"GSM1": [1.0, 3.0, 8.0], "GSM2": [2.0, 5.0, 10.0]},
        index=pd.Index(["probe1", "probe2", "probe3"], name="probe_id"),
    )
    probe_to_symbol_df = pd.DataFrame(
        {
            "probe_id": ["probe1", "probe2", "probe3"],
            "gene_symbol": ["SNCA", "SNCA", "MAPT"],
        }
    )

    gene_df = aggregate_probes_to_genes(expression_df, probe_to_symbol_df, ["MAPT", "SNCA"])

    assert gene_df.loc["SNCA", "GSM1"] == 2.0
    assert gene_df.loc["SNCA", "GSM2"] == 3.5
    assert gene_df.loc["MAPT", "GSM1"] == 8.0


def test_gene_space_ordering(tmp_path: Path) -> None:
    gene_space_path = tmp_path / "gene_space.txt"
    gene_space_path.write_text("MAPT\nSNCA\nLRRK2\n", encoding="utf-8")

    expression_df = pd.DataFrame(
        {"GSM1": [5.0, 2.0, 7.0], "GSM2": [6.0, 3.0, 8.0]},
        index=pd.Index(["probe1", "probe2", "probe3"], name="probe_id"),
    )
    probe_to_symbol_df = pd.DataFrame(
        {
            "probe_id": ["probe1", "probe2", "probe3"],
            "gene_symbol": ["SNCA", "MAPT", "LRRK2"],
        }
    )

    gene_df = aggregate_probes_to_genes(expression_df, probe_to_symbol_df, load_gene_space(gene_space_path))
    ordered_gene_df = gene_df.loc[load_gene_space(gene_space_path)]

    assert list(ordered_gene_df.index) == ["MAPT", "SNCA", "LRRK2"]


def test_within_sample_zscore_has_zero_mean_and_unit_std() -> None:
    gene_df = pd.DataFrame(
        {
            "GSM1": [1.0, 2.0, 3.0, 4.0],
            "GSM2": [10.0, 20.0, 30.0, 40.0],
        },
        index=["gene_a", "gene_b", "gene_c", "gene_d"],
    )

    zscored = within_sample_zscore(gene_df)

    assert np.allclose(zscored.mean(axis=0).to_numpy(), np.zeros(2), atol=1e-9)
    assert np.allclose(zscored.std(axis=0, ddof=0).to_numpy(), np.ones(2), atol=1e-9)
