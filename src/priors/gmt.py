"""Local GMT parsing and deterministic pathway-set preparation."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping


RNA_PROCESSING_KEYWORDS = (
    "SPLICEOSOME",
    "RNA_TRANSPORT",
    "MRNA_SURVEILLANCE",
    "RNA_DEGRADATION",
    "MRNA_SPLICING",
    "RNA_BINDING",
    "RIBONUCLEOPROTEIN",
)


def normalize_gene_symbol(symbol: str) -> str | None:
    """Return a stable, case-normalized single gene symbol, if present."""
    normalized = str(symbol).strip().strip('"').strip("'").upper()
    if not normalized or any(character.isspace() for character in normalized):
        return None
    return normalized


def read_gmt(path: str | Path) -> dict[str, set[str]]:
    """Read one local GMT file, rejecting malformed or duplicate pathway names."""
    gmt_path = Path(path)
    if not gmt_path.is_file():
        raise FileNotFoundError(f"GMT file does not exist: {gmt_path}")

    gene_sets: dict[str, set[str]] = {}
    with gmt_path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            fields = raw_line.rstrip("\n\r").split("\t")
            if len(fields) < 3:
                raise ValueError(
                    f"Malformed GMT line {line_number} in {gmt_path}: "
                    "expected pathway name, description, and at least one gene."
                )
            pathway_name = fields[0].strip()
            if not pathway_name:
                raise ValueError(f"Missing pathway name on GMT line {line_number} in {gmt_path}.")
            if pathway_name in gene_sets:
                raise ValueError(f"Duplicate pathway name {pathway_name!r} in {gmt_path}.")

            members = {gene for value in fields[2:] if (gene := normalize_gene_symbol(value))}
            if not members:
                raise ValueError(
                    f"Pathway {pathway_name!r} on GMT line {line_number} in {gmt_path} has no valid genes."
                )
            gene_sets[pathway_name] = members
    return gene_sets


def filter_gene_sets_to_gene_space(
    gene_sets: Mapping[str, Iterable[str]],
    gene_space: Iterable[str],
    min_genes: int = 10,
    max_genes: int | None = None,
) -> dict[str, set[str]]:
    """Keep pathways whose normalized members are in the fixed gene space."""
    if min_genes < 1:
        raise ValueError("min_genes must be at least 1.")
    if max_genes is not None and max_genes < min_genes:
        raise ValueError("max_genes must be greater than or equal to min_genes.")

    normalized_space = {
        normalized for gene in gene_space if (normalized := normalize_gene_symbol(gene))
    }
    filtered: dict[str, set[str]] = {}
    for pathway_name in sorted(gene_sets):
        present = {
            normalized
            for gene in gene_sets[pathway_name]
            if (normalized := normalize_gene_symbol(gene)) in normalized_space
        }
        if len(present) < min_genes:
            continue
        if max_genes is not None and len(present) > max_genes:
            continue
        filtered[pathway_name] = present
    return filtered


def classify_rna_processing_pathways(pathway_name: str) -> bool:
    """Classify pathways by the predefined MSigDB RNA-processing keyword set."""
    normalized_name = pathway_name.upper()
    return any(keyword in normalized_name for keyword in RNA_PROCESSING_KEYWORDS)
