from __future__ import annotations

import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from priors.build_mask import build_pathway_mask
from priors.gmt import read_gmt


GENE_SPACE_PATH = ROOT / "data" / "processed" / "gene_space.txt"
CONFIG_PATH = ROOT / "config" / "pathways.yaml"
OUTPUT_DIR = ROOT / "data" / "processed"
MASK_PATH = OUTPUT_DIR / "pathway_mask.npz"
PATHWAY_NAMES_PATH = OUTPUT_DIR / "pathway_names.txt"
EDGES_PATH = OUTPUT_DIR / "pathway_gene_edges.tsv"
AUDIT_PATH = ROOT / "docs" / "phase3_pathway_mask_audit.md"
DECISION_LOG_PATH = ROOT / "docs" / "decision_log.md"


def append_if_missing(path: Path, block: str) -> None:
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    if block.strip() in existing:
        return
    separator = "" if not existing or existing.endswith("\n\n") else "\n"
    path.write_text(existing + separator + block.rstrip() + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def display_path(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def configured_gmt_paths(config: dict) -> list[Path]:
    try:
        configured_paths = config["msigdb"]["gmt_files"]
    except (KeyError, TypeError) as error:
        raise ValueError("config/pathways.yaml must define msigdb.gmt_files.") from error
    if not isinstance(configured_paths, list) or not configured_paths:
        raise ValueError("config/pathways.yaml must define a non-empty msigdb.gmt_files list.")
    paths = [Path(value) if Path(value).is_absolute() else ROOT / Path(value) for value in configured_paths]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        missing_text = ", ".join(display_path(path) for path in missing)
        raise FileNotFoundError(
            f"Missing configured MSigDB GMT file(s): {missing_text}. "
            "Place GMT files under data/raw/msigdb/ or configure paths in config/pathways.yaml. "
            "Automatic MSigDB downloads are disabled."
        )
    return paths


def main() -> None:
    if not GENE_SPACE_PATH.is_file():
        raise FileNotFoundError(f"Missing fixed gene space: {GENE_SPACE_PATH}.")
    with CONFIG_PATH.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    gmt_paths = configured_gmt_paths(config)
    msigdb_config = config["msigdb"]
    min_genes = int(msigdb_config.get("min_genes_present", 10))
    max_genes = msigdb_config.get("max_genes_present")
    if max_genes is not None:
        max_genes = int(max_genes)

    gene_space = [line.strip() for line in GENE_SPACE_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    gene_sets: dict[str, set[str]] = {}
    for gmt_path in gmt_paths:
        for pathway_name, genes in read_gmt(gmt_path).items():
            if pathway_name in gene_sets:
                raise ValueError(f"Duplicate pathway name across configured GMT files: {pathway_name!r}.")
            gene_sets[pathway_name] = genes

    if max_genes is not None:
        from priors.gmt import filter_gene_sets_to_gene_space

        gene_sets = filter_gene_sets_to_gene_space(
            gene_sets, gene_space, min_genes=min_genes, max_genes=max_genes
        )
    mask, pathway_names, gene_names, stats = build_pathway_mask(gene_space, gene_sets, min_genes=min_genes)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(MASK_PATH, mask)
    PATHWAY_NAMES_PATH.write_text("\n".join(pathway_names) + "\n", encoding="utf-8")
    edges = [
        (pathway_name, gene_names[column])
        for row, pathway_name in enumerate(pathway_names)
        for column in mask.getrow(row).indices
    ]
    pd.DataFrame(edges, columns=["pathway_name", "gene_symbol"]).to_csv(EDGES_PATH, sep="\t", index=False)

    provenance = "\n".join(
        f"- `{display_path(path)}` — SHA-256 `{sha256_file(path)}`" for path in gmt_paths
    )
    metric_rows = "".join(f"| {key} | {value} |\n" for key, value in stats.items())
    audit = f"""# Phase 3 pathway mask audit

Generated at: {datetime.now(timezone.utc).isoformat()}\n
## MSigDB provenance

- Configured MSigDB version: `{msigdb_config.get('version', 'MISSING')}`
- Minimum genes present: `{min_genes}`
- Maximum genes present: `{max_genes}`
{provenance}

## Mask summary

| Metric | Value |
| --- | ---: |
{metric_rows}

Outputs: `{MASK_PATH.relative_to(ROOT)}`, `{PATHWAY_NAMES_PATH.relative_to(ROOT)}`, and `{EDGES_PATH.relative_to(ROOT)}`.
"""
    AUDIT_PATH.write_text(audit, encoding="utf-8")
    append_if_missing(
        DECISION_LOG_PATH,
        """## 2026-07-10 — Phase 3 pathway-mask construction foundation

- Added local-only MSigDB GMT parsing and deterministic sparse pathway-mask construction.
- A real mask is written only when configured local GMT files and the fixed `gene_space.txt` are available.
- The mask audit records configured MSigDB version, exact input paths, and SHA-256 checksums.
- No BINN model, training, attention, Integrated Gradients, external cohort, or held-out NDD data was used.
""",
    )

    print(f"mask shape: {mask.shape}")
    print(f"n_edges: {stats['n_edges']}")
    print(f"density: {stats['density']:.8f}")
    print(f"pathway count: {stats['n_pathways']}")
    print(f"RNA-processing pathway count: {stats['n_rna_processing_pathways']}")
    print(f"genes with no pathway: {stats['n_genes_with_no_pathway']}")


if __name__ == "__main__":
    main()
