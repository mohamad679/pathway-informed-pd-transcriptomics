from __future__ import annotations

from collections import Counter
from pathlib import Path


GSE99039_FIELDS_USED = ("characteristics_ch1", "description")
GSE6613_FIELDS_USED = ("characteristics_ch1", "description", "title")

GSE99039_PD_LABELS = {"IPD"}
GSE99039_HC_LABELS = {"CONTROL"}
GSE99039_NDD_LABELS = {
    "CBD",
    "HD",
    "HD_HD_BATCH",
    "MSA",
    "PD_DEMENTIA",
    "PSP",
    "Vascular dementia",
}
GSE99039_EXCLUDE_LABELS = {
    "ATYPICAL_PD",
    "DRD",
    "DRD-DYT5",
    "GPD",
    "GENETIC_UNAFFECTED",
}
LABEL_KEYWORDS = (
    "control",
    "parkinson",
    "disease label",
    "ipd",
    "gpd",
    "ndd",
    "neuro",
    "alzheimer",
    "psp",
    "msa",
    "cbd",
    "tremor",
    "dementia",
    "unaffected",
)


def _as_text_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def _field_text(sample_metadata: dict[str, object], field_names: tuple[str, ...]) -> str:
    parts: list[str] = []
    for field_name in field_names:
        parts.extend(_as_text_list(sample_metadata.get(field_name)))
    return "\n".join(parts).lower()


def _extract_disease_label(sample_metadata: dict[str, object]) -> str | None:
    for value in _as_text_list(sample_metadata.get("characteristics_ch1")):
        prefix = "disease label:"
        if value.lower().startswith(prefix):
            return value.split(":", 1)[1].strip()
    return None


def extract_candidate_label_fields(parsed_metadata: dict[str, object]) -> list[str]:
    samples = parsed_metadata.get("samples", {})
    if not isinstance(samples, dict):
        return []

    matching_fields: set[str] = set()
    for sample in samples.values():
        if not isinstance(sample, dict):
            continue
        for field_name, value in sample.items():
            if field_name == "sample_id":
                continue
            text = "\n".join(_as_text_list(value)).lower()
            if text and any(keyword in text for keyword in LABEL_KEYWORDS):
                matching_fields.add(field_name)
    return sorted(matching_fields)


def infer_gse99039_group(sample_metadata: dict[str, object]) -> str:
    disease_label = _extract_disease_label(sample_metadata)
    if disease_label in GSE99039_PD_LABELS:
        return "PD"
    if disease_label in GSE99039_HC_LABELS:
        return "HC"
    if disease_label in GSE99039_NDD_LABELS:
        return "NDD"
    if disease_label in GSE99039_EXCLUDE_LABELS:
        return "EXCLUDE"

    text = _field_text(sample_metadata, GSE99039_FIELDS_USED)
    if "control patient" in text:
        return "HC"
    if "genetic_unaffected patient" in text:
        return "EXCLUDE"
    if "ipd patient" in text:
        return "PD"
    if "gpd patient" in text:
        return "EXCLUDE"
    if any(marker in text for marker in ("hd patient", "msa patient", "psp patient")):
        return "NDD"
    if "cbd patient" in text:
        return "NDD"
    if any(marker in text for marker in ("drd patient", "drd-dyt5 patient", "atypical_pd patient")):
        return "EXCLUDE"
    if "pd_dementia patient" in text:
        return "NDD"
    if "vascular dementia patient" in text or "hd_hd_batch patient" in text:
        return "NDD"
    return "UNKNOWN"


def infer_gse6613_group(sample_metadata: dict[str, object]) -> str:
    text = _field_text(sample_metadata, GSE6613_FIELDS_USED)
    if "neurological disease control" in text:
        return "EXCLUDE"
    if "healthy control" in text:
        return "HC"
    if "parkinson's disease" in text or "parkinsons disease" in text:
        return "PD"
    return "UNKNOWN"


def summarize_label_counts(accession: str, sample_to_group: dict[str, str]) -> dict[str, int]:
    label_counts = Counter(sample_to_group.values())
    if accession == "GSE99039":
        order = ("PD", "HC", "NDD", "EXCLUDE", "UNKNOWN")
    elif accession == "GSE6613":
        order = ("PD", "HC", "EXCLUDE", "UNKNOWN")
    else:
        order = tuple(sorted(label_counts))
    return {label: label_counts.get(label, 0) for label in order}


def write_label_audit_report(
    accession_to_summary: dict[str, dict[str, object]], output_path: str | Path
) -> None:
    lines = [
        "# Label Audit",
        "",
        "This report covers Phase 1 metadata-only label extraction from cached GEO series matrix files.",
        "The expression table was not read.",
        "",
    ]

    for accession in sorted(accession_to_summary):
        summary = accession_to_summary[accession]
        fields_used = ", ".join(f"`{field}`" for field in summary["fields_used"])
        candidate_fields = ", ".join(f"`{field}`" for field in summary["candidate_fields"])
        counts = ", ".join(
            f"`{label}`={count}" for label, count in summary["label_counts"].items()
        )
        lines.extend(
            [
                f"## {accession}",
                "",
                f"- Raw file: `{summary['path']}`",
                f"- Sample count: `{summary['sample_count']}`",
                f"- Metadata lines read before table marker: `{summary['metadata_line_count']}`",
                f"- Exact metadata fields used for label inference: {fields_used}",
                f"- Candidate label-bearing fields observed: {candidate_fields}",
                f"- Label rule: {summary['rule_text']}",
                f"- Counts per group: {counts}",
                "- Expression table was not read.",
                "",
            ]
        )

    Path(output_path).write_text("\n".join(lines), encoding="utf-8")
