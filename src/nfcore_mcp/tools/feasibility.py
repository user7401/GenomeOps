"""Feasibility checking tool: given a goal + files, what pipeline fits and what's missing."""

import logging
import re
from pathlib import Path
from typing import Any

from nfcore_mcp.clients.nfcore_api import get_pipelines

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Static knowledge base — updated rarely (tracks nf-core releases)
# ---------------------------------------------------------------------------

# Keyword sets for scoring a user's goal against a pipeline.
# Keys are nf-core pipeline names; values are lowercased terms that signal fit.
_GOAL_KEYWORDS: dict[str, list[str]] = {
    "rnaseq": [
        "rna", "rna-seq", "rnaseq", "transcriptom", "gene expression",
        "mrna", "salmon", "star", "hisat", "deseq", "edger",
        "differential expression", "kallisto", "rsem",
    ],
    "sarek": [
        "variant", "snp", "indel", "mutation", "germline", "somatic",
        "tumor", "tumour", "cancer", "wgs", "whole genome", "wes",
        "exome", "variant calling", "gatk", "strelka",
    ],
    "chipseq": [
        "chip", "chip-seq", "chipseq", "histone", "peak calling",
        "transcription factor", "tf binding", "h3k27ac", "h3k4me3",
    ],
    "atacseq": [
        "atac", "atac-seq", "atacseq", "chromatin accessibility",
        "open chromatin", "nucleosome",
    ],
    "methylseq": [
        "methylat", "bisulfite", "wgbs", "rrbs", "cpg", "epigenome",
        "dna methylation", "methyl",
    ],
    "scrnaseq": [
        "single cell", "single-cell", "scrna", "10x genomics", "10x",
        "cell ranger", "cellranger", "seurat", "scanpy", "sc rna",
    ],
    "mag": [
        "metagenom", "metagenome", "microbiome assembly",
        "metagenome-assembled genome", "mag", "binning",
    ],
    "ampliseq": [
        "amplicon", "16s", "18s rrna", "its", "microbiome amplicon",
        "marker gene", "qiime", "dada2",
    ],
    "hic": [
        "hi-c", "hic", "3d genome", "chromatin conformation",
        "chromosome conformation", "tad", "loop",
    ],
    "differentialabundance": [
        "differential abundance", "count matrix", "feature counts",
        "deseq2 downstream", "edger downstream", "salmon output",
    ],
    "nanoseq": [
        "nanopore", "oxford nanopore", "ont", "long read", "pacbio",
        "nanopore variant", "nanopore rna",
    ],
    "taxprofiler": [
        "taxonomic", "taxonomy", "profil", "kraken", "bracken",
        "metagenomics classification", "read classification",
    ],
    "proteomics": [
        "proteom", "mass spec", "mass spectrometry", "peptide",
        "protein quantif", "dda", "dia", "maxquant", "fragpipe",
    ],
}

# What file types each pipeline accepts as primary input.
_PIPELINE_ACCEPTS: dict[str, set[str]] = {
    "rnaseq":               {"fastq", "bam"},
    "sarek":                {"fastq", "bam", "cram", "vcf"},
    "chipseq":              {"fastq"},
    "atacseq":              {"fastq"},
    "methylseq":            {"fastq"},
    "scrnaseq":             {"fastq"},
    "mag":                  {"fastq"},
    "ampliseq":             {"fastq"},
    "hic":                  {"fastq"},
    "nanoseq":              {"fastq", "bam"},
    "taxprofiler":          {"fastq", "bam"},
    "differentialabundance": {"csv", "tsv", "txt", "rds"},
    "proteomics":           {"raw", "mzml", "mzxml", "d"},
}

# Requirements that cannot be inferred from file paths alone.
# Each tuple: (requirement_key, human-readable gap message)
_PIPELINE_GAPS: dict[str, list[tuple[str, str]]] = {
    "rnaseq": [
        ("genome", "Reference genome — specify --genome (e.g. GRCh38) or provide a FASTA file"),
        ("strandedness", "Library strandedness (forward/reverse/unstranded) — check your sequencing kit or run RSeQC"),
    ],
    "sarek": [
        ("genome", "Reference genome — specify --genome (e.g. GRCh38) or provide a FASTA + VCF bundle"),
        ("sample_status", "Tumor/normal status for each sample must be annotated in the samplesheet"),
    ],
    "chipseq": [
        ("genome", "Reference genome required"),
        ("antibody", "Antibody target for each sample must be specified in the samplesheet"),
        ("control", "Input/control samples should be paired with ChIP samples"),
    ],
    "atacseq": [
        ("genome", "Reference genome required"),
    ],
    "methylseq": [
        ("genome", "Reference genome required (used for bismark index building)"),
    ],
    "scrnaseq": [
        ("genome", "Reference genome and GTF annotation required"),
        ("protocol", "Sequencing protocol/chemistry must be specified (e.g. 10XV2, 10XV3)"),
    ],
    "mag": [
        ("assembler", "Assembler choice (SPAdes/Megahit) — defaults work but worth confirming"),
    ],
    "ampliseq": [
        ("primers", "Primer sequences or metadata file required"),
        ("metadata", "Sample metadata (grouping variables) needed for differential analysis"),
    ],
    "hic": [
        ("genome", "Reference genome required"),
        ("restriction_site", "Restriction enzyme site must be specified"),
    ],
    "nanoseq": [
        ("genome", "Reference genome required"),
        ("protocol", "Protocol (DNA/RNA) must be specified"),
    ],
    "taxprofiler": [
        ("databases", "Classifier database paths (Kraken2, Bracken, etc.) must be provided"),
    ],
    "differentialabundance": [
        ("sample_metadata", "Sample metadata/contrast table required for comparison groups"),
        ("gtf", "GTF annotation file needed for gene-level summaries"),
    ],
    "proteomics": [
        ("database", "Protein sequence database (FASTA) required for peptide search"),
    ],
}

# File extensions → canonical type name
_EXT_TO_TYPE: dict[str, str] = {
    ".fastq.gz": "fastq", ".fq.gz": "fastq",
    ".fastq": "fastq",    ".fq": "fastq",
    ".bam": "bam",        ".cram": "cram",
    ".vcf": "vcf",        ".vcf.gz": "vcf",
    ".bed": "bed",        ".bed.gz": "bed",
    ".csv": "csv",        ".tsv": "tsv",
    ".txt": "txt",        ".rds": "rds",
    ".raw": "raw",        ".mzml": "mzml",   ".mzxml": "mzxml",
    ".fa": "fasta",       ".fasta": "fasta",  ".fa.gz": "fasta",
    ".fna": "fasta",      ".fna.gz": "fasta",
    ".gtf": "gtf",        ".gff": "gff",
    ".bai": "index",      ".crai": "index",
}

_R1_RE = re.compile(r"(_R1_001|_R1|_1)(\.(fastq|fq)(\.gz)?)$")
_R2_RE = re.compile(r"(_R2_001|_R2|_2)(\.(fastq|fq)(\.gz)?)$")

_OVERALL_LABELS = {
    "yes":           "Ready — all required information appears to be present.",
    "yes_with_gaps": "Likely feasible — files match, but some information must still be supplied.",
    "unlikely":      "Possible mismatch — files may not match any pipeline that fits the goal.",
    "no":            "Not feasible — file types are incompatible with any matching pipeline.",
}


async def check_feasibility(
    user_goal: str,
    file_paths: list[str],
    organism: str | None = None,
) -> dict[str, Any]:
    """Check whether an analysis goal is feasible with nf-core pipelines given the available files.

    This is the entry-point tool for users who arrive with data and a goal but have
    not yet chosen a pipeline. It answers three questions:
      1. Which nf-core pipeline(s) best match the goal?
      2. Do the provided files satisfy that pipeline's input requirements?
      3. What additional information or files are still needed?

    Does NOT require files to be readable — analysis is based entirely on file
    paths, names, and extensions. Use this before get_samplesheet_schema or
    generate_samplesheet.

    Args:
        user_goal: Plain-language description of what the user wants to do.
            E.g. "differential gene expression in human fibroblasts",
                 "somatic variant calling from tumour/normal WGS",
                 "16S microbiome profiling of gut samples".
        file_paths: List of input file paths (local or cloud). Extensions and
            naming patterns are used to infer data type and sample structure.
            Include ALL relevant files — FASTQs, BAMs, reference files, etc.
        organism: Optional organism hint (e.g. "human", "mouse", "zebrafish").
            Helps resolve whether a reference genome is already covered.

    Returns:
        {
            "goal": str,
            "data_summary": {
                "file_count": int,
                "sample_count": int,
                "data_type": str,
                "file_types": [str],
                "paired_end": bool,
                "has_reference_genome": bool,
                "has_index_files": bool
            },
            "matched_pipelines": [
                {
                    "name": str,
                    "confidence": "high" | "medium" | "low",
                    "match_reason": str,
                    "sufficient": [str],
                    "missing": [str],
                    "ready_to_proceed": bool,
                    "next_steps": [str]
                }
            ],
            "overall_feasibility": "yes" | "yes_with_gaps" | "unlikely" | "no",
            "feasibility_summary": str,
            "review_required": true
        }
    """
    logger.info("Tool call: check_feasibility(goal=%r, files=%d)", user_goal, len(file_paths))

    try:
        all_pipelines = await get_pipelines()
    except Exception as exc:
        logger.warning("Could not fetch pipeline list: %s", exc)
        all_pipelines = []

    data_summary = _analyze_files(file_paths)
    pipeline_scores = _score_pipelines(user_goal, all_pipelines)
    top_matches = _filter_by_file_compatibility(pipeline_scores, data_summary)

    matched = []
    for name, score, reason in top_matches[:3]:
        assessment = _assess_pipeline(name, data_summary, organism)
        matched.append({
            "name": name,
            "confidence": _confidence_label(score),
            "match_reason": reason,
            "sufficient": assessment["sufficient"],
            "missing": assessment["missing"],
            "ready_to_proceed": len(assessment["missing"]) == 0,
            "next_steps": _build_next_steps(name, assessment["missing"]),
        })

    overall = _overall_feasibility(matched, data_summary)

    return {
        "goal": user_goal,
        "data_summary": data_summary,
        "matched_pipelines": matched,
        "overall_feasibility": overall,
        "feasibility_summary": _OVERALL_LABELS[overall],
        "review_required": True,
    }


# ---------------------------------------------------------------------------
# File analysis
# ---------------------------------------------------------------------------

def _analyze_files(file_paths: list[str]) -> dict[str, Any]:
    type_counts: dict[str, int] = {}
    r1_samples: set[str] = set()
    r2_samples: set[str] = set()
    other_samples: set[str] = set()
    has_reference = False
    has_index = False

    for fp in file_paths:
        p = Path(fp)
        ext = _detect_extension(p)
        ftype = _EXT_TO_TYPE.get(ext, "other")
        type_counts[ftype] = type_counts.get(ftype, 0) + 1

        if ftype in {"fasta", "gtf", "gff"}:
            has_reference = True
        if ftype == "index":
            has_index = True

        if ftype == "fastq":
            name = fp
            if _R1_RE.search(name):
                r1_samples.add(_strip_read_suffix(p.name))
            elif _R2_RE.search(name):
                r2_samples.add(_strip_read_suffix(p.name))
            else:
                other_samples.add(p.stem)
        elif ftype in {"bam", "cram", "vcf"}:
            other_samples.add(p.stem)

    paired = bool(r1_samples and r2_samples and r1_samples & r2_samples)
    sample_count = max(len(r1_samples), len(other_samples), 1) if (r1_samples or other_samples) else 0

    primary_types = [t for t in type_counts if t not in {"index", "fasta", "gtf", "gff", "other"}]
    data_type = _describe_data_type(primary_types, paired)

    return {
        "file_count": len(file_paths),
        "sample_count": sample_count,
        "data_type": data_type,
        "file_types": sorted(set(type_counts.keys())),
        "paired_end": paired,
        "has_reference_genome": has_reference,
        "has_index_files": has_index,
        "_primary_types": primary_types,  # used internally for matching
    }


def _detect_extension(p: Path) -> str:
    name = p.name.lower()
    for ext in sorted(_EXT_TO_TYPE.keys(), key=len, reverse=True):
        if name.endswith(ext):
            return ext
    return p.suffix.lower()


def _strip_read_suffix(name: str) -> str:
    return _R1_RE.sub("", _R2_RE.sub("", name))


def _describe_data_type(types: list[str], paired: bool) -> str:
    if "fastq" in types:
        return f"{'paired-end' if paired else 'single-end'} FASTQ reads"
    if "bam" in types:
        return "aligned reads (BAM)"
    if "cram" in types:
        return "aligned reads (CRAM)"
    if "vcf" in types:
        return "variant calls (VCF)"
    if types:
        return ", ".join(types)
    return "unknown"


# ---------------------------------------------------------------------------
# Pipeline scoring
# ---------------------------------------------------------------------------

def _score_pipelines(
    goal: str,
    all_pipelines: list[dict[str, Any]],
) -> list[tuple[str, float, str]]:
    """Return (name, score, reason) triples, sorted descending by score."""
    goal_lower = goal.lower()
    scores: list[tuple[str, float, str]] = []

    # Score against our static keyword map first
    keyword_scores: dict[str, tuple[float, list[str]]] = {}
    for pipeline_name, keywords in _GOAL_KEYWORDS.items():
        hits = [kw for kw in keywords if kw in goal_lower]
        if hits:
            keyword_scores[pipeline_name] = (len(hits) / len(keywords), hits)

    # Score against live pipeline list (name, description, topics)
    live_names = {p["name"] for p in all_pipelines}
    for pipeline in all_pipelines:
        name = pipeline["name"]
        base_score, hits = keyword_scores.get(name, (0.0, []))

        desc_lower = pipeline.get("description", "").lower()
        topic_lower = " ".join(pipeline.get("topics", [])).lower()
        combined = f"{name} {desc_lower} {topic_lower}"

        # Extra points for goal words appearing in live metadata
        goal_words = set(re.findall(r"\b\w{4,}\b", goal_lower))
        meta_hits = [w for w in goal_words if w in combined]
        live_score = len(meta_hits) * 0.05

        total = base_score + live_score
        if total > 0:
            reasons = hits if hits else meta_hits
            reason = (
                f"Goal mentions {', '.join(repr(h) for h in reasons[:3])} "
                f"which matches nf-core/{name}"
            )
            scores.append((name, total, reason))

    # Pipelines in our keyword map that aren't in the live list still count
    for name, (score, hits) in keyword_scores.items():
        if name not in live_names and score > 0:
            reason = f"Goal mentions {', '.join(repr(h) for h in hits[:3])} which matches nf-core/{name}"
            scores.append((name, score, reason))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


def _filter_by_file_compatibility(
    scores: list[tuple[str, float, str]],
    data_summary: dict[str, Any],
) -> list[tuple[str, float, str]]:
    """Keep only pipelines whose accepted file types overlap with what the user has."""
    user_types = set(data_summary["_primary_types"])
    if not user_types:
        return scores  # no files provided — return all matches

    compatible = []
    for name, score, reason in scores:
        accepted = _PIPELINE_ACCEPTS.get(name, set())
        if not accepted or user_types & accepted:
            compatible.append((name, score, reason))

    return compatible if compatible else scores  # fall back to all if nothing matches


# ---------------------------------------------------------------------------
# Gap analysis
# ---------------------------------------------------------------------------

def _assess_pipeline(
    name: str,
    data_summary: dict[str, Any],
    organism: str | None,
) -> dict[str, list[str]]:
    sufficient = []
    missing = []
    primary_types = data_summary["_primary_types"]
    accepted = _PIPELINE_ACCEPTS.get(name, set())

    # What's present
    if primary_types:
        dt = data_summary["data_type"]
        if accepted and set(primary_types) & accepted:
            sufficient.append(f"Input file type ({dt}) is compatible with nf-core/{name}")
    if data_summary["sample_count"] > 0:
        sufficient.append(
            f"{data_summary['sample_count']} sample(s) detected from file naming"
        )
    if data_summary["paired_end"]:
        sufficient.append("Paired-end R1/R2 files are correctly matched by name")
    if data_summary["has_reference_genome"]:
        sufficient.append("A reference genome file (FASTA/GFF/GTF) is present in your file list")
    if data_summary["has_index_files"]:
        sufficient.append("Index files (.bai/.crai) detected alongside alignment files")

    # What's missing — from static gap table
    for req_key, gap_msg in _PIPELINE_GAPS.get(name, []):
        if req_key == "genome" and (data_summary["has_reference_genome"] or _organism_has_igenomes(organism)):
            if _organism_has_igenomes(organism):
                sufficient.append(
                    f"Organism '{organism}' is available in nf-core iGenomes — "
                    f"you can use --genome instead of a FASTA file"
                )
            # genome gap resolved
            continue
        missing.append(gap_msg)

    # Generic gaps
    if not primary_types:
        missing.append("No recognised input files detected — check file extensions")
    if data_summary["sample_count"] == 0:
        missing.append("Sample names could not be inferred from file paths")

    return {"sufficient": sufficient, "missing": missing}


def _organism_has_igenomes(organism: str | None) -> bool:
    if not organism:
        return False
    known = {
        "human", "homo sapiens", "hg38", "grch38", "hg19", "grch37",
        "mouse", "mus musculus", "mm10", "grcm38", "mm39", "grcm39",
        "rat", "rattus norvegicus", "rnor6", "zebrafish", "danio rerio",
        "drosophila", "saccharomyces cerevisiae", "yeast",
        "arabidopsis", "c elegans", "caenorhabditis elegans",
    }
    return organism.lower().strip() in known


def _build_next_steps(name: str, missing: list[str]) -> list[str]:
    steps = []
    if not missing:
        steps.append(f"Run get_samplesheet_schema('{name}') to see the required CSV format")
        steps.append(f"Run generate_samplesheet('{name}', file_paths) to draft your samplesheet")
        steps.append(f"Run validate_samplesheet('{name}', csv) to check for errors")
        steps.append(f"Run get_parameters('{name}') to review available options")
        return steps

    if any("genome" in m.lower() for m in missing):
        steps.append(
            f"Decide on a reference genome: use --genome <name> for iGenomes "
            f"(run get_parameters('{name}', group='reference') to see options) "
            f"or provide your own FASTA"
        )
    if any("strandedness" in m.lower() for m in missing):
        steps.append(
            "Determine library strandedness from your sequencing kit documentation "
            "or run RSeQC/infer_experiment.py on a small subset of reads"
        )
    if missing:
        steps.append(
            f"Resolve the items listed in 'missing', then run "
            f"get_samplesheet_schema('{name}') to continue"
        )
    return steps


# ---------------------------------------------------------------------------
# Overall verdict
# ---------------------------------------------------------------------------

def _overall_feasibility(
    matched: list[dict[str, Any]],
    data_summary: dict[str, Any],
) -> str:
    if not matched:
        return "unlikely"
    if not data_summary["_primary_types"]:
        return "unlikely"

    top = matched[0]
    if top["ready_to_proceed"]:
        return "yes"
    # High-confidence match with only soft gaps (genome, strandedness)
    if top["confidence"] in {"high", "medium"} and top["sufficient"]:
        return "yes_with_gaps"
    if top["confidence"] == "low" and not top["sufficient"]:
        return "unlikely"
    return "yes_with_gaps"


def _confidence_label(score: float) -> str:
    if score >= 0.25:
        return "high"
    if score >= 0.08:
        return "medium"
    return "low"
