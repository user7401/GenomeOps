"""Feasibility checking tool: given a goal + files, what pipeline fits and what's missing.

File identification path:
  1. _probe_files reads the actual content of accessible local files (magic bytes,
     first lines, gzip decompression) — so even a file with a scrambled name is
     identified from what it contains.
  2. _llm_feasibility hands the probes, the goal, and the live pipeline list to
     an LLM which reasons over the actual evidence rather than name patterns.
  3. If no ANTHROPIC_API_KEY is set, or if the LLM call fails, the function falls
     back to the heuristic that reads extensions and name patterns.

For cloud paths (s3://, gs://, etc.) content cannot be read; the probe records the
path type and falls through to extension-based hints for the LLM/heuristic.
"""

import gzip
import logging
import os
import re
from pathlib import Path
from typing import Any

from genomeops_mcp.clients.nfcore_api import get_pipelines

logger = logging.getLogger(__name__)

_MAX_PROBE_LINES = 20
_MAX_PROBE_FILES = 25    # probe a representative sample; summarise the rest

# ---------------------------------------------------------------------------
# Magic-byte table for binary file identification
# ---------------------------------------------------------------------------
_MAGIC: list[tuple[bytes, str]] = [
    (b"BAM\x01",    "BAM binary alignment file"),
    (b"CRAM",       "CRAM compressed alignment file"),
    (b"\x1f\x8b",   "gzip-compressed file"),
    (b"\x89HDF",    "HDF5 file (e.g. 10x Genomics h5)"),
    (b"PK\x03\x04", "ZIP archive"),
    (b"BCF\x02",    "BCF binary variant call file"),
]


# ===========================================================================
# Public tool
# ===========================================================================

async def check_feasibility(
    user_goal: str,
    file_paths: list[str],
    organism: str | None = None,
) -> dict[str, Any]:
    """Check whether an analysis goal is feasible with nf-core pipelines given the available files.

    This is the entry-point tool for users who arrive with data and a goal but have
    not yet chosen a pipeline.

    File identification is done from file *content*, not from names or extensions.
    Even a file whose name has been scrambled is identified from what is inside it
    (FASTQ records, VCF headers, BAM magic bytes, etc.). Files whose format cannot
    be determined are flagged as novel so the agent can ask the user.

    Local files are read for identification; cloud paths (s3://, gs://, etc.) are
    identified from their names only.

    Args:
        user_goal: Plain-language description of what the user wants to do.
        file_paths: List of file paths. Include ALL relevant files — FASTQs, BAMs,
            reference files, metadata tables, etc.
        organism: Optional organism hint (e.g. "human", "mouse"). Helps resolve
            whether a reference genome is already covered by iGenomes.

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
                "has_index_files": bool,
                "file_identifications": [{path,format,confidence,evidence,novel}]
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
            "analysis_method": "llm" | "heuristic",
            "review_required": true
        }
    """
    logger.info("Tool call: check_feasibility(goal=%r, files=%d)", user_goal, len(file_paths))

    try:
        all_pipelines = await get_pipelines()
    except Exception as exc:
        logger.warning("Could not fetch pipeline list: %s", exc)
        all_pipelines = []

    probes = _probe_files(file_paths)

    if os.environ.get("ANTHROPIC_API_KEY"):
        result = await _llm_feasibility(user_goal, probes, organism, all_pipelines, file_paths)
        if result is not None:
            return result

    return _heuristic_feasibility(user_goal, probes, file_paths, organism, all_pipelines)


# ===========================================================================
# File probing — reads content, not names
# ===========================================================================

def _probe_files(file_paths: list[str]) -> list[dict[str, Any]]:
    """Read the beginning of each file to gather format evidence.

    Probes up to _MAX_PROBE_FILES files in full; beyond that records only
    the path and extension so the LLM/heuristic still sees the full inventory.
    """
    probes = []
    for i, path in enumerate(file_paths):
        if i < _MAX_PROBE_FILES:
            probes.append(_probe_one(path))
        else:
            probes.append({
                "path": path,
                "name": Path(path).name,
                "accessible": False,
                "is_cloud": _is_cloud_path(path),
                "skipped": True,
                "format_hints": [f"not probed (>{_MAX_PROBE_FILES} files — extension: {Path(path).suffix})"],
            })
    return probes


def _probe_one(path: str) -> dict[str, Any]:
    base: dict[str, Any] = {
        "path": path,
        "name": Path(path).name,
        "accessible": False,
        "is_cloud": _is_cloud_path(path),
        "is_binary": False,
        "compressed": False,
        "first_lines": None,
        "magic_description": None,
        "format_hints": [],
        "size_bytes": None,
    }

    if base["is_cloud"]:
        scheme = path.split("://")[0] if "://" in path else "cloud"
        base["format_hints"] = [f"{scheme}:// path — content not readable without credentials"]
        return base

    p = Path(path)
    if not p.exists():
        base["format_hints"] = ["file not found at this path — name/extension only"]
        return base

    base["accessible"] = True
    try:
        base["size_bytes"] = p.stat().st_size
    except OSError:
        pass

    name_lower = p.name.lower()

    # gzip-compressed (most nf-core FASTQ files are .gz)
    if name_lower.endswith((".gz", ".bgz")):
        base["compressed"] = True
        try:
            with gzip.open(p, "rt", errors="replace") as fh:
                lines = [fh.readline() for _ in range(_MAX_PROBE_LINES)]
            base["first_lines"] = [ln.rstrip("\n") for ln in lines if ln]
            return base
        except Exception as exc:
            base["format_hints"] = [f"gzip decompression failed: {exc}"]
            return base

    # BAM / CRAM / other binary
    try:
        with open(p, "rb") as fh:
            magic = fh.read(8)
        for signature, description in _MAGIC:
            if magic[: len(signature)] == signature:
                base["is_binary"] = True
                base["magic_description"] = description
                return base
        # Non-magic binary check: high proportion of null bytes in first 512
        if magic.count(b"\x00") > 2:
            base["is_binary"] = True
            base["format_hints"] = ["binary file (unknown format)"]
            return base
    except OSError as exc:
        base["format_hints"] = [f"could not read: {exc}"]
        return base

    # Plain text
    try:
        with open(p, "r", errors="replace") as fh:
            lines = [fh.readline() for _ in range(_MAX_PROBE_LINES)]
        base["first_lines"] = [ln.rstrip("\n") for ln in lines if ln]
    except OSError as exc:
        base["format_hints"] = [f"could not read text: {exc}"]

    return base


def _is_cloud_path(path: str) -> bool:
    return path.startswith(("s3://", "gs://", "az://", "wasb://", "abfs://"))


# ===========================================================================
# LLM feasibility analysis
# ===========================================================================

async def _llm_feasibility(
    goal: str,
    probes: list[dict[str, Any]],
    organism: str | None,
    all_pipelines: list[dict[str, Any]],
    file_paths: list[str],
) -> dict[str, Any] | None:
    """Run feasibility analysis via LLM reading actual file content.

    Returns a complete check_feasibility result dict, or None on any failure
    so the caller falls back to the heuristic.
    """
    try:
        import anthropic
    except ImportError:
        return None

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None

    pipeline_list = "\n".join(
        f"- {p['name']}: {(p.get('description') or '').strip()}"
        for p in all_pipelines[:60]
    )
    if not pipeline_list:
        pipeline_list = "(pipeline list unavailable)"

    probe_block = _format_probes_for_prompt(probes, len(file_paths))
    organism_line = f"Organism: {organism}" if organism else "Organism: not specified"

    prompt = f"""You are a bioinformatics expert. Analyse the files below and determine
whether the user's goal is feasible with nf-core pipelines.

Goal: {goal}
{organism_line}
Total files provided: {len(file_paths)}

--- FILE PROBES ---
{probe_block}

--- AVAILABLE nf-core PIPELINES ---
{pipeline_list}

Instructions:
1. Identify each file's format from its content (first lines, magic bytes), NOT from its
   name. A file named "sample_abc123.dat" is still identifiable if it contains FASTQ
   records. Mark "novel": true if you genuinely cannot determine the format.
2. Use the file formats + the goal to pick the best-matching nf-core pipeline(s) (up to 3).
3. List what is already present and what is still missing for each pipeline.
4. A "missing" item is only something the pipeline truly requires that is absent —
   do not list things that can be inferred from the files or derived at runtime.
5. If organism is provided, check whether it is covered by nf-core iGenomes
   (human/mouse/rat/zebrafish/drosophila/yeast/arabidopsis/C. elegans are covered).
   If covered, genome is NOT missing for pipelines that accept --genome.

Respond ONLY with valid JSON:
{{
  "file_identifications": [
    {{
      "path": "<original path>",
      "format": "<human-readable format, e.g. FASTQ paired-end R1>",
      "confidence": "high|medium|low",
      "evidence": "<brief reason from the content or name>",
      "novel": false
    }}
  ],
  "data_summary": {{
    "data_type": "<overall description, e.g. paired-end RNA-seq FASTQ reads>",
    "sample_count": <int>,
    "paired_end": <bool>,
    "has_reference_genome": <bool>,
    "has_index_files": <bool>,
    "notes": "<any observations>"
  }},
  "matched_pipelines": [
    {{
      "name": "<exact nf-core pipeline name>",
      "confidence": "high|medium|low",
      "match_reason": "<why this pipeline fits goal + files>",
      "sufficient": ["<what is present and correct>"],
      "missing": ["<what is required but absent>"],
      "ready_to_proceed": <bool>
    }}
  ],
  "overall_feasibility": "yes|yes_with_gaps|unlikely|no",
  "feasibility_summary": "<one or two sentences>"
}}"""

    try:
        import json as _json
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model="claude-opus-4-8",
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = message.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        parsed = _json.loads(raw.strip())
    except Exception as exc:
        logger.warning("LLM feasibility call failed: %s", exc)
        return None

    return _build_result_from_llm(parsed, goal, file_paths, probes)


def _format_probes_for_prompt(probes: list[dict[str, Any]], total: int) -> str:
    lines = []
    for p in probes:
        path = p["path"]
        if p.get("skipped"):
            lines.append(f"[{path}] (not probed — extension only)")
            continue
        if p.get("is_binary"):
            desc = p.get("magic_description") or "binary (unknown)"
            lines.append(f"[{path}] BINARY — {desc}")
            continue
        if not p.get("accessible"):
            hints = "; ".join(p.get("format_hints", ["inaccessible"]))
            lines.append(f"[{path}] NOT READABLE — {hints}")
            continue
        first = p.get("first_lines") or []
        preview = "\n  ".join(first[:8]) if first else "(empty)"
        compressed = " (gzip-decompressed)" if p.get("compressed") else ""
        lines.append(f"[{path}]{compressed}\n  {preview}")

    if total > _MAX_PROBE_FILES:
        lines.append(f"... and {total - _MAX_PROBE_FILES} more files (not probed)")

    return "\n".join(lines)


def _build_result_from_llm(
    parsed: dict[str, Any],
    goal: str,
    file_paths: list[str],
    probes: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Validate and normalise the LLM response into the standard result format."""
    valid_feasibility = {"yes", "yes_with_gaps", "unlikely", "no"}
    valid_confidence = {"high", "medium", "low"}

    if parsed.get("overall_feasibility") not in valid_feasibility:
        return None

    ds_raw = parsed.get("data_summary", {})
    file_ids = parsed.get("file_identifications", [])

    # Derive file_types list from identifications for downstream compatibility
    file_types = _infer_file_types_from_ids(file_ids, probes)

    data_summary = {
        "file_count": len(file_paths),
        "sample_count": max(int(ds_raw.get("sample_count", 0)), 0),
        "data_type": ds_raw.get("data_type", "unknown"),
        "file_types": sorted(file_types),
        "paired_end": bool(ds_raw.get("paired_end", False)),
        "has_reference_genome": bool(ds_raw.get("has_reference_genome", False)),
        "has_index_files": bool(ds_raw.get("has_index_files", False)),
        "file_identifications": file_ids,
        "_primary_types": list(file_types - {"fasta", "gtf", "gff", "index", "other"}),
    }

    matched = []
    for pm in parsed.get("matched_pipelines", [])[:3]:
        name = pm.get("name", "")
        conf = pm.get("confidence", "medium")
        if conf not in valid_confidence:
            conf = "medium"
        missing = pm.get("missing", [])
        matched.append({
            "name": name,
            "confidence": conf,
            "match_reason": pm.get("match_reason", ""),
            "sufficient": pm.get("sufficient", []),
            "missing": missing,
            "ready_to_proceed": bool(pm.get("ready_to_proceed", len(missing) == 0)),
            "next_steps": _build_next_steps(name, missing),
        })

    return {
        "goal": goal,
        "data_summary": data_summary,
        "matched_pipelines": matched,
        "overall_feasibility": parsed["overall_feasibility"],
        "feasibility_summary": parsed.get("feasibility_summary", ""),
        "analysis_method": "llm",
        "review_required": True,
    }


def _infer_file_types_from_ids(
    file_ids: list[dict[str, Any]],
    probes: list[dict[str, Any]],
) -> set[str]:
    """Map LLM-identified format strings to the canonical type names used internally."""
    fmt_map = {
        "fastq": "fastq", "fq": "fastq", "bam": "bam", "cram": "cram",
        "vcf": "vcf", "bcf": "vcf", "bed": "bed", "fasta": "fasta",
        "fa": "fasta", "fna": "fasta", "gtf": "gtf", "gff": "gff",
        "csv": "csv", "tsv": "tsv", "txt": "txt", "bai": "index",
        "crai": "index", "hdf5": "hdf5", "h5": "hdf5", "rds": "rds",
    }
    types: set[str] = set()
    for fi in file_ids:
        fmt = fi.get("format", "").lower()
        for key, canonical in fmt_map.items():
            if key in fmt:
                types.add(canonical)
                break
        else:
            # Fall back to extension probe if LLM format unrecognised
            for p in probes:
                if p["path"] == fi.get("path") and p.get("magic_description"):
                    desc = p["magic_description"].lower()
                    if "bam" in desc:
                        types.add("bam")
                    elif "cram" in desc:
                        types.add("cram")
    return types or {"other"}


# ===========================================================================
# Heuristic feasibility (fallback — extension + name patterns)
# ===========================================================================

# ---------------------------------------------------------------------------
# Static knowledge base — used when LLM is unavailable
# ---------------------------------------------------------------------------

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


def _heuristic_feasibility(
    goal: str,
    probes: list[dict[str, Any]],
    file_paths: list[str],
    organism: str | None,
    all_pipelines: list[dict[str, Any]],
) -> dict[str, Any]:
    data_summary = _analyze_files(file_paths)
    pipeline_scores = _score_pipelines(goal, all_pipelines)
    top_matches = _filter_by_file_compatibility(pipeline_scores, data_summary)

    matched = []
    for name, score, reason in top_matches[:3]:
        assessment = _assess_pipeline(name, data_summary, organism)
        missing = assessment["missing"]
        matched.append({
            "name": name,
            "confidence": _confidence_label(score),
            "match_reason": reason,
            "sufficient": assessment["sufficient"],
            "missing": missing,
            "ready_to_proceed": len(missing) == 0,
            "next_steps": _build_next_steps(name, missing),
        })

    overall = _overall_feasibility(matched, data_summary)

    return {
        "goal": goal,
        "data_summary": data_summary,
        "matched_pipelines": matched,
        "overall_feasibility": overall,
        "feasibility_summary": _OVERALL_LABELS[overall],
        "analysis_method": "heuristic",
        "review_required": True,
    }


# ---------------------------------------------------------------------------
# Heuristic: file analysis
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
        "_primary_types": primary_types,
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
# Heuristic: pipeline scoring
# ---------------------------------------------------------------------------

def _score_pipelines(
    goal: str,
    all_pipelines: list[dict[str, Any]],
) -> list[tuple[str, float, str]]:
    goal_lower = goal.lower()
    scores: list[tuple[str, float, str]] = []

    keyword_scores: dict[str, tuple[float, list[str]]] = {}
    for pipeline_name, keywords in _GOAL_KEYWORDS.items():
        hits = [kw for kw in keywords if kw in goal_lower]
        if hits:
            keyword_scores[pipeline_name] = (len(hits) / len(keywords), hits)

    live_names = {p["name"] for p in all_pipelines}
    for pipeline in all_pipelines:
        name = pipeline["name"]
        base_score, hits = keyword_scores.get(name, (0.0, []))
        desc_lower = (pipeline.get("description") or "").lower()
        topic_lower = " ".join(pipeline.get("topics") or []).lower()
        combined = f"{name} {desc_lower} {topic_lower}"
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
    user_types = set(data_summary["_primary_types"])
    if not user_types:
        return scores
    compatible = [
        (name, score, reason) for name, score, reason in scores
        if not _PIPELINE_ACCEPTS.get(name) or user_types & _PIPELINE_ACCEPTS[name]
    ]
    return compatible if compatible else scores


# ---------------------------------------------------------------------------
# Heuristic: gap analysis
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

    if primary_types:
        dt = data_summary["data_type"]
        if accepted and set(primary_types) & accepted:
            sufficient.append(f"Input file type ({dt}) is compatible with nf-core/{name}")
    if data_summary["sample_count"] > 0:
        sufficient.append(f"{data_summary['sample_count']} sample(s) detected from file naming")
    if data_summary["paired_end"]:
        sufficient.append("Paired-end R1/R2 files are correctly matched by name")
    if data_summary["has_reference_genome"]:
        sufficient.append("A reference genome file (FASTA/GFF/GTF) is present in your file list")
    if data_summary["has_index_files"]:
        sufficient.append("Index files (.bai/.crai) detected alongside alignment files")

    for req_key, gap_msg in _PIPELINE_GAPS.get(name, []):
        if req_key == "genome" and (data_summary["has_reference_genome"] or _organism_has_igenomes(organism)):
            if _organism_has_igenomes(organism):
                sufficient.append(
                    f"Organism '{organism}' is available in nf-core iGenomes — "
                    f"you can use --genome instead of a FASTA file"
                )
            continue
        missing.append(gap_msg)

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
        steps.append(f"Run analyze_pipeline_schema('{name}') to map all parameter choices")
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
# Heuristic: verdict
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
