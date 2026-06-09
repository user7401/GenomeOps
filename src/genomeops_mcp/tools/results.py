"""Launch command generation and run results parsing tools."""

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants for inventory_results
# ---------------------------------------------------------------------------

# Directory names that are always skipped during tree walk.
_SKIP_DIRS = {"work", ".nextflow", "__pycache__", ".git", ".nf-cache"}

# File names (exact) that are always skipped.
_SKIP_NAMES = {
    ".command.sh", ".command.err", ".command.out", ".command.log",
    ".command.run", ".command.begin", ".exitcode", ".nf-cache",
}

# Multi-part / unusual extensions checked before the single-suffix map.
# Mapping: lower-case suffix string → (format, base_role)
_EXT2: dict[str, tuple[str, str]] = {
    ".vcf.gz": ("vcf_gz", "variants"),
    ".vcf.bgz": ("vcf_gz", "variants"),
    ".fastq.gz": ("fastq_gz", "raw_reads"),
    ".fq.gz": ("fastq_gz", "raw_reads"),
    ".fa.gz": ("fasta_gz", "sequence"),
    ".fasta.gz": ("fasta_gz", "sequence"),
    ".tar.gz": ("archive", "archive"),
    ".narrowpeak": ("narrowpeak", "peaks"),
    ".broadpeak": ("broadpeak", "peaks"),
    ".gappedpeak": ("gappedpeak", "peaks"),
    ".gff3": ("gff3", "annotation"),
}

# Single-extension map.
_EXT1: dict[str, tuple[str, str]] = {
    ".bam": ("bam", "alignment"),
    ".cram": ("cram", "alignment"),
    ".bai": ("bai", "alignment_index"),
    ".crai": ("crai", "alignment_index"),
    ".vcf": ("vcf", "variants"),
    ".bcf": ("bcf", "variants"),
    ".tbi": ("tabix_index", "index"),
    ".csi": ("csi_index", "index"),
    ".h5": ("hdf5", "quantification"),
    ".h5ad": ("h5ad", "quantification"),
    ".loom": ("loom", "quantification"),
    ".mtx": ("matrix_market", "quantification"),
    ".bed": ("bed", "intervals"),
    ".gff": ("gff", "annotation"),
    ".gtf": ("gtf", "annotation"),
    ".fa": ("fasta", "sequence"),
    ".fasta": ("fasta", "sequence"),
    ".fastq": ("fastq", "raw_reads"),
    ".fq": ("fastq", "raw_reads"),
    ".tsv": ("tsv", "table"),
    ".csv": ("csv", "table"),
    ".html": ("html", "report"),
    ".pdf": ("pdf", "report"),
    ".json": ("json", "metadata"),
    ".yml": ("yaml", "metadata"),
    ".yaml": ("yaml", "metadata"),
    ".txt": ("text", "log_or_misc"),
    ".log": ("log", "log"),
    ".png": ("image", "figure"),
    ".svg": ("svg", "figure"),
    ".eps": ("eps", "figure"),
}

# Path-pattern rules that refine the base role when a path segment matches.
# Ordered: first match wins.  Applied to the relative path string (lowercased).
_PATH_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"(^|/)pipeline_info/", re.I), "provenance"),
    (re.compile(r"(^|/)multiqc[_/]", re.I), "qc_report"),
    (re.compile(r"(^|/)fastqc[_/]", re.I), "qc_report"),
    (re.compile(r"(^|/)(trimgalore|fastp|trim_galore)[_/]", re.I), "qc_trimming"),
    (re.compile(r"(^|/)(salmon|kallisto|rsem)[_/]", re.I), "transcript_quantification"),
    (re.compile(r"(counts?[_.]|featurecounts|htseq)", re.I), "gene_counts"),
    (re.compile(r"(^|/)(star|hisat2|bowtie2?|bwa|minimap2|bwamem)[_/]", re.I), "alignment"),
    (re.compile(r"(^|/)(mutect2?|strelka|freebayes|deepvariant|gatk|haplotypecaller)[_/]", re.I), "variants"),
    (re.compile(r"(^|/)(macs[23]?|homer|epic2?|seacr)[_/]|/peaks?[_/]", re.I), "peaks"),
    (re.compile(r"(^|/)(cellranger|alevin|starsolo|scrublet|seurat)[_/]", re.I), "single_cell"),
]

# Which roles land in which output bucket.
_PRIMARY_ROLES = {
    "alignment", "variants", "quantification", "transcript_quantification",
    "gene_counts", "peaks", "sequence", "annotation", "single_cell",
    "raw_reads", "intervals",
}
_QC_ROLES = {"qc_report", "qc_trimming"}
_PROVENANCE_ROLES = {"provenance"}
_INDEX_ROLES = {"alignment_index", "index"}

# Max unclassified entries to include in full detail (avoids flooding the output).
_MAX_UNCLASSIFIED = 50

_VALID_PROFILES = {
    "docker", "singularity", "conda", "podman", "apptainer",
    # Common institutional profiles
    "standard", "test", "test_full", "slurm", "sge", "lsf", "pbs",
    "aws", "google", "azure",
}


async def generate_launch_command(
    pipeline_name: str,
    version: str,
    profile: str,
    samplesheet_path: str,
    outdir: str,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate a validated nextflow run command for an nf-core pipeline.

    Assembles the full `nextflow run` command string from the provided inputs.
    Validates that the execution profile is a recognised value. Additional
    parameters are appended as --key value flags.

    Always returns review_required=true. Execute only after human review —
    pipelines may use significant compute resources and incur cloud costs.

    Args:
        pipeline_name: nf-core pipeline name (e.g. "rnaseq").
        version: Pipeline release version (e.g. "3.14.0").
        profile: Execution profile. One of: docker, singularity, conda, podman,
                 apptainer, or an institutional profile (e.g. "slurm", "aws").
        samplesheet_path: Path to the validated samplesheet CSV.
        outdir: Output directory path (local or cloud, e.g. "s3://bucket/results").
        params: Optional dict of additional parameters. Keys should include
                the leading "--" prefix (e.g. {"--genome": "GRCh38"}).

    Returns:
        {
            "command": str,
            "review_required": true,
            "notes": [str]
        }
    """
    logger.info("Tool call: generate_launch_command(%r, %r)", pipeline_name, version)

    if not version or version == "main":
        return {
            "error": True,
            "code": "VERSION_REQUIRED",
            "message": (
                "A specific version tag is required (e.g. '3.14.0'). "
                "Using 'main' for production runs is not recommended. "
                "Check get_pipeline_info for the latest_version."
            ),
        }

    # Validate profile
    known = {p.lower() for p in _VALID_PROFILES}
    if profile.lower() not in known:
        suggestion = _closest(profile.lower(), known)
        msg = f"Profile '{profile}' is not a standard nf-core profile."
        if suggestion:
            msg += f" Did you mean '{suggestion}'?"
        msg += f" Standard profiles: {', '.join(sorted(_VALID_PROFILES))}."
        return {"error": True, "code": "INVALID_PROFILE", "message": msg}

    parts = [
        "nextflow run",
        f"nf-core/{pipeline_name}",
        f"-r {version}",
        f"-profile {profile}",
        f"--input {samplesheet_path}",
        f"--outdir {outdir}",
    ]

    if params:
        for key, value in params.items():
            flag = key if key.startswith("--") else f"--{key}"
            if isinstance(value, bool):
                if value:
                    parts.append(flag)
            else:
                parts.append(f"{flag} {value}")

    command = " \\\n  ".join(parts)
    notes = _build_notes(pipeline_name, profile, samplesheet_path)

    return {
        "command": command,
        "review_required": True,
        "notes": notes,
    }


async def inventory_results(results_dir: str) -> dict[str, Any]:
    """Inventory and classify every output file in a completed pipeline results directory.

    Walks the output tree, skips Nextflow work dirs and noise files, then
    classifies each file by format (bam, vcf, tsv, …) and semantic role
    (alignment, variants, gene_counts, qc_report, provenance, …).  Files are
    grouped into primary_outputs, qc, provenance, indices, and unclassified so
    an agent can immediately see what was produced and what to do with it.

    Format/role classification is structural (extension + path conventions) and
    does not require a container or bioinformatics tool.  When ANTHROPIC_API_KEY
    is set, an LLM pass adds a one-sentence downstream hint per primary output
    and corrects any obvious role misclassifications.

    Args:
        results_dir: Path to the pipeline output directory (local filesystem).

    Returns:
        {
          "results_dir": str,
          "primary_outputs": [{path, format, role, size_mb, downstream?}],
          "qc":             [{path, format, role, size_mb}],
          "provenance":     [{path, format, role, size_mb}],
          "indices":        [{path, format, role, size_mb}],
          "unclassified":   [{path, format, size_mb}],   # capped at 50
          "file_count":     {"total": int, "primary": int, "qc": int,
                             "provenance": int, "indices": int, "unclassified": int},
          "total_size_mb":  float,
          "analysis_method": "structural" | "llm_refined",
          "review_required": true
        }
    """
    logger.info("Tool call: inventory_results(%r)", results_dir)

    base = Path(results_dir)
    if not base.exists():
        return {
            "error": True,
            "code": "DIR_NOT_FOUND",
            "message": f"Results directory '{results_dir}' does not exist.",
        }

    primary: list[dict[str, Any]] = []
    qc: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    indices: list[dict[str, Any]] = []
    unclassified: list[dict[str, Any]] = []
    total_size = 0.0

    for fpath in sorted(base.rglob("*")):
        if not fpath.is_file():
            continue
        if _is_noise(fpath, base):
            continue

        rel = fpath.relative_to(base)
        size_mb = round(fpath.stat().st_size / (1024 * 1024), 3)
        total_size += size_mb
        fmt, role = _classify_file(rel)

        entry: dict[str, Any] = {
            "path": str(rel),
            "format": fmt,
            "size_mb": size_mb,
        }

        if role in _PRIMARY_ROLES:
            entry["role"] = role
            primary.append(entry)
        elif role in _QC_ROLES:
            entry["role"] = role
            qc.append(entry)
        elif role in _PROVENANCE_ROLES:
            entry["role"] = role
            provenance.append(entry)
        elif role in _INDEX_ROLES:
            entry["role"] = role
            indices.append(entry)
        else:
            unclassified.append(entry)

    all_unclassified_count = len(unclassified)
    if len(unclassified) > _MAX_UNCLASSIFIED:
        unclassified = unclassified[:_MAX_UNCLASSIFIED]

    method = "structural"
    primary = await _llm_annotate(primary) or primary
    if any("downstream" in e for e in primary):
        method = "llm_refined"

    return {
        "results_dir": str(base),
        "primary_outputs": primary,
        "qc": qc,
        "provenance": provenance,
        "indices": indices,
        "unclassified": unclassified,
        "file_count": {
            "total": len(primary) + len(qc) + len(provenance) + len(indices) + all_unclassified_count,
            "primary": len(primary),
            "qc": len(qc),
            "provenance": len(provenance),
            "indices": len(indices),
            "unclassified": all_unclassified_count,
        },
        "total_size_mb": round(total_size, 3),
        "analysis_method": method,
        "review_required": True,
    }


# ---------------------------------------------------------------------------
# inventory_results helpers
# ---------------------------------------------------------------------------

def _is_noise(path: Path, base: Path) -> bool:
    """Return True for files that should never appear in the inventory."""
    rel = path.relative_to(base)
    for part in rel.parts[:-1]:  # ancestor dirs
        if part.lower() in _SKIP_DIRS or part.startswith("."):
            return True
    name = path.name
    if name in _SKIP_NAMES or ".command." in name:
        return True
    return False


def _classify_file(rel: Path) -> tuple[str, str]:
    """Return (format, role) for a file given its path relative to results_dir."""
    name_l = rel.name.lower()
    rel_str = str(rel)

    # Resolve format via extension (compound suffixes checked first).
    fmt = "unknown"
    role = "unclassified"
    for ext, (f, r) in _EXT2.items():
        if name_l.endswith(ext):
            fmt, role = f, r
            break
    else:
        suffix = rel.suffix.lower()
        if suffix in _EXT1:
            fmt, role = _EXT1[suffix]

    # Refine role from path context (first matching rule wins).
    # Skip refinement for index files — their role is fixed by extension.
    if role not in _INDEX_ROLES:
        for pattern, path_role in _PATH_RULES:
            if pattern.search(rel_str):
                role = path_role
                break

    return fmt, role


async def _llm_annotate(
    primary: list[dict[str, Any]],
) -> list[dict[str, Any]] | None:
    """Ask Claude to add downstream hints and correct obvious misclassifications.

    Returns an updated list on success; None on any failure (caller keeps the
    structural classification unchanged).
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key or not primary:
        return None
    try:
        import anthropic
    except ImportError:
        return None

    # Keep the prompt compact — list path + format + role, cap at 40 files.
    items = primary[:40]
    lines = [f"- {e['path']} ({e['format']}, role={e['role']})" for e in items]
    prompt = (
        "You are a bioinformatics expert reviewing the output of an nf-core pipeline.\n\n"
        "Below is a list of primary output files with their detected format and role.\n"
        "For each file:\n"
        "  1. Add a 'downstream' field: one sentence describing the most common next "
        "     analysis step for this file type (e.g. 'Load into DESeq2 for differential "
        "     expression analysis' for a gene counts table).\n"
        "  2. Optionally correct the 'role' if it is obviously wrong.\n\n"
        "Files:\n" + "\n".join(lines) + "\n\n"
        "Respond ONLY with valid JSON:\n"
        '{"annotations": [{"path": "<relative path>", "downstream": "<one sentence>", '
        '"role": "<corrected role or same>"}]}'
    )

    try:
        client = anthropic.Anthropic(api_key=api_key)
        msg = client.messages.create(
            model="claude-opus-4-8",
            max_tokens=1500,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = msg.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        parsed = json.loads(raw.strip())
        by_path = {a["path"]: a for a in parsed.get("annotations", []) if "path" in a}

        _VALID_ROLES = _PRIMARY_ROLES | _QC_ROLES | _PROVENANCE_ROLES | _INDEX_ROLES
        updated = []
        for entry in primary:
            ann = by_path.get(entry["path"])
            if ann:
                e = dict(entry)
                if ann.get("downstream"):
                    e["downstream"] = ann["downstream"]
                if ann.get("role") in _VALID_ROLES:
                    e["role"] = ann["role"]
                updated.append(e)
            else:
                updated.append(entry)
        return updated
    except Exception as exc:  # noqa: BLE001
        logger.warning("LLM annotation for inventory_results failed: %s", exc)
        return None


async def parse_run_summary(results_dir: str) -> dict[str, Any]:
    """Parse a completed nf-core pipeline run and return a structured summary.

    Reads MultiQC general statistics (if present) and the Nextflow execution
    report to produce per-sample QC metrics, flagged samples, and overall
    run status. Returns structured JSON — not raw file contents.

    Use this after a pipeline run completes to assess data quality before
    downstream analysis.

    Args:
        results_dir: Path to the pipeline output directory (local filesystem).

    Returns:
        {
            "run_status": "success" | "failed" | "unknown",
            "samples": [{"name": str, "metrics": dict, "flagged": bool, "flag_reason": str}],
            "flagged_samples": [str],
            "summary_available": bool,
            "notes": [str]
        }
    """
    logger.info("Tool call: parse_run_summary(%r)", results_dir)

    base = Path(results_dir)
    if not base.exists():
        return {
            "error": True,
            "code": "DIR_NOT_FOUND",
            "message": f"Results directory '{results_dir}' does not exist.",
        }

    result: dict[str, Any] = {
        "run_status": "unknown",
        "samples": [],
        "flagged_samples": [],
        "summary_available": False,
        "notes": [],
    }

    # Parse Nextflow execution report for run status
    run_status = _parse_execution_status(base)
    if run_status:
        result["run_status"] = run_status

    # Parse MultiQC general stats
    multiqc_path = base / "multiqc" / "multiqc_data" / "multiqc_general_stats.json"
    if not multiqc_path.exists():
        # Some pipelines put it at a slightly different path
        for candidate in base.rglob("multiqc_general_stats.json"):
            multiqc_path = candidate
            break

    if multiqc_path.exists():
        result["summary_available"] = True
        samples, flagged = _parse_multiqc_stats(multiqc_path)
        result["samples"] = samples
        result["flagged_samples"] = flagged
    else:
        result["notes"].append(
            "MultiQC general stats not found. "
            "Run may not have completed or pipeline does not produce MultiQC output."
        )

    # Parse pipeline_info for execution metrics
    pipeline_info_dir = base / "pipeline_info"
    if pipeline_info_dir.exists():
        result["notes"].extend(_parse_pipeline_info(pipeline_info_dir))

    return result


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _closest(name: str, options: set[str]) -> str | None:
    import difflib
    matches = difflib.get_close_matches(name, options, n=1, cutoff=0.6)
    return matches[0] if matches else None


def _build_notes(pipeline_name: str, profile: str, samplesheet_path: str) -> list[str]:
    notes = []
    if profile == "docker":
        notes.append("Ensure Docker daemon is running before executing.")
    elif profile == "singularity":
        notes.append("Ensure Singularity/Apptainer is installed and configured.")
    elif profile == "conda":
        notes.append("Conda environments will be created on first run — may take time.")

    if samplesheet_path.startswith("s3://") or samplesheet_path.startswith("gs://"):
        notes.append("Cloud input detected — ensure credentials are configured (AWS/GCP).")

    notes.append(
        f"Estimated runtime varies by sample count and compute resources. "
        f"See https://nf-co.re/{pipeline_name}/docs/usage for benchmarks."
    )
    return notes


def _parse_execution_status(base: Path) -> str | None:
    pipeline_info = base / "pipeline_info"
    if not pipeline_info.exists():
        return None

    for html_file in pipeline_info.glob("execution_report*.html"):
        try:
            content = html_file.read_text(errors="ignore")
            if "Workflow complete" in content or "succeeded" in content.lower():
                return "success"
            if "failed" in content.lower() or "error" in content.lower():
                return "failed"
        except Exception:
            pass

    for trace_file in pipeline_info.glob("execution_trace*.txt"):
        try:
            content = trace_file.read_text(errors="ignore")
            if re.search(r"\bFAILED\b", content):
                return "failed"
            if re.search(r"\bCOMPLETED\b", content):
                return "success"
        except Exception:
            pass

    return None


def _parse_multiqc_stats(stats_path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    try:
        raw: dict[str, Any] = json.loads(stats_path.read_text())
    except Exception:
        return [], []

    samples = []
    flagged = []

    for sample_name, metrics in raw.items():
        flagged_reasons = []

        # Common QC thresholds
        pct_dup = _extract_metric(metrics, "pct_duplication", "FastQC_mqc-generalstats-fastqc-percent_duplicates")
        pct_aligned = _extract_metric(metrics, "pct_aligned", "STAR_mqc-generalstats-star-uniquely_mapped_percent")
        total_reads = _extract_metric(metrics, "total_reads", "FastQC_mqc-generalstats-fastqc-total_sequences")

        if pct_dup is not None and float(pct_dup) > 80:
            flagged_reasons.append(f"High duplication rate: {pct_dup:.1f}%")
        if pct_aligned is not None and float(pct_aligned) < 50:
            flagged_reasons.append(f"Low alignment rate: {pct_aligned:.1f}%")
        if total_reads is not None and int(total_reads) < 1_000_000:
            flagged_reasons.append(f"Low read count: {int(total_reads):,}")

        is_flagged = len(flagged_reasons) > 0
        if is_flagged:
            flagged.append(sample_name)

        samples.append({
            "name": sample_name,
            "metrics": {k: v for k, v in metrics.items()},
            "flagged": is_flagged,
            "flag_reason": "; ".join(flagged_reasons) if flagged_reasons else "",
        })

    return samples, flagged


def _extract_metric(metrics: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        if key in metrics:
            try:
                return float(metrics[key])
            except (TypeError, ValueError):
                pass
    return None


def _parse_pipeline_info(info_dir: Path) -> list[str]:
    notes = []
    for software_file in info_dir.glob("software_versions*.yml"):
        notes.append(f"Software versions recorded in: {software_file.name}")
        break
    for params_file in info_dir.glob("params*.json"):
        notes.append(f"Run parameters recorded in: {params_file.name}")
        break
    return notes
