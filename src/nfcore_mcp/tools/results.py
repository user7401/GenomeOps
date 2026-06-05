"""Launch command generation and run results parsing tools."""

import json
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

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
