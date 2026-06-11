"""Launch command assembly.

Turns the inputs an expert has already decided on — pipeline, pinned version,
profile, samplesheet, outdir, and any parameters they chose — into a validated
`nextflow run` command string. It assembles; it never proposes. Parameters are
passed through verbatim: the server does not inspect, suggest, or modify them.
"""

import logging
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

    Assembles the full `nextflow run` command string from the inputs you provide.
    Validates only mechanical facts: that the version is pinned (not 'main') and
    that the execution profile is a recognised value. Any parameters you pass are
    appended verbatim as --key value flags — the server does not suggest, infer,
    or alter parameter values; choosing them is your job.

    Always returns review_required=true. Execute only after review — pipelines may
    use significant compute resources and incur cloud costs.

    Args:
        pipeline_name: nf-core pipeline name (e.g. "rnaseq").
        version: Pipeline release version (e.g. "3.14.0").
        profile: Execution profile. One of: docker, singularity, conda, podman,
                 apptainer, or an institutional profile (e.g. "slurm", "aws").
        samplesheet_path: Path to your validated samplesheet CSV.
        outdir: Output directory path (local or cloud, e.g. "s3://bucket/results").
        params: Optional dict of parameters you have chosen. Keys may include the
                leading "--" prefix (e.g. {"--genome": "GRCh38"}). Passed through
                unchanged.

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
