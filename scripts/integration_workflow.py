"""End-to-end walkthrough of the GenomeOps MCP workflow (faithful plumbing only).

Scenario
--------
An expert wants to run nf-core/rnaseq on 4 human paired-end RNA-seq samples
(2 conditions x 2 replicates). They have realistic FASTQ paths and have already
decided their parameters. GenomeOps just makes running it easy — it does not pick
the pipeline, set parameters, or interpret results.

Network assumptions
-------------------
raw.githubusercontent.com  — accessible (schema fetching)
nf-co.re API               — may be blocked in CI; steps degrade gracefully
api.github.com/search      — accessible (pipeline discovery fallback)

Run with:

    uv run python scripts/integration_workflow.py

Or as pytest:

    uv run pytest scripts/integration_workflow.py -v -s
"""

import asyncio
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Test data — a coherent, realistic scenario
# ---------------------------------------------------------------------------

DATA_DIR = "/data/rnaseq/human_cancer_lines"
FILE_PATHS = [
    f"{DATA_DIR}/MCF7_ctrl_rep1_R1.fastq.gz",
    f"{DATA_DIR}/MCF7_ctrl_rep1_R2.fastq.gz",
    f"{DATA_DIR}/MCF7_ctrl_rep2_R1.fastq.gz",
    f"{DATA_DIR}/MCF7_ctrl_rep2_R2.fastq.gz",
    f"{DATA_DIR}/MCF7_treated_rep1_R1.fastq.gz",
    f"{DATA_DIR}/MCF7_treated_rep1_R2.fastq.gz",
    f"{DATA_DIR}/MCF7_treated_rep2_R1.fastq.gz",
    f"{DATA_DIR}/MCF7_treated_rep2_R2.fastq.gz",
]

PIPELINE = "rnaseq"
VERSION = "3.14.0"          # known stable; release API may be blocked in CI
PROFILE = "docker"
OUTDIR = "/results/MCF7_rnaseq"

# Parameters the EXPERT chose. GenomeOps passes these through verbatim; it never
# suggests, classifies, or alters them.
USER_PARAMS = {
    "--genome": "GRCh38",
    "--aligner": "star_salmon",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def header(step: str, title: str) -> None:
    bar = "─" * 66
    print(f"\n{bar}\n  Step {step}: {title}\n{bar}")


def ok(msg: str) -> None:
    print(f"  ✓  {msg}")


def info(msg: str) -> None:
    print(f"     {msg}")


def warn(msg: str) -> None:
    print(f"  ⚠  {msg}")


def assert_no_error(result: dict, step: str) -> None:
    if result.get("error"):
        raise AssertionError(
            f"{step} returned an error: {result.get('code')} — {result.get('message')}"
        )


# ===========================================================================
# Steps
# ===========================================================================

async def step_list_pipelines() -> str:
    from genomeops_mcp.tools.discovery import list_pipelines
    header("1", "list_pipelines(topic='rna')")
    result = await list_pipelines(topic="rna")
    if result.get("error"):
        warn(f"API unavailable: {result.get('message')} — continuing with known pipeline")
        return PIPELINE
    names = [p["name"] for p in result["pipelines"]]
    info(f"Found {result['total']} pipelines: {', '.join(names[:8])}")
    ok("discovery returned a pipeline list")
    return PIPELINE


async def step_get_pipeline_info(pipeline: str) -> None:
    from genomeops_mcp.tools.discovery import get_pipeline_info
    header("2", f"get_pipeline_info('{pipeline}')")
    result = await get_pipeline_info(pipeline)
    if result.get("error"):
        warn(f"Could not fetch info ({result.get('code')}); known metadata is sufficient")
        return
    info(f"Schema URL:       {result.get('schema_url', '?')}")
    info(f"Input schema URL: {result.get('input_schema_url', '?')}")
    ok("pipeline metadata retrieved")


async def step_get_latest_version(pipeline: str) -> str:
    from genomeops_mcp.tools.versioning import get_latest_version
    header("3", f"get_latest_version('{pipeline}')")
    result = await get_latest_version(pipeline)
    if result.get("error"):
        warn(f"Release API blocked ({result.get('code')}); pinning known stable {VERSION}")
        return VERSION
    version = result.get("recommended") or result.get("stable") or VERSION
    if result.get("is_dev"):
        warn(f"Latest is a dev/pre-release — an agent must stop and ask before using it")
    ok(f"pinned version: {version}")
    return version


async def step_samplesheet(pipeline: str) -> str:
    from genomeops_mcp.tools.samplesheet import generate_samplesheet, validate_samplesheet
    header("4", f"generate_samplesheet + validate_samplesheet ('{pipeline}')")

    gen = await generate_samplesheet(pipeline, FILE_PATHS)
    assert_no_error(gen, "generate_samplesheet")
    csv = gen["samplesheet"]
    info("Generated samplesheet (sample/R1/R2 paired mechanically):")
    for line in csv.splitlines():
        info(f"    {line}")
    assert "unstranded" not in csv, "strandedness must be left blank, not guessed"
    ok("review_required is set: " + str(gen["review_required"]))
    for w in gen["warnings"]:
        warn(w)

    # In a real run the expert now fills in the blank strandedness column.
    # Validating the draft as-is shows the pipeline's own schema flagging the
    # blanks the expert still has to fill — the server makes no decision for them.
    val = await validate_samplesheet(pipeline, csv)
    assert_no_error(val, "validate_samplesheet")
    info(f"valid={val['valid']}, {len(val['errors'])} error(s) against the pipeline's own schema")
    ok("samplesheet validated against the pipeline schema")
    return csv


async def step_environment() -> None:
    from genomeops_mcp.tools.execution import check_execution_environment
    header("5", "check_execution_environment()")
    env = await check_execution_environment(PROFILE)
    info(f"ready={env['ready']}, recommended_profile={env['recommended_profile']}")
    if env["missing"]:
        warn(f"missing: {', '.join(env['missing'])} (install hints provided)")
    ok("environment probed (read-only)")


async def step_launch_command(pipeline: str, version: str) -> None:
    from genomeops_mcp.tools.launch import generate_launch_command
    header("6", "generate_launch_command(..., params=<expert's choices>)")
    result = await generate_launch_command(
        pipeline, version, PROFILE, "samplesheet.csv", OUTDIR, params=USER_PARAMS,
    )
    assert_no_error(result, "generate_launch_command")
    info("Assembled command (params passed through verbatim):")
    for line in result["command"].splitlines():
        info(f"    {line}")
    assert "--genome GRCh38" in result["command"]
    assert "--aligner star_salmon" in result["command"]
    ok(f"review_required is set: {result['review_required']}")


async def step_run_preview(pipeline: str, version: str) -> None:
    from genomeops_mcp.tools.execution import run_pipeline
    header("7", "run_pipeline(..., confirm=False) — preview only, nothing launches")
    result = await run_pipeline(
        pipeline, version, PROFILE, "samplesheet.csv", OUTDIR,
        params=USER_PARAMS, confirm=False,
    )
    if result.get("error"):
        warn(f"Preview blocked by environment: {result.get('code')} — {result.get('message')}")
        return
    assert result.get("requires_confirmation") is True, "must require confirmation"
    info("Preview returned requires_confirmation=True and changed nothing.")
    ok("launch is gated behind confirm=true")


async def main() -> None:
    print("\nGenomeOps — faithful-plumbing workflow walkthrough\n")
    pipeline = await step_list_pipelines()
    await step_get_pipeline_info(pipeline)
    version = await step_get_latest_version(pipeline)
    await step_samplesheet(pipeline)
    await step_environment()
    await step_launch_command(pipeline, version)
    await step_run_preview(pipeline, version)
    print("\nDone. The expert chose the pipeline and parameters; GenomeOps did the plumbing.\n")


@pytest.mark.integration
def test_integration_workflow() -> None:
    asyncio.run(main())


if __name__ == "__main__":
    asyncio.run(main())
