"""End-to-end integration test of the GenomeOps MCP workflow.

Scenario
--------
A researcher wants to run nf-core/rnaseq on 4 human paired-end RNA-seq
samples (2 conditions × 2 replicates). They have realistic FASTQ file
paths but haven't run anything yet.

Network assumptions
-------------------
raw.githubusercontent.com  — accessible (all schema fetching, samplesheet schema)
nf-co.re API               — blocked in this CI environment
api.github.com/repos/*     — 403 in this environment (releases endpoint)
api.github.com/search      — accessible (GitHub search API)

Each step prints what it tested, what the result was, and what a real
agent would do next. Run with:

    uv run python scripts/integration_workflow.py

Or as pytest:

    uv run pytest scripts/integration_workflow.py -v -s
"""

import asyncio
import json
import textwrap
from pathlib import Path
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Test data — realistic, coherent scenario
# ---------------------------------------------------------------------------

# 4 paired-end RNA-seq samples: 2 conditions × 2 reps
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

PIPELINE   = "rnaseq"
VERSION    = "3.14.0"          # known stable; release API blocked in CI env
PROFILE    = "docker"
OUTDIR     = "/results/MCF7_rnaseq"
EXPERIMENT = (
    "Paired-end bulk RNA-seq of MCF7 human breast cancer cell line. "
    "Two conditions: vehicle control vs. estradiol treatment, 48h. "
    "2 biological replicates each. Library prep: TruSeq Stranded mRNA. "
    "Sequenced on NovaSeq 6000, 150bp paired-end reads. "
    "Goal: differential expression between treated and control."
)
DATA_SUMMARY = {
    "paired_end": True,
    "has_reference_genome": True,
    "organism": "Homo sapiens",
    "sample_count": 4,
    "estimated_reads_per_sample_millions": 50,
}
USER_CHOICES = {
    "--genome": "GRCh38",
    "--aligner": "star_salmon",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def header(step: str, title: str) -> None:
    bar = "─" * 66
    print(f"\n{bar}")
    print(f"  Step {step}: {title}")
    print(bar)


def ok(msg: str) -> None:
    print(f"  ✓  {msg}")


def info(msg: str) -> None:
    print(f"     {msg}")


def warn(msg: str) -> None:
    print(f"  ⚠  {msg}")


def fail(msg: str) -> None:
    print(f"  ✗  {msg}")


def assert_no_error(result: dict, step: str) -> None:
    if result.get("error"):
        raise AssertionError(
            f"{step} returned an error: {result.get('code')} — {result.get('message')}"
        )


# ===========================================================================
# Step 1 — List pipelines (GitHub search API)
# ===========================================================================

async def step_list_pipelines() -> str:
    from genomeops_mcp.tools.discovery import list_pipelines
    header("1", "list_pipelines(topic='RNA-seq')")

    result = await list_pipelines(topic="RNA-seq")

    if result.get("error"):
        warn(f"API unavailable: {result.get('message')} — will continue with known pipeline")
        return PIPELINE

    info(f"Total RNA-seq pipelines found: {result['total']}")
    names = [p["name"] for p in result["pipelines"]]
    info(f"Pipelines: {', '.join(names[:8])}{'…' if len(names) > 8 else ''}")
    assert any(p["name"] == "rnaseq" for p in result["pipelines"]), \
        "rnaseq must appear in RNA-seq topic results"
    ok("rnaseq is listed under RNA-seq topic")
    ok("result structure has 'pipelines' list and 'total' count")
    return "rnaseq"


# ===========================================================================
# Step 2 — Get pipeline info (real API or partial)
# ===========================================================================

async def step_get_pipeline_info(pipeline: str) -> dict[str, Any]:
    from genomeops_mcp.tools.discovery import get_pipeline_info
    header("2", f"get_pipeline_info('{pipeline}')")

    result = await get_pipeline_info(pipeline)

    if result.get("error"):
        warn(f"Could not fetch pipeline info via API ({result.get('code')}), using known metadata")
        return {
            "name": pipeline,
            "schema_url": f"https://raw.githubusercontent.com/nf-core/{pipeline}/{VERSION}/nextflow_schema.json",
            "input_schema_url": f"https://raw.githubusercontent.com/nf-core/{pipeline}/{VERSION}/assets/schema_input.json",
            "supported_profiles": ["docker", "singularity", "conda"],
            "required_params": ["input", "outdir"],
            "latest_version": VERSION,
        }

    info(f"Schema URL:       {result.get('schema_url','?')}")
    info(f"Input schema URL: {result.get('input_schema_url','?')}")
    info(f"Profiles:         {result.get('supported_profiles', [])}")
    info(f"Required params:  {result.get('required_params', [])}")
    assert result.get("schema_url"), "schema_url must be present"
    assert "input" in (result.get("required_params") or []), \
        "'input' must be listed as a required param"
    ok("schema and input_schema URLs are present")
    ok("'input' is in required_params")
    return result


# ===========================================================================
# Step 3 — Get samplesheet schema (real network call)
# ===========================================================================

async def step_get_samplesheet_schema() -> dict[str, Any]:
    from genomeops_mcp.tools.samplesheet import get_samplesheet_schema
    header("3", f"get_samplesheet_schema('{PIPELINE}', '{VERSION}')")

    result = await get_samplesheet_schema(PIPELINE, VERSION)
    assert_no_error(result, "get_samplesheet_schema")

    fields = result.get("fields", [])
    info(f"Samplesheet fields ({len(fields)}): {[f['name'] for f in fields]}")
    required = [f["name"] for f in fields if f.get("required")]
    info(f"Required: {required}")

    field_names = {f["name"] for f in fields}
    assert "sample" in field_names, "samplesheet must have a 'sample' column"
    assert "fastq_1" in field_names, "samplesheet must have a 'fastq_1' column"
    ok(f"{len(fields)} columns defined, 'sample' and 'fastq_1' present")

    strandedness_field = next((f for f in fields if f["name"] == "strandedness"), None)
    if strandedness_field:
        allowed = strandedness_field.get("allowed_values", [])
        info(f"strandedness allowed values: {allowed}")
        assert "unstranded" in allowed, "unstranded must be a valid strandedness"
        ok("strandedness field has expected allowed values")

    return result


# ===========================================================================
# Step 4 — Generate samplesheet (real schema + realistic paths)
# ===========================================================================

async def step_generate_samplesheet() -> str:
    from genomeops_mcp.tools.samplesheet import generate_samplesheet
    header("4", f"generate_samplesheet('{PIPELINE}', {len(FILE_PATHS)} files)")

    result = await generate_samplesheet(PIPELINE, FILE_PATHS, VERSION)
    assert_no_error(result, "generate_samplesheet")

    csv = result["samplesheet"]
    lines = [l for l in csv.strip().splitlines() if l]
    header_cols = [c.strip() for c in lines[0].split(",")]
    data_rows = lines[1:]

    info(f"CSV columns: {header_cols}")
    info(f"Data rows:   {len(data_rows)}")
    info("First 3 rows:")
    for row in data_rows[:3]:
        info(f"  {row}")
    if result.get("warnings"):
        for w in result["warnings"]:
            warn(w)

    assert "sample" in header_cols, "samplesheet must have 'sample' column"
    assert "fastq_1" in header_cols, "samplesheet must have 'fastq_1' column"
    assert len(data_rows) == 4, f"expected 4 samples (R1/R2 paired), got {len(data_rows)}"
    assert result["review_required"] is True, "review_required must always be True"

    # Every R1 file must appear in the CSV
    for fp in FILE_PATHS:
        if "R1" in fp:
            sample = Path(fp).name.replace("_R1.fastq.gz", "")
            assert any(sample in row for row in data_rows), \
                f"sample derived from {fp} not found in CSV"

    ok(f"4 paired samples correctly paired and written to CSV")
    ok("review_required=True (strandedness must be verified by human)")
    return csv


# ===========================================================================
# Step 5 — Validate samplesheet (round-trip: real schema validation)
# ===========================================================================

async def step_validate_samplesheet(csv: str) -> None:
    from genomeops_mcp.tools.samplesheet import validate_samplesheet
    header("5", f"validate_samplesheet('{PIPELINE}', generated_csv)")

    result = await validate_samplesheet(PIPELINE, csv, VERSION)
    assert_no_error(result, "validate_samplesheet")

    info(f"Valid:   {result.get('valid')}")
    info(f"Errors:  {result.get('errors', [])}")
    info(f"Row count: {result.get('row_count', '?')}")

    assert result.get("valid") is True, \
        f"Generated samplesheet failed validation: {result.get('errors')}"
    ok("Generated samplesheet passes validation against the real rnaseq schema")


# ===========================================================================
# Step 6 — Analyze pipeline schema (real nextflow_schema.json, heuristic path)
# ===========================================================================

async def step_analyze_pipeline_schema() -> dict[str, Any]:
    from genomeops_mcp.tools.configuration import analyze_pipeline_schema
    header("6", f"analyze_pipeline_schema('{PIPELINE}', '{VERSION}') [no LLM key → heuristic]")

    result = await analyze_pipeline_schema(PIPELINE, VERSION)
    assert_no_error(result, "analyze_pipeline_schema")

    info(f"Classification method: {result.get('classification_method')}")
    info(f"Path count estimate:   {result.get('path_count_estimate')}")
    dps = result.get("decision_points", [])
    info(f"Decision points: {len(dps)}")
    for dp in dps[:5]:
        info(f"  {dp['param']} ({dp['kind']}) options={dp.get('options', [])[:3]}")

    tiers = result.get("tiers", {})
    for tier, td in tiers.items():
        count = len(td.get("params", []))
        info(f"  tier {tier}: {count} params")

    assert result.get("classification_method") in ("heuristic", "llm", "llm-cached")
    assert len(dps) >= 2, "rnaseq must expose at least 2 decision points"
    # aligner choice must be a decision point
    dp_params = {dp["param"] for dp in dps}
    assert "--aligner" in dp_params, f"--aligner must be a decision point, got {dp_params}"
    ok(f"--aligner surfaced as a decision point (options={next(d for d in dps if d['param']=='--aligner')['options']})")
    ok(f"{len(dps)} decision points identified")
    return result


# ===========================================================================
# Step 7 — Configure parameters (real schema, user choices + data_summary)
# ===========================================================================

async def step_configure_parameters() -> dict[str, Any]:
    from genomeops_mcp.tools.configuration import configure_parameters
    header("7", f"configure_parameters('{PIPELINE}', version='{VERSION}')")
    info(f"user_choices:  {USER_CHOICES}")
    info(f"data_summary:  {DATA_SUMMARY}")

    result = await configure_parameters(
        PIPELINE,
        version=VERSION,
        experiment_description=EXPERIMENT,
        data_summary=DATA_SUMMARY,
        user_choices=USER_CHOICES,
    )
    assert_no_error(result, "configure_parameters")

    resolved   = result["resolved_params"]
    sources    = result["param_sources"]
    auto_d     = result["auto_derived"]
    questions  = result["needs_user_input"]
    defaults   = result["using_defaults"]

    info(f"Resolved:       {len(resolved)} params")
    info(f"  User pinned:  {[k for k,v in sources.items() if v=='user']}")
    info(f"  Auto-derived: {[d['param'] for d in auto_d]}")
    info(f"Needs input:    {len(questions)} params")
    info(f"Defaults:       {len(defaults)} params")
    info("Open questions (first 8):")
    for q in questions[:8]:
        info(f"  [{q['tier']:16s}] {q['param']}: {q['question'][:70]}")

    # User choices must be honoured
    assert resolved.get("--genome") == "GRCh38", "--genome should be resolved from user_choices"
    assert resolved.get("--aligner") == "star_salmon", "--aligner should be resolved from user_choices"
    assert sources.get("--genome") == "user"
    assert sources.get("--aligner") == "user"
    ok("User choices (--genome, --aligner) honoured and sources tagged 'user'")

    # Data summary must drive single_end derivation
    single_end_derived = next((d for d in auto_d if "single_end" in d["param"]), None)
    if single_end_derived:
        assert single_end_derived["value"] is False, \
            "paired_end=True in data_summary must derive single_end=False"
        ok(f"single_end auto-derived: {single_end_derived['value']} ({single_end_derived.get('evidence','')})")
    else:
        warn("single_end not auto-derived (may be absent from this schema version)")

    # Reference inputs must be surfaced (gap-#3 fix)
    asked_params = {q["param"] for q in questions}
    assert "--input" in asked_params, "--input must be in needs_user_input"
    assert "--outdir" in asked_params, "--outdir must be in needs_user_input"
    ok("--input and --outdir correctly surfaced (not yet provided)")

    # Genome/fasta: genome was pinned via user_choices, fasta may or may not surface
    if "--fasta" in asked_params:
        ok("--fasta surfaced as a question (GRCh38 genome chosen, custom fasta still optional)")

    # No boolean toggle should appear as EXPERT_REQUIRED
    expert_booleans = [q for q in questions if q.get("tier") == "expert_required"
                       and q["param"].startswith(("--skip_", "--save_", "--with_"))]
    assert not expert_booleans, f"Boolean toggles wrongly expert: {[q['param'] for q in expert_booleans]}"
    ok("No boolean toggle escalated to expert_required")

    # Review required
    assert result["review_required"] is True
    ok(f"review_required=True — {len(questions)} open questions remain for the human")
    return result


# ===========================================================================
# Step 8 — Generate launch command (pure logic, validate structure)
# ===========================================================================

async def step_generate_launch_command(configure_result: dict[str, Any]) -> str:
    from genomeops_mcp.tools.results import generate_launch_command
    header("8", "generate_launch_command(...) — assemble the final nextflow run command")

    # Build params from resolved + an explicit input/outdir supplied by the user
    resolved = dict(configure_result["resolved_params"])
    # Simulate: user has now supplied the two remaining required inputs
    resolved["--input"]  = "/data/rnaseq/samplesheet.csv"
    resolved["--outdir"] = OUTDIR
    # Drop the open questions (they aren't resolved yet — don't pass them)
    asked = {q["param"] for q in configure_result["needs_user_input"]}
    final_params = {k: v for k, v in resolved.items() if k not in asked or k in ("--input", "--outdir")}

    result = await generate_launch_command(
        pipeline_name=PIPELINE,
        version=VERSION,
        profile=PROFILE,
        samplesheet_path="/data/rnaseq/samplesheet.csv",
        outdir=OUTDIR,
        params={k: v for k, v in final_params.items()
                if k not in ("--input", "--outdir")},
    )
    assert_no_error(result, "generate_launch_command")

    command = result["command"]
    print()
    print("  Generated command:")
    for line in command.splitlines():
        print(f"    {line}")

    assert "nextflow run" in command
    assert f"nf-core/{PIPELINE}" in command
    assert f"-r {VERSION}" in command
    assert f"-profile {PROFILE}" in command
    assert "--input" in command
    assert "--outdir" in command
    assert result["review_required"] is True
    ok("Command includes: nextflow run, nf-core/rnaseq, -r, -profile, --input, --outdir")

    if "--genome GRCh38" in command:
        ok("--genome GRCh38 (user choice) present in command")
    if "--aligner star_salmon" in command:
        ok("--aligner star_salmon (user choice) present in command")

    ok("review_required=True — human must review before executing")
    return command


# ===========================================================================
# Step 9 — Coherence: cross-check the full chain
# ===========================================================================

def step_coherence(samplesheet_csv: str, command: str, configure_result: dict[str, Any]) -> None:
    header("9", "Cross-chain coherence check")

    # Samplesheet column count matches schema
    lines = [l for l in samplesheet_csv.strip().splitlines() if l]
    header_cols = [c.strip() for c in lines[0].split(",")]
    data_rows = lines[1:]
    for i, row in enumerate(data_rows, 1):
        assert len(row.split(",")) == len(header_cols), \
            f"Row {i} column count {len(row.split(','))} != header {len(header_cols)}"
    ok(f"All {len(data_rows)} CSV rows have correct column count ({len(header_cols)})")

    # Version in command matches the version we used for schema fetching
    assert f"-r {VERSION}" in command, "command version must match schema version"
    ok(f"Version in command matches schema version ({VERSION})")

    # User choices flow from configure_parameters into the command
    resolved = configure_result["resolved_params"]
    user_params = {k: v for k, v in resolved.items()
                   if configure_result["param_sources"].get(k) == "user"}
    for param, value in user_params.items():
        expected = f"{param} {value}"
        assert expected in command, f"User choice {expected} not found in command"
    ok(f"All user choices ({list(user_params.keys())}) flow into the command")

    # No param appears in both resolved and needs_user_input
    asked = {q["param"] for q in configure_result["needs_user_input"]}
    overlap = asked & set(resolved.keys())
    overlap -= {"--input", "--outdir"}  # these are legitimately in both
    assert not overlap, f"Params appear in both resolved and needs_user_input: {overlap}"
    ok("No param double-counted between resolved and needs_user_input")


# ===========================================================================
# Orchestrator
# ===========================================================================

async def run_workflow() -> dict[str, Any]:
    from genomeops_mcp.cache import cache_clear
    cache_clear()

    print("\n" + "═" * 68)
    print("  GenomeOps MCP — Real end-to-end workflow test")
    print("  Pipeline: nf-core/rnaseq  |  Version: 3.14.0")
    print("  Scenario: MCF7 human breast cancer, 4 paired-end RNA-seq samples")
    print("  Network:  raw.githubusercontent.com (schemas) + GitHub search API")
    print("═" * 68)

    _pipeline          = await step_list_pipelines()
    _info              = await step_get_pipeline_info(_pipeline)
    _schema            = await step_get_samplesheet_schema()
    samplesheet_csv    = await step_generate_samplesheet()
    await step_validate_samplesheet(samplesheet_csv)
    _analysis          = await step_analyze_pipeline_schema()
    configure_result   = await step_configure_parameters()
    command            = await step_generate_launch_command(configure_result)
    step_coherence(samplesheet_csv, command, configure_result)

    header("✓", "All steps passed")
    print()
    return {
        "samplesheet_csv": samplesheet_csv,
        "configure_result": configure_result,
        "command": command,
    }


# ---------------------------------------------------------------------------
# pytest entry point
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.integration
async def test_full_rnaseq_workflow():
    """Full rnaseq workflow integration test.

    Makes real HTTP calls to raw.githubusercontent.com and the GitHub search
    API. Validates the complete MCP tool chain from pipeline discovery through
    to launch command assembly against real nf-core schemas.

    Run with:  uv run pytest scripts/integration_workflow.py -v -s -m integration
    """
    await run_workflow()


# ---------------------------------------------------------------------------
# Direct script entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    asyncio.run(run_workflow())
