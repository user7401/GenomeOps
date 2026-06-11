"""Tests for generate_launch_command — faithful command assembly, no opinions."""

import pytest

from genomeops_mcp.tools.launch import generate_launch_command


@pytest.mark.asyncio
async def test_assembles_command_with_pinned_version_and_profile():
    result = await generate_launch_command(
        "rnaseq", "3.14.0", "docker", "samplesheet.csv", "results",
    )
    cmd = result["command"]
    assert "nextflow run" in cmd
    assert "nf-core/rnaseq" in cmd
    assert "-r 3.14.0" in cmd
    assert "-profile docker" in cmd
    assert "--input samplesheet.csv" in cmd
    assert "--outdir results" in cmd
    assert result["review_required"] is True
    assert isinstance(result["notes"], list)


@pytest.mark.asyncio
async def test_params_passed_through_verbatim():
    result = await generate_launch_command(
        "rnaseq", "3.14.0", "docker", "ss.csv", "out",
        params={"--genome": "GRCh38", "aligner": "star_salmon", "save_reference": True},
    )
    cmd = result["command"]
    # Leading "--" added when missing; values left exactly as given.
    assert "--genome GRCh38" in cmd
    assert "--aligner star_salmon" in cmd
    # Boolean true becomes a bare flag.
    assert "--save_reference" in cmd
    assert "--save_reference True" not in cmd


@pytest.mark.asyncio
async def test_false_boolean_is_omitted():
    result = await generate_launch_command(
        "rnaseq", "3.14.0", "docker", "ss.csv", "out",
        params={"--skip_qc": False},
    )
    assert "--skip_qc" not in result["command"]


@pytest.mark.asyncio
async def test_rejects_unpinned_version():
    result = await generate_launch_command("rnaseq", "main", "docker", "ss.csv", "out")
    assert result.get("error") is True
    assert result["code"] == "VERSION_REQUIRED"


@pytest.mark.asyncio
async def test_rejects_unknown_profile_with_suggestion():
    result = await generate_launch_command("rnaseq", "3.14.0", "dcoker", "ss.csv", "out")
    assert result.get("error") is True
    assert result["code"] == "INVALID_PROFILE"
    assert "docker" in result["message"]


@pytest.mark.asyncio
async def test_accepts_institutional_profile():
    result = await generate_launch_command("rnaseq", "3.14.0", "slurm", "ss.csv", "out")
    assert "error" not in result
    assert "-profile slurm" in result["command"]
