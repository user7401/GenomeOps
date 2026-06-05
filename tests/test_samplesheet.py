"""Tests for samplesheet tools: schema, validation, and generation."""

import pytest
from nfcore_mcp.tools.samplesheet import (
    get_samplesheet_schema,
    validate_samplesheet,
    generate_samplesheet,
)
from nfcore_mcp.cache import cache_clear


@pytest.fixture(autouse=True)
def clear_cache():
    cache_clear()
    yield
    cache_clear()


VALID_SAMPLESHEET = """\
sample,fastq_1,fastq_2,strandedness
ctrl_rep1,/data/ctrl_rep1_R1.fastq.gz,/data/ctrl_rep1_R2.fastq.gz,forward
ctrl_rep2,/data/ctrl_rep2_R1.fastq.gz,/data/ctrl_rep2_R2.fastq.gz,forward
treat_rep1,/data/treat_rep1_R1.fastq.gz,/data/treat_rep1_R2.fastq.gz,reverse
"""

SAMPLESHEET_WRONG_STRANDEDNESS = """\
sample,fastq_1,fastq_2,strandedness
ctrl_rep1,/data/ctrl_rep1_R1.fastq.gz,/data/ctrl_rep1_R2.fastq.gz,fwd
"""

SAMPLESHEET_MISSING_REQUIRED_COLUMN = """\
fastq_1,fastq_2,strandedness
/data/ctrl_rep1_R1.fastq.gz,/data/ctrl_rep1_R2.fastq.gz,forward
"""

SAMPLESHEET_EMPTY_REQUIRED_FIELD = """\
sample,fastq_1,fastq_2,strandedness
ctrl_rep1,,/data/ctrl_rep1_R2.fastq.gz,forward
"""


# ---------------------------------------------------------------------------
# get_samplesheet_schema
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_samplesheet_schema_returns_fields(mock_github_schemas):
    result = await get_samplesheet_schema("rnaseq")
    assert "fields" in result
    assert result["pipeline"] == "rnaseq"
    assert len(result["fields"]) > 0


@pytest.mark.asyncio
async def test_get_samplesheet_schema_field_structure(mock_github_schemas):
    result = await get_samplesheet_schema("rnaseq")
    strandedness = next(f for f in result["fields"] if f["name"] == "strandedness")
    assert strandedness["required"] is True
    assert "forward" in strandedness["allowed_values"]
    assert "reverse" in strandedness["allowed_values"]
    assert "unstranded" in strandedness["allowed_values"]


@pytest.mark.asyncio
async def test_get_samplesheet_schema_not_found(mock_github_schemas):
    result = await get_samplesheet_schema("notexist")
    assert result.get("error") is True
    assert result["code"] == "SCHEMA_NOT_FOUND"


# ---------------------------------------------------------------------------
# validate_samplesheet
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_validate_samplesheet_valid_passes(mock_github_schemas):
    result = await validate_samplesheet("rnaseq", VALID_SAMPLESHEET)
    assert result["valid"] is True
    assert result["errors"] == []


@pytest.mark.asyncio
async def test_validate_samplesheet_wrong_strandedness(mock_github_schemas):
    result = await validate_samplesheet("rnaseq", SAMPLESHEET_WRONG_STRANDEDNESS)
    assert result["valid"] is False
    errors = result["errors"]
    assert len(errors) >= 1

    strand_error = next((e for e in errors if e["column"] == "strandedness"), None)
    assert strand_error is not None
    # Error message must be plain English, not raw jsonschema output
    assert "fwd" in strand_error["message"]
    assert "forward" in strand_error["message"].lower() or "allowed" in strand_error["message"].lower()
    assert strand_error["fix_suggestion"] != ""


@pytest.mark.asyncio
async def test_validate_samplesheet_missing_required_column(mock_github_schemas):
    result = await validate_samplesheet("rnaseq", SAMPLESHEET_MISSING_REQUIRED_COLUMN)
    assert result["valid"] is False
    errors = result["errors"]
    col_names = [e["column"] for e in errors]
    assert "sample" in col_names


@pytest.mark.asyncio
async def test_validate_samplesheet_empty_required_field(mock_github_schemas):
    result = await validate_samplesheet("rnaseq", SAMPLESHEET_EMPTY_REQUIRED_FIELD)
    assert result["valid"] is False
    errors = result["errors"]
    fastq_error = next((e for e in errors if e["column"] == "fastq_1"), None)
    assert fastq_error is not None
    assert fastq_error["row"] == 2


@pytest.mark.asyncio
async def test_validate_samplesheet_error_has_required_fields(mock_github_schemas):
    result = await validate_samplesheet("rnaseq", SAMPLESHEET_WRONG_STRANDEDNESS)
    for error in result["errors"]:
        assert "row" in error
        assert "column" in error
        assert "message" in error
        assert "fix_suggestion" in error


# ---------------------------------------------------------------------------
# generate_samplesheet
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_generate_samplesheet_pairs_r1_r2(mock_github_schemas):
    files = [
        "/data/sample_A_R1_001.fastq.gz",
        "/data/sample_A_R2_001.fastq.gz",
        "/data/sample_B_R1_001.fastq.gz",
        "/data/sample_B_R2_001.fastq.gz",
    ]
    result = await generate_samplesheet("rnaseq", files)
    assert "samplesheet" in result
    csv_content = result["samplesheet"]
    # Both samples should appear
    assert "sample_A" in csv_content
    assert "sample_B" in csv_content
    # R1 paths should appear
    assert "sample_A_R1_001.fastq.gz" in csv_content
    assert "sample_B_R1_001.fastq.gz" in csv_content


@pytest.mark.asyncio
async def test_generate_samplesheet_review_required(mock_github_schemas):
    files = ["/data/sample_A_R1.fastq.gz", "/data/sample_A_R2.fastq.gz"]
    result = await generate_samplesheet("rnaseq", files)
    assert result["review_required"] is True
    assert "review_hint" in result


@pytest.mark.asyncio
async def test_generate_samplesheet_strandedness_warning(mock_github_schemas):
    files = ["/data/sample_A_R1.fastq.gz"]
    result = await generate_samplesheet("rnaseq", files)
    warnings_text = " ".join(result["warnings"]).lower()
    assert "strandedness" in warnings_text


@pytest.mark.asyncio
async def test_generate_samplesheet_unpaired_warning(mock_github_schemas):
    files = ["/data/orphan_R1.fastq.gz"]  # no R2
    result = await generate_samplesheet("rnaseq", files)
    warnings_text = " ".join(result["warnings"]).lower()
    assert "r2" in warnings_text or "pair" in warnings_text


@pytest.mark.asyncio
async def test_generate_samplesheet_has_header(mock_github_schemas):
    files = ["/data/ctrl_R1_001.fastq.gz", "/data/ctrl_R2_001.fastq.gz"]
    result = await generate_samplesheet("rnaseq", files)
    lines = result["samplesheet"].strip().split("\n")
    assert len(lines) >= 2
    header = lines[0]
    assert "sample" in header
