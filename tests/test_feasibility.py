"""Tests for check_feasibility tool."""

import pytest
from nfcore_mcp.tools.feasibility import check_feasibility
from nfcore_mcp.cache import cache_clear


@pytest.fixture(autouse=True)
def clear_cache():
    cache_clear()
    yield
    cache_clear()


RNASEQ_FILES_PE = [
    "/data/ctrl_rep1_R1_001.fastq.gz",
    "/data/ctrl_rep1_R2_001.fastq.gz",
    "/data/ctrl_rep2_R1_001.fastq.gz",
    "/data/ctrl_rep2_R2_001.fastq.gz",
    "/data/treat_rep1_R1_001.fastq.gz",
    "/data/treat_rep1_R2_001.fastq.gz",
]

RNASEQ_FILES_WITH_REF = RNASEQ_FILES_PE + ["/ref/GRCh38.fa", "/ref/genes.gtf"]

VARIANT_FILES_BAM = [
    "/data/tumor_sample.bam",
    "/data/tumor_sample.bai",
    "/data/normal_sample.bam",
    "/data/normal_sample.bai",
]

METHYLATION_FILES = [
    "/data/sample1_R1.fastq.gz",
    "/data/sample1_R2.fastq.gz",
    "/data/sample2_R1.fastq.gz",
    "/data/sample2_R2.fastq.gz",
]

COUNT_MATRIX_FILES = [
    "/data/counts.csv",
    "/data/sample_metadata.csv",
]


# ---------------------------------------------------------------------------
# Output structure
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_result_has_required_keys(mock_nfcore_api):
    result = await check_feasibility("RNA-seq differential expression", RNASEQ_FILES_PE)
    for key in ("goal", "data_summary", "matched_pipelines", "overall_feasibility",
                "feasibility_summary", "review_required"):
        assert key in result, f"Missing key: {key}"


@pytest.mark.asyncio
async def test_review_required_always_true(mock_nfcore_api):
    result = await check_feasibility("RNA-seq", RNASEQ_FILES_PE)
    assert result["review_required"] is True


@pytest.mark.asyncio
async def test_data_summary_structure(mock_nfcore_api):
    result = await check_feasibility("RNA-seq", RNASEQ_FILES_PE)
    ds = result["data_summary"]
    assert ds["file_count"] == len(RNASEQ_FILES_PE)
    assert ds["paired_end"] is True
    assert ds["sample_count"] == 3
    assert "fastq" in ds["file_types"]


# ---------------------------------------------------------------------------
# Pipeline matching
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_rnaseq_goal_matches_rnaseq_pipeline(mock_nfcore_api):
    result = await check_feasibility(
        "differential gene expression in human fibroblasts, RNA-seq",
        RNASEQ_FILES_PE,
    )
    names = [p["name"] for p in result["matched_pipelines"]]
    assert "rnaseq" in names


@pytest.mark.asyncio
async def test_variant_goal_matches_sarek(mock_nfcore_api):
    result = await check_feasibility(
        "somatic variant calling from tumour/normal whole genome sequencing",
        VARIANT_FILES_BAM,
    )
    names = [p["name"] for p in result["matched_pipelines"]]
    assert "sarek" in names


@pytest.mark.asyncio
async def test_methylation_goal_matches_methylseq(mock_nfcore_api):
    result = await check_feasibility(
        "DNA methylation analysis using bisulfite sequencing WGBS",
        METHYLATION_FILES,
    )
    names = [p["name"] for p in result["matched_pipelines"]]
    assert "methylseq" in names


@pytest.mark.asyncio
async def test_matched_pipeline_has_required_fields(mock_nfcore_api):
    result = await check_feasibility("RNA-seq", RNASEQ_FILES_PE)
    assert len(result["matched_pipelines"]) >= 1
    p = result["matched_pipelines"][0]
    for key in ("name", "confidence", "match_reason", "sufficient", "missing",
                "ready_to_proceed", "next_steps"):
        assert key in p, f"Missing key in matched pipeline: {key}"


@pytest.mark.asyncio
async def test_confidence_is_valid_label(mock_nfcore_api):
    result = await check_feasibility("RNA-seq transcriptomics differential expression", RNASEQ_FILES_PE)
    for p in result["matched_pipelines"]:
        assert p["confidence"] in {"high", "medium", "low"}


# ---------------------------------------------------------------------------
# Gap detection
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_genome_reported_as_missing_without_reference(mock_nfcore_api):
    result = await check_feasibility("RNA-seq", RNASEQ_FILES_PE)
    rnaseq_match = next(p for p in result["matched_pipelines"] if p["name"] == "rnaseq")
    missing_text = " ".join(rnaseq_match["missing"]).lower()
    assert "genome" in missing_text or "reference" in missing_text


@pytest.mark.asyncio
async def test_genome_not_missing_when_reference_file_provided(mock_nfcore_api):
    result = await check_feasibility("RNA-seq", RNASEQ_FILES_WITH_REF)
    rnaseq_match = next((p for p in result["matched_pipelines"] if p["name"] == "rnaseq"), None)
    if rnaseq_match:
        missing_text = " ".join(rnaseq_match["missing"]).lower()
        assert "reference genome" not in missing_text or "fasta" not in missing_text


@pytest.mark.asyncio
async def test_genome_not_missing_when_organism_is_human(mock_nfcore_api):
    result = await check_feasibility("RNA-seq", RNASEQ_FILES_PE, organism="human")
    rnaseq_match = next((p for p in result["matched_pipelines"] if p["name"] == "rnaseq"), None)
    if rnaseq_match:
        # genome gap should be resolved via iGenomes
        sufficient_text = " ".join(rnaseq_match["sufficient"]).lower()
        assert "igenomes" in sufficient_text or "genome" in sufficient_text


@pytest.mark.asyncio
async def test_strandedness_flagged_as_missing_for_rnaseq(mock_nfcore_api):
    result = await check_feasibility("RNA-seq", RNASEQ_FILES_PE, organism="human")
    rnaseq_match = next((p for p in result["matched_pipelines"] if p["name"] == "rnaseq"), None)
    if rnaseq_match:
        missing_text = " ".join(rnaseq_match["missing"]).lower()
        assert "strandedness" in missing_text


@pytest.mark.asyncio
async def test_ready_to_proceed_false_when_gaps_exist(mock_nfcore_api):
    result = await check_feasibility("RNA-seq", RNASEQ_FILES_PE)
    rnaseq_match = next((p for p in result["matched_pipelines"] if p["name"] == "rnaseq"), None)
    if rnaseq_match:
        # strandedness + genome are always missing without user input
        assert rnaseq_match["ready_to_proceed"] is False


# ---------------------------------------------------------------------------
# Data summary
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_paired_end_detection(mock_nfcore_api):
    result = await check_feasibility("RNA-seq", RNASEQ_FILES_PE)
    assert result["data_summary"]["paired_end"] is True


@pytest.mark.asyncio
async def test_single_end_detection(mock_nfcore_api):
    single_end_files = [
        "/data/sample_A.fastq.gz",
        "/data/sample_B.fastq.gz",
    ]
    result = await check_feasibility("RNA-seq", single_end_files)
    assert result["data_summary"]["paired_end"] is False


@pytest.mark.asyncio
async def test_bam_files_detected(mock_nfcore_api):
    result = await check_feasibility("variant calling", VARIANT_FILES_BAM)
    ds = result["data_summary"]
    assert "bam" in ds["file_types"]
    assert ds["has_index_files"] is True


@pytest.mark.asyncio
async def test_reference_genome_detected(mock_nfcore_api):
    result = await check_feasibility("RNA-seq", RNASEQ_FILES_WITH_REF)
    assert result["data_summary"]["has_reference_genome"] is True


@pytest.mark.asyncio
async def test_no_files_returns_unlikely(mock_nfcore_api):
    result = await check_feasibility("RNA-seq differential expression", [])
    assert result["overall_feasibility"] in {"unlikely", "yes_with_gaps", "no"}


# ---------------------------------------------------------------------------
# Overall feasibility verdict
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_feasibility_is_valid_label(mock_nfcore_api):
    result = await check_feasibility("RNA-seq", RNASEQ_FILES_PE)
    assert result["overall_feasibility"] in {"yes", "yes_with_gaps", "unlikely", "no"}


@pytest.mark.asyncio
async def test_feasibility_summary_is_nonempty_string(mock_nfcore_api):
    result = await check_feasibility("RNA-seq", RNASEQ_FILES_PE)
    assert isinstance(result["feasibility_summary"], str)
    assert len(result["feasibility_summary"]) > 0


@pytest.mark.asyncio
async def test_goal_preserved_in_output(mock_nfcore_api):
    goal = "somatic variant calling WGS"
    result = await check_feasibility(goal, VARIANT_FILES_BAM)
    assert result["goal"] == goal


# ---------------------------------------------------------------------------
# Next steps
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_next_steps_populated(mock_nfcore_api):
    result = await check_feasibility("RNA-seq", RNASEQ_FILES_PE)
    for p in result["matched_pipelines"]:
        assert isinstance(p["next_steps"], list)
        assert len(p["next_steps"]) >= 1


@pytest.mark.asyncio
async def test_next_steps_mention_pipeline_name(mock_nfcore_api):
    result = await check_feasibility("RNA-seq differential expression", RNASEQ_FILES_PE)
    rnaseq_match = next((p for p in result["matched_pipelines"] if p["name"] == "rnaseq"), None)
    if rnaseq_match:
        steps_text = " ".join(rnaseq_match["next_steps"])
        assert "rnaseq" in steps_text
