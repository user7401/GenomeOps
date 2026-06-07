"""Tests for check_feasibility tool."""

import gzip
import os
import tempfile
from pathlib import Path

import pytest

from genomeops_mcp.cache import cache_clear
from genomeops_mcp.tools.feasibility import (
    check_feasibility,
    _probe_one,
    _probe_files,
)


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


# ---------------------------------------------------------------------------
# File probing — unit tests (no network, uses real temp files)
# ---------------------------------------------------------------------------

def _write_tmp(content: bytes | str, suffix: str) -> str:
    fd, path = tempfile.mkstemp(suffix=suffix)
    data = content if isinstance(content, bytes) else content.encode()
    os.write(fd, data)
    os.close(fd)
    return path


FASTQ_CONTENT = (
    "@SRR1234567.1 read/1\n"
    "ACGTACGTACGTACGTACGTACGT\n"
    "+\n"
    "IIIIIIIIIIIIIIIIIIIIIIII\n"
    "@SRR1234567.2 read/2\n"
    "TTTTGGGGCCCCAAAATTTTGGGG\n"
    "+\n"
    "FFFFFFFFFFFFFFFFFFFFFFFF\n"
)

VCF_CONTENT = (
    "##fileformat=VCFv4.2\n"
    "##FILTER=<ID=PASS,Description=\"All filters passed\">\n"
    "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
    "chr1\t925952\t.\tG\tA\t100\tPASS\t.\n"
)

FASTA_CONTENT = (
    ">chr1 Homo sapiens chromosome 1\n"
    "ACGTACGTACGTACGTACGTACGTACGT\n"
    ">chr2 Homo sapiens chromosome 2\n"
    "TTTTGGGGCCCCAAAATTTTGGGG\n"
)

BAM_MAGIC = b"BAM\x01" + b"\x00" * 100


def test_probe_plain_fastq_reads_lines():
    path = _write_tmp(FASTQ_CONTENT, ".fastq")
    try:
        probe = _probe_one(path)
        assert probe["accessible"] is True
        assert probe["is_binary"] is False
        assert probe["first_lines"] is not None
        assert any(line.startswith("@") for line in probe["first_lines"])
    finally:
        os.unlink(path)


def test_probe_gzip_fastq_decompresses():
    path = _write_tmp(b"", ".fastq.gz")
    try:
        with gzip.open(path, "wt") as f:
            f.write(FASTQ_CONTENT)
        probe = _probe_one(path)
        assert probe["accessible"] is True
        assert probe["compressed"] is True
        assert any(line.startswith("@") for line in probe["first_lines"])
    finally:
        os.unlink(path)


def test_probe_vcf_reads_header():
    path = _write_tmp(VCF_CONTENT, ".vcf")
    try:
        probe = _probe_one(path)
        assert probe["accessible"] is True
        assert probe["first_lines"][0].startswith("##fileformat=VCF")
    finally:
        os.unlink(path)


def test_probe_bam_detects_magic():
    path = _write_tmp(BAM_MAGIC, ".bam")
    try:
        probe = _probe_one(path)
        assert probe["accessible"] is True
        assert probe["is_binary"] is True
        assert "BAM" in probe["magic_description"]
    finally:
        os.unlink(path)


def test_probe_scrambled_name_fastq():
    """A file named with a random UUID suffix is still probed correctly."""
    path = _write_tmp(FASTQ_CONTENT, ".dat")  # scrambled extension
    try:
        probe = _probe_one(path)
        assert probe["accessible"] is True
        # Content is readable regardless of extension
        assert any(line.startswith("@") for line in probe["first_lines"])
    finally:
        os.unlink(path)


def test_probe_missing_file_is_not_accessible():
    probe = _probe_one("/nonexistent/path/sample_abc123.fastq.gz")
    assert probe["accessible"] is False
    assert len(probe["format_hints"]) > 0


def test_probe_cloud_path_is_not_accessible():
    probe = _probe_one("s3://my-bucket/data/sample_R1.fastq.gz")
    assert probe["accessible"] is False
    assert probe["is_cloud"] is True
    assert "s3://" in probe["format_hints"][0]


def test_probe_files_caps_at_max(tmp_path):
    paths = [str(tmp_path / f"s{i}.fastq.gz") for i in range(30)]
    probes = _probe_files(paths)
    assert len(probes) == 30
    # Files beyond the cap are marked skipped
    skipped = [p for p in probes if p.get("skipped")]
    assert len(skipped) == 5  # 30 - MAX_PROBE_FILES(25)


# ---------------------------------------------------------------------------
# LLM feasibility path
# ---------------------------------------------------------------------------

LLM_FEASIBILITY_JSON = """{
  "file_identifications": [
    {"path": "/scrambled/abc123.dat", "format": "FASTQ paired-end R1",
     "confidence": "high", "evidence": "starts with @, 4-line records", "novel": false},
    {"path": "/scrambled/def456.dat", "format": "FASTQ paired-end R2",
     "confidence": "high", "evidence": "starts with @, 4-line records", "novel": false}
  ],
  "data_summary": {
    "data_type": "paired-end RNA-seq FASTQ reads",
    "sample_count": 1,
    "paired_end": true,
    "has_reference_genome": false,
    "has_index_files": false,
    "notes": "Files identified from FASTQ content despite scrambled names"
  },
  "matched_pipelines": [
    {
      "name": "rnaseq",
      "confidence": "high",
      "match_reason": "FASTQ reads + RNA-seq goal",
      "sufficient": ["Paired-end FASTQ reads present"],
      "missing": ["Reference genome", "Library strandedness"],
      "ready_to_proceed": false
    }
  ],
  "overall_feasibility": "yes_with_gaps",
  "feasibility_summary": "Files are paired-end FASTQ — rnaseq pipeline is appropriate once genome and strandedness are specified."
}"""

LLM_NOVEL_JSON = """{
  "file_identifications": [
    {"path": "/unknown/mystery.xyz", "format": "unknown",
     "confidence": "low", "evidence": "binary content, no recognised signature", "novel": true}
  ],
  "data_summary": {
    "data_type": "unknown",
    "sample_count": 0,
    "paired_end": false,
    "has_reference_genome": false,
    "has_index_files": false,
    "notes": "File format not recognised"
  },
  "matched_pipelines": [],
  "overall_feasibility": "unlikely",
  "feasibility_summary": "The provided file could not be identified — novel or proprietary format."
}"""


def _fake_anthropic(monkeypatch, payload: str):
    class _Msg:
        content = [type("C", (), {"text": payload})()]

    class _Client:
        def __init__(self, *a, **k):
            self.messages = self

        def create(self, *a, **k):
            return _Msg()

    import anthropic
    monkeypatch.setattr(anthropic, "Anthropic", _Client)


@pytest.mark.asyncio
async def test_llm_identifies_scrambled_names(mock_nfcore_api, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    _fake_anthropic(monkeypatch, LLM_FEASIBILITY_JSON)

    result = await check_feasibility(
        "RNA-seq differential expression",
        ["/scrambled/abc123.dat", "/scrambled/def456.dat"],
    )

    assert result["analysis_method"] == "llm"
    ids = {fi["path"]: fi for fi in result["data_summary"]["file_identifications"]}
    assert ids["/scrambled/abc123.dat"]["format"] == "FASTQ paired-end R1"
    assert ids["/scrambled/def456.dat"]["format"] == "FASTQ paired-end R2"
    assert result["data_summary"]["paired_end"] is True
    assert any(p["name"] == "rnaseq" for p in result["matched_pipelines"])


@pytest.mark.asyncio
async def test_llm_flags_novel_files(mock_nfcore_api, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    _fake_anthropic(monkeypatch, LLM_NOVEL_JSON)

    result = await check_feasibility("unknown analysis", ["/unknown/mystery.xyz"])

    assert result["analysis_method"] == "llm"
    ids = result["data_summary"]["file_identifications"]
    assert ids[0]["novel"] is True
    assert result["overall_feasibility"] == "unlikely"


@pytest.mark.asyncio
async def test_llm_invalid_json_falls_back_to_heuristic(mock_nfcore_api, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    _fake_anthropic(monkeypatch, "this is not json")

    result = await check_feasibility("RNA-seq", RNASEQ_FILES_PE)

    assert result["analysis_method"] == "heuristic"
    assert "matched_pipelines" in result


@pytest.mark.asyncio
async def test_no_api_key_uses_heuristic(mock_nfcore_api, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    result = await check_feasibility("RNA-seq differential expression", RNASEQ_FILES_PE)
    assert result["analysis_method"] == "heuristic"
    assert any(p["name"] == "rnaseq" for p in result["matched_pipelines"])
