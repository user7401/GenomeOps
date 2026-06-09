"""Tests for inventory_results (and supporting helpers in results.py)."""

import pytest

from genomeops_mcp.tools.results import inventory_results, _classify_file, _is_noise
from pathlib import Path


# ---------------------------------------------------------------------------
# _classify_file — pure unit tests
# ---------------------------------------------------------------------------

def test_classify_bam():
    fmt, role = _classify_file(Path("star/sample.bam"))
    assert fmt == "bam"
    assert role == "alignment"


def test_classify_vcf_gz():
    fmt, role = _classify_file(Path("variant_calling/sample.vcf.gz"))
    assert fmt == "vcf_gz"
    assert role == "variants"


def test_classify_tsv_in_salmon_dir():
    fmt, role = _classify_file(Path("salmon/quant.sf"))
    # quant.sf has no known extension → unknown/unclassified, but role refined by path
    _, role = _classify_file(Path("salmon/salmon.merged.gene_counts.tsv"))
    assert role == "transcript_quantification"


def test_classify_tsv_counts_by_name():
    _, role = _classify_file(Path("star_salmon/counts.tsv"))
    assert role == "gene_counts"


def test_classify_multiqc_html():
    _, role = _classify_file(Path("multiqc/multiqc_report.html"))
    assert role == "qc_report"


def test_classify_pipeline_info_json():
    _, role = _classify_file(Path("pipeline_info/params_2024.json"))
    assert role == "provenance"


def test_classify_bai_index():
    fmt, role = _classify_file(Path("star/sample.bam.bai"))
    assert fmt == "bai"
    assert role == "alignment_index"


def test_classify_narrowpeak():
    fmt, role = _classify_file(Path("macs2/sample.narrowPeak"))
    assert fmt == "narrowpeak"
    assert role == "peaks"


def test_classify_unknown_extension():
    fmt, role = _classify_file(Path("some/file.xyz"))
    assert fmt == "unknown"
    assert role == "unclassified"


# ---------------------------------------------------------------------------
# _is_noise — pure unit tests
# ---------------------------------------------------------------------------

def test_noise_command_file(tmp_path):
    f = tmp_path / ".command.sh"
    f.write_text("#!/bin/bash")
    assert _is_noise(f, tmp_path) is True


def test_noise_work_subdir(tmp_path):
    work = tmp_path / "work" / "ab" / "cdef"
    work.mkdir(parents=True)
    f = work / "sample.bam"
    f.write_text("")
    assert _is_noise(f, tmp_path) is True


def test_not_noise_regular_file(tmp_path):
    f = tmp_path / "results" / "sample.bam"
    f.parent.mkdir()
    f.write_text("")
    assert _is_noise(f, tmp_path) is False


# ---------------------------------------------------------------------------
# inventory_results integration tests (filesystem-based, no network)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_inventory_dir_not_found():
    result = await inventory_results("/no/such/path/ever")
    assert result.get("error") is True
    assert result["code"] == "DIR_NOT_FOUND"


@pytest.mark.asyncio
async def test_inventory_empty_dir(tmp_path):
    result = await inventory_results(str(tmp_path))
    assert result["error"] is False if "error" in result else True
    assert result["file_count"]["total"] == 0
    assert result["primary_outputs"] == []
    assert result["review_required"] is True


@pytest.mark.asyncio
async def test_inventory_classifies_primary_outputs(tmp_path):
    # Alignment
    bam = tmp_path / "star" / "sample.bam"
    bam.parent.mkdir()
    bam.write_bytes(b"\x00" * 100)

    # Variants
    vcf = tmp_path / "variant_calling" / "calls.vcf.gz"
    vcf.parent.mkdir()
    vcf.write_bytes(b"\x00" * 200)

    # Gene counts
    counts = tmp_path / "star_salmon" / "salmon.merged.gene_counts.tsv"
    counts.parent.mkdir()
    counts.write_text("gene\tsample1\n")

    result = await inventory_results(str(tmp_path))
    paths = {e["path"] for e in result["primary_outputs"]}
    assert "star/sample.bam" in paths
    assert "variant_calling/calls.vcf.gz" in paths
    assert "star_salmon/salmon.merged.gene_counts.tsv" in paths
    assert result["file_count"]["primary"] == 3


@pytest.mark.asyncio
async def test_inventory_qc_bucket(tmp_path):
    mq = tmp_path / "multiqc" / "multiqc_data" / "multiqc_report.html"
    mq.parent.mkdir(parents=True)
    mq.write_text("<html/>")

    result = await inventory_results(str(tmp_path))
    assert result["file_count"]["qc"] >= 1
    qc_paths = {e["path"] for e in result["qc"]}
    assert any("multiqc" in p for p in qc_paths)


@pytest.mark.asyncio
async def test_inventory_provenance_bucket(tmp_path):
    info_dir = tmp_path / "pipeline_info"
    info_dir.mkdir()
    (info_dir / "params_2024-01-01.json").write_text("{}")
    (info_dir / "software_versions.yml").write_text("STAR: 2.7.10a\n")

    result = await inventory_results(str(tmp_path))
    assert result["file_count"]["provenance"] == 2
    assert result["file_count"]["primary"] == 0


@pytest.mark.asyncio
async def test_inventory_indices_bucket(tmp_path):
    bam = tmp_path / "star" / "sample.bam"
    bam.parent.mkdir()
    bam.write_bytes(b"\x00" * 50)
    bai = tmp_path / "star" / "sample.bam.bai"
    bai.write_bytes(b"\x00" * 10)

    result = await inventory_results(str(tmp_path))
    assert result["file_count"]["indices"] == 1
    assert result["file_count"]["primary"] == 1


@pytest.mark.asyncio
async def test_inventory_excludes_work_dir(tmp_path):
    # Files inside work/ must be invisible to the inventory
    work = tmp_path / "work" / "ab" / "cdef1234"
    work.mkdir(parents=True)
    (work / "sample.bam").write_bytes(b"\x00" * 100)
    (work / ".command.sh").write_text("#!/bin/bash")

    # A real output alongside the work dir
    real = tmp_path / "star" / "real.bam"
    real.parent.mkdir()
    real.write_bytes(b"\x00" * 50)

    result = await inventory_results(str(tmp_path))
    paths = {e["path"] for e in result["primary_outputs"]}
    assert "star/real.bam" in paths
    assert not any("work" in p for p in paths)
    assert result["file_count"]["primary"] == 1


@pytest.mark.asyncio
async def test_inventory_unclassified_bucket(tmp_path):
    weird = tmp_path / "custom" / "output.xyz"
    weird.parent.mkdir()
    weird.write_text("data")

    result = await inventory_results(str(tmp_path))
    assert result["file_count"]["unclassified"] == 1
    assert result["unclassified"][0]["path"] == "custom/output.xyz"
    assert result["unclassified"][0]["format"] == "unknown"


@pytest.mark.asyncio
async def test_inventory_total_size_mb(tmp_path):
    f = tmp_path / "star" / "sample.bam"
    f.parent.mkdir()
    f.write_bytes(b"\x00" * 1_048_576)  # exactly 1 MiB

    result = await inventory_results(str(tmp_path))
    assert abs(result["total_size_mb"] - 1.0) < 0.01


@pytest.mark.asyncio
async def test_inventory_no_key_method_is_structural(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    bam = tmp_path / "star" / "sample.bam"
    bam.parent.mkdir()
    bam.write_bytes(b"\x00" * 10)

    result = await inventory_results(str(tmp_path))
    assert result["analysis_method"] == "structural"
    assert not any("downstream" in e for e in result["primary_outputs"])


@pytest.mark.asyncio
async def test_inventory_llm_adds_downstream_hints(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    counts = tmp_path / "star_salmon" / "gene_counts.tsv"
    counts.parent.mkdir()
    counts.write_text("gene\tsample1\n")

    payload = (
        '{"annotations": [{"path": "star_salmon/gene_counts.tsv", '
        '"downstream": "Load into DESeq2 for differential expression analysis.", '
        '"role": "gene_counts"}]}'
    )

    class _Msg:
        content = [type("C", (), {"text": payload})()]

    class _Client:
        def __init__(self, *a, **k):
            self.messages = self

        def create(self, *a, **k):
            return _Msg()

    import anthropic
    monkeypatch.setattr(anthropic, "Anthropic", _Client)

    result = await inventory_results(str(tmp_path))
    assert result["analysis_method"] == "llm_refined"
    outputs = {e["path"]: e for e in result["primary_outputs"]}
    assert outputs["star_salmon/gene_counts.tsv"]["downstream"].startswith("Load into DESeq2")
