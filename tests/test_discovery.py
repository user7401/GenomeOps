"""Tests for discovery tools: list_pipelines and get_pipeline_info."""

import pytest
from genomeops_mcp.tools.discovery import list_pipelines, get_pipeline_info
from genomeops_mcp.cache import cache_clear


@pytest.fixture(autouse=True)
def clear_cache():
    cache_clear()
    yield
    cache_clear()


@pytest.mark.asyncio
async def test_list_pipelines_returns_all(mock_nfcore_api):
    result = await list_pipelines()
    assert "pipelines" in result
    assert "total" in result
    assert result["total"] == len(result["pipelines"])
    assert result["total"] >= 1


@pytest.mark.asyncio
async def test_list_pipelines_no_topic_returns_all(mock_nfcore_api):
    result = await list_pipelines()
    names = [p["name"] for p in result["pipelines"]]
    assert "rnaseq" in names
    assert "sarek" in names


@pytest.mark.asyncio
async def test_list_pipelines_filter_by_topic_rnaseq(mock_nfcore_api):
    result = await list_pipelines(topic="RNA-seq")
    assert result["total"] >= 1
    names = [p["name"] for p in result["pipelines"]]
    assert "rnaseq" in names
    # sarek is variant-calling, should not appear
    assert "sarek" not in names


@pytest.mark.asyncio
async def test_list_pipelines_filter_by_topic_variant(mock_nfcore_api):
    result = await list_pipelines(topic="variant")
    names = [p["name"] for p in result["pipelines"]]
    assert "sarek" in names
    assert "rnaseq" not in names


@pytest.mark.asyncio
async def test_list_pipelines_filter_no_match_returns_empty(mock_nfcore_api):
    result = await list_pipelines(topic="nonexistent_topic_xyz")
    assert result["total"] == 0
    assert result["pipelines"] == []


@pytest.mark.asyncio
async def test_list_pipelines_pipeline_structure(mock_nfcore_api):
    result = await list_pipelines()
    pipeline = result["pipelines"][0]
    assert "name" in pipeline
    assert "description" in pipeline
    assert "topics" in pipeline
    assert "latest_version" in pipeline
    assert "url" in pipeline
    assert pipeline["url"].startswith("https://nf-co.re/")


@pytest.mark.asyncio
async def test_get_pipeline_info_rnaseq(mock_nfcore_api):
    result = await get_pipeline_info("rnaseq")
    assert result.get("error") is None
    assert result["name"] == "rnaseq"
    assert "description" in result
    assert "latest_version" in result
    assert "url" in result
    assert "docs_url" in result
    assert "schema_url" in result
    assert "input_schema_url" in result
    assert "supported_profiles" in result
    assert "required_params" in result
    assert isinstance(result["supported_profiles"], list)
    assert "docker" in result["supported_profiles"]


@pytest.mark.asyncio
async def test_get_pipeline_info_not_found_returns_error(mock_nfcore_api):
    result = await get_pipeline_info("notexist")
    assert result.get("error") is True
    assert result["code"] == "PIPELINE_NOT_FOUND"
    assert "notexist" in result["message"]


@pytest.mark.asyncio
async def test_get_pipeline_info_typo_suggests_correction(mock_nfcore_api):
    result = await get_pipeline_info("rnseq")
    assert result.get("error") is True
    assert result["code"] == "PIPELINE_NOT_FOUND"
    # Should suggest "rnaseq"
    assert "rnaseq" in result["message"].lower()


@pytest.mark.asyncio
async def test_get_pipeline_info_with_version_overrides_urls(mock_nfcore_api):
    result = await get_pipeline_info("rnaseq", version="3.14.0")
    assert "3.14.0" in result["schema_url"]
    assert "3.14.0" in result["input_schema_url"]


@pytest.mark.asyncio
async def test_list_pipelines_topic_case_insensitive(mock_nfcore_api):
    result_upper = await list_pipelines(topic="RNA-SEQ")
    result_lower = await list_pipelines(topic="rna-seq")
    assert result_upper["total"] == result_lower["total"]
