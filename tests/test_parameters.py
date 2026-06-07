"""Tests for parameter tools: get_parameters and suggest_parameters."""

import pytest
from nfcore_mcp.tools.parameters import get_parameters, suggest_parameters
from nfcore_mcp.cache import cache_clear


@pytest.fixture(autouse=True)
def clear_cache():
    cache_clear()
    yield
    cache_clear()


@pytest.mark.asyncio
async def test_get_parameters_returns_groups(mock_github_schemas):
    result = await get_parameters("rnaseq")
    assert "groups" in result
    assert result["pipeline"] == "rnaseq"
    assert len(result["groups"]) > 0


@pytest.mark.asyncio
async def test_get_parameters_group_names(mock_github_schemas):
    result = await get_parameters("rnaseq")
    group_names = list(result["groups"].keys())
    # Should contain at least input/output and reference groups from our mock
    assert any("input" in g.lower() or "output" in g.lower() for g in group_names)


@pytest.mark.asyncio
async def test_get_parameters_param_structure(mock_github_schemas):
    result = await get_parameters("rnaseq")
    for group_params in result["groups"].values():
        for param in group_params:
            assert "name" in param
            assert param["name"].startswith("--")
            assert "type" in param
            assert "description" in param
            assert "required" in param
            assert "allowed_values" in param
            assert "is_path" in param


@pytest.mark.asyncio
async def test_get_parameters_required_flags(mock_github_schemas):
    result = await get_parameters("rnaseq")
    all_params = [p for g in result["groups"].values() for p in g]
    required = [p for p in all_params if p["required"]]
    names = [p["name"] for p in required]
    assert "--input" in names
    assert "--outdir" in names


@pytest.mark.asyncio
async def test_get_parameters_allowed_values(mock_github_schemas):
    result = await get_parameters("rnaseq")
    all_params = [p for g in result["groups"].values() for p in g]
    genome = next((p for p in all_params if p["name"] == "--genome"), None)
    assert genome is not None
    assert "GRCh38" in genome["allowed_values"]


@pytest.mark.asyncio
async def test_get_parameters_filter_by_group(mock_github_schemas):
    result = await get_parameters("rnaseq", group="reference genome")
    assert len(result["groups"]) >= 1
    group_names = [k.lower() for k in result["groups"].keys()]
    assert any("reference" in k for k in group_names)


@pytest.mark.asyncio
async def test_get_parameters_group_filter_no_match(mock_github_schemas):
    result = await get_parameters("rnaseq", group="nonexistent_section_xyz")
    assert result["groups"] == {}


@pytest.mark.asyncio
async def test_get_parameters_is_path_detected(mock_github_schemas):
    result = await get_parameters("rnaseq")
    all_params = [p for g in result["groups"].values() for p in g]
    fasta = next((p for p in all_params if p["name"] == "--fasta"), None)
    assert fasta is not None
    assert fasta["is_path"] is True


@pytest.mark.asyncio
async def test_get_parameters_schema_not_found(mock_github_schemas):
    result = await get_parameters("notexist")
    assert result.get("error") is True
    assert result["code"] == "SCHEMA_NOT_FOUND"


@pytest.mark.asyncio
async def test_suggest_parameters_no_key_defers_to_host(mock_github_schemas, monkeypatch):
    """Without a key, suggest_parameters degrades gracefully instead of erroring."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    result = await suggest_parameters("rnaseq", "human paired-end RNA-seq")
    assert result.get("error") is not True
    assert result["analysis_method"] == "deferred_to_host"
    assert result["llm_available"] is False
    assert result["review_required"] is True
    # The schema is handed back so an agent host can reason over it itself.
    assert result["schema_summary"]
    assert result["experiment_description"] == "human paired-end RNA-seq"
