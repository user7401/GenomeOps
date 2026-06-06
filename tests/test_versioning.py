"""Tests for get_latest_version tool and get_pipeline_release client."""

import pytest
import httpx
import respx

from nfcore_mcp.tools.versioning import get_latest_version
from nfcore_mcp.clients.github import get_pipeline_release, _DEV_RE
from nfcore_mcp.cache import cache_clear


RELEASES_BASE = "https://api.github.com/repos/nf-core/{}/releases"

STABLE_RELEASES = [
    {"tag_name": "3.14.0", "prerelease": False, "draft": False, "name": "v3.14.0"},
    {"tag_name": "3.13.0", "prerelease": False, "draft": False, "name": "v3.13.0"},
    {"tag_name": "3.12.0", "prerelease": False, "draft": False, "name": "v3.12.0"},
]

DEV_LATEST_RELEASES = [
    {"tag_name": "3.15.0dev", "prerelease": True, "draft": False, "name": "v3.15.0dev"},
    {"tag_name": "3.14.0",    "prerelease": False, "draft": False, "name": "v3.14.0"},
    {"tag_name": "3.13.0",    "prerelease": False, "draft": False, "name": "v3.13.0"},
]

# Pipeline whose only release is a dev tag (no stable yet)
ONLY_DEV_RELEASES = [
    {"tag_name": "1.0.0dev", "prerelease": True, "draft": False, "name": "v1.0.0dev"},
]

# Pipeline with a stable-looking tag but GitHub marks it prerelease
MARKED_PRERELEASE = [
    {"tag_name": "2.0.0rc1", "prerelease": True, "draft": False, "name": "v2.0.0rc1"},
    {"tag_name": "1.9.0",    "prerelease": False, "draft": False, "name": "v1.9.0"},
]

DRAFT_THEN_STABLE = [
    {"tag_name": "4.0.0", "prerelease": False, "draft": True,  "name": "v4.0.0-draft"},
    {"tag_name": "3.5.0", "prerelease": False, "draft": False, "name": "v3.5.0"},
]


@pytest.fixture(autouse=True)
def clear_cache():
    cache_clear()
    yield
    cache_clear()


# ---------------------------------------------------------------------------
# _DEV_RE pattern tests — no network
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tag,expected", [
    ("3.14.0",      False),
    ("3.15.0dev",   True),
    ("1.0.0alpha",  True),
    ("2.0beta1",    True),
    ("3.0.0rc1",    True),
    ("1.0.0pre1",   True),
    ("dev",         True),
    ("3.14.0-rc2",  True),
    ("3.14.0",      False),
    ("10.2.1",      False),
])
def test_dev_re_pattern(tag, expected):
    assert bool(_DEV_RE.search(tag)) == expected


# ---------------------------------------------------------------------------
# get_pipeline_release — mocked HTTP
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_stable_release_no_warning():
    with respx.mock(assert_all_called=False) as mock:
        mock.get(RELEASES_BASE.format("rnaseq")).mock(
            return_value=httpx.Response(200, json=STABLE_RELEASES)
        )
        result = await get_pipeline_release("rnaseq")

    assert result["stable_version"] == "3.14.0"
    assert result["latest_tag"] == "3.14.0"
    assert result["is_dev"] is False
    assert result["dev_warning"] is None
    assert result["recommended"] == "3.14.0"


@pytest.mark.asyncio
async def test_dev_latest_warns_and_recommends_stable():
    with respx.mock(assert_all_called=False) as mock:
        mock.get(RELEASES_BASE.format("newpipe")).mock(
            return_value=httpx.Response(200, json=DEV_LATEST_RELEASES)
        )
        result = await get_pipeline_release("newpipe")

    assert result["latest_tag"] == "3.15.0dev"
    assert result["stable_version"] == "3.14.0"
    assert result["is_dev"] is True
    assert result["dev_warning"] is not None
    assert "3.15.0dev" in result["dev_warning"]
    assert "3.14.0" in result["dev_warning"]
    assert result["recommended"] == "3.14.0"  # stable, not dev


@pytest.mark.asyncio
async def test_only_dev_release_warns_no_stable():
    with respx.mock(assert_all_called=False) as mock:
        mock.get(RELEASES_BASE.format("newpipe")).mock(
            return_value=httpx.Response(200, json=ONLY_DEV_RELEASES)
        )
        result = await get_pipeline_release("newpipe")

    assert result["stable_version"] is None
    assert result["is_dev"] is True
    assert result["dev_warning"] is not None
    assert "no stable release" in result["dev_warning"].lower()
    assert result["recommended"] == "1.0.0dev"


@pytest.mark.asyncio
async def test_rc_marked_prerelease_warns():
    with respx.mock(assert_all_called=False) as mock:
        mock.get(RELEASES_BASE.format("testpipe")).mock(
            return_value=httpx.Response(200, json=MARKED_PRERELEASE)
        )
        result = await get_pipeline_release("testpipe")

    assert result["latest_tag"] == "2.0.0rc1"
    assert result["stable_version"] == "1.9.0"
    assert result["is_dev"] is True
    assert result["recommended"] == "1.9.0"


@pytest.mark.asyncio
async def test_draft_releases_ignored():
    with respx.mock(assert_all_called=False) as mock:
        mock.get(RELEASES_BASE.format("testpipe")).mock(
            return_value=httpx.Response(200, json=DRAFT_THEN_STABLE)
        )
        result = await get_pipeline_release("testpipe")

    # Draft 4.0.0 must be skipped; 3.5.0 is the effective latest
    assert result["latest_tag"] == "3.5.0"
    assert result["stable_version"] == "3.5.0"
    assert result["is_dev"] is False


@pytest.mark.asyncio
async def test_no_releases_returns_master():
    with respx.mock(assert_all_called=False) as mock:
        mock.get(RELEASES_BASE.format("newpipe")).mock(
            return_value=httpx.Response(200, json=[])
        )
        result = await get_pipeline_release("newpipe")

    assert result["recommended"] == "master"
    assert result["is_dev"] is False


@pytest.mark.asyncio
async def test_404_pipeline_returns_master():
    with respx.mock(assert_all_called=False) as mock:
        mock.get(RELEASES_BASE.format("notexist")).mock(
            return_value=httpx.Response(404)
        )
        result = await get_pipeline_release("notexist")

    assert result["recommended"] == "master"


# ---------------------------------------------------------------------------
# get_latest_version MCP tool
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_tool_stable_output_structure(mock_nfcore_api):
    with respx.mock(assert_all_called=False) as mock:
        mock.get(RELEASES_BASE.format("rnaseq")).mock(
            return_value=httpx.Response(200, json=STABLE_RELEASES)
        )
        result = await get_latest_version("rnaseq")

    assert result.get("error") is None
    for key in ("pipeline", "recommended_version", "stable_version",
                "latest_tag", "is_dev", "dev_warning", "review_required"):
        assert key in result, f"Missing key: {key}"

    assert result["pipeline"] == "rnaseq"
    assert result["recommended_version"] == "3.14.0"
    assert result["is_dev"] is False
    assert result["review_required"] is False
    assert result["dev_warning"] is None


@pytest.mark.asyncio
async def test_tool_dev_sets_review_required(mock_nfcore_api):
    with respx.mock(assert_all_called=False) as mock:
        mock.get(RELEASES_BASE.format("rnaseq")).mock(
            return_value=httpx.Response(200, json=DEV_LATEST_RELEASES)
        )
        result = await get_latest_version("rnaseq")

    assert result["is_dev"] is True
    assert result["review_required"] is True
    assert result["dev_warning"] is not None
    assert len(result["dev_warning"]) > 20
    # Should still surface a usable recommended version
    assert result["recommended_version"] == "3.14.0"


@pytest.mark.asyncio
async def test_tool_unknown_pipeline_returns_error(mock_nfcore_api):
    with respx.mock(assert_all_called=False) as mock:
        mock.get(RELEASES_BASE.format("notexist")).mock(
            return_value=httpx.Response(404)
        )
        result = await get_latest_version("notexist")

    assert result.get("error") is True
    assert result["code"] == "PIPELINE_NOT_FOUND"


@pytest.mark.asyncio
async def test_tool_typo_suggestion(mock_nfcore_api):
    with respx.mock(assert_all_called=False) as mock:
        mock.get(RELEASES_BASE.format("rnseq")).mock(
            return_value=httpx.Response(404)
        )
        result = await get_latest_version("rnseq")

    assert result.get("error") is True
    assert "rnaseq" in result["message"].lower()


@pytest.mark.asyncio
async def test_tool_dev_warning_mentions_both_versions(mock_nfcore_api):
    with respx.mock(assert_all_called=False) as mock:
        mock.get(RELEASES_BASE.format("rnaseq")).mock(
            return_value=httpx.Response(200, json=DEV_LATEST_RELEASES)
        )
        result = await get_latest_version("rnaseq")

    warning = result["dev_warning"]
    assert "3.15.0dev" in warning
    assert "3.14.0" in warning
