"""GitHub raw-content client for fetching nf-core pipeline schemas and releases.

Set the GITHUB_TOKEN environment variable to use authenticated requests and
avoid the 60 req/hour unauthenticated rate limit:

    export GITHUB_TOKEN=ghp_...
"""

import logging
import os
import re
from typing import Any

import httpx

from genomeops_mcp.cache import cache_get, cache_set

logger = logging.getLogger(__name__)

_SCHEMA_TTL = 86400   # 24 hours — immutable once a version is tagged
_RELEASE_TTL = 3600   # 1 hour
_RAW_BASE = "https://raw.githubusercontent.com/nf-core"
_RELEASES_URL = "https://api.github.com/repos/nf-core/{pipeline}/releases"

# Matches any tag that looks like a development/pre-release marker.
# nf-core uses "3.15.0dev" style; we also catch alpha/beta/rc.
_DEV_RE = re.compile(r"(dev|alpha|beta|rc\d*|pre[-_.]?\d*)", re.IGNORECASE)


def _github_headers() -> dict[str, str]:
    """Build GitHub API request headers, adding auth token if available."""
    headers = {"Accept": "application/vnd.github+json"}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


async def get_samplesheet_schema(pipeline: str, version: str = "master") -> dict[str, Any]:
    """Fetch and return the parsed samplesheet JSON schema for a pipeline.

    Retrieves assets/schema_input.json from the nf-core GitHub repository.
    Schemas are cached for 24 hours per (pipeline, version) pair.

    Raises:
        ValueError: if the schema cannot be found or parsed.
    """
    key = f"github:schema_input:{pipeline}:{version}"
    cached = cache_get(key, _SCHEMA_TTL)
    if cached is not None:
        logger.debug("Cache hit for samplesheet schema %s@%s", pipeline, version)
        return cached

    url = f"{_RAW_BASE}/{pipeline}/{version}/assets/schema_input.json"
    schema = await _fetch_json(url, pipeline, version, "assets/schema_input.json")
    cache_set(key, schema)
    return schema


async def get_nextflow_schema(pipeline: str, version: str = "master") -> dict[str, Any]:
    """Fetch and return the parsed nextflow_schema.json for a pipeline.

    This is the full parameter schema including all pipeline options, defaults,
    and groupings. Cached for 24 hours per (pipeline, version) pair.

    Raises:
        ValueError: if the schema cannot be found or parsed.
    """
    key = f"github:nextflow_schema:{pipeline}:{version}"
    cached = cache_get(key, _SCHEMA_TTL)
    if cached is not None:
        logger.debug("Cache hit for nextflow schema %s@%s", pipeline, version)
        return cached

    url = f"{_RAW_BASE}/{pipeline}/{version}/nextflow_schema.json"
    schema = await _fetch_json(url, pipeline, version, "nextflow_schema.json")
    cache_set(key, schema)
    return schema


async def get_pipeline_release(pipeline: str) -> dict[str, Any]:
    """Fetch and classify release versions for an nf-core pipeline.

    Queries the GitHub Releases API and distinguishes between stable releases
    and development/pre-release versions. Cached for 1 hour.

    Returns a dict with:
        stable_version  — latest non-dev, non-prerelease tag (or None)
        latest_tag      — newest tag regardless of stability
        is_dev          — True if latest_tag is a dev/pre-release
        dev_warning     — human-readable warning string (None when stable)
        recommended     — the version callers should use by default
    """
    key = f"github:releases:{pipeline}"
    cached = cache_get(key, _RELEASE_TTL)
    if cached is not None:
        logger.debug("Cache hit for releases %s", pipeline)
        return cached

    result = await _fetch_releases(pipeline)
    cache_set(key, result)
    return result


async def _fetch_releases(pipeline: str) -> dict[str, Any]:
    url = _RELEASES_URL.format(pipeline=pipeline)
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            logger.info("Fetching releases for %s", pipeline)
            resp = await client.get(
                url,
                params={"per_page": 30},
                headers=_github_headers(),
            )
            if resp.status_code in (404, 403):
                return _no_releases(pipeline)
            resp.raise_for_status()
            releases = resp.json()
    except httpx.HTTPError as exc:
        logger.warning("Could not fetch releases for %s: %s", pipeline, exc)
        return _no_releases(pipeline)

    if not isinstance(releases, list) or not releases:
        return _no_releases(pipeline)

    # Filter drafts — we never want those
    published = [r for r in releases if not r.get("draft")]
    if not published:
        return _no_releases(pipeline)

    latest = published[0]
    latest_tag: str = latest.get("tag_name", "master")
    latest_is_dev = latest.get("prerelease", False) or bool(_DEV_RE.search(latest_tag))

    # Find latest stable: non-prerelease AND non-dev-tagged
    stable_tag: str | None = None
    for rel in published:
        tag = rel.get("tag_name", "")
        if not rel.get("prerelease") and not _DEV_RE.search(tag):
            stable_tag = tag
            break

    is_dev = latest_is_dev or stable_tag is None
    recommended = stable_tag if stable_tag else latest_tag

    warning: str | None = None
    if latest_is_dev and stable_tag:
        warning = (
            f"The latest release '{latest_tag}' is a development/pre-release version "
            f"and is not recommended for production use. "
            f"The latest stable release is '{stable_tag}'. "
            f"Confirm with the user whether to proceed with '{latest_tag}' or use '{stable_tag}'."
        )
    elif stable_tag is None:
        warning = (
            f"Pipeline '{pipeline}' has no stable release yet (latest tag: '{latest_tag}'). "
            f"This pipeline may still be under active development. "
            f"Confirm with the user before proceeding."
        )

    return {
        "pipeline": pipeline,
        "stable_version": stable_tag,
        "latest_tag": latest_tag,
        "is_dev": is_dev,
        "dev_warning": warning,
        "recommended": recommended,
    }


def _no_releases(pipeline: str) -> dict[str, Any]:
    return {
        "pipeline": pipeline,
        "stable_version": None,
        "latest_tag": "master",
        "is_dev": False,
        "dev_warning": None,
        "recommended": "master",
    }


async def _fetch_json(url: str, pipeline: str, version: str, filename: str) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=30) as client:
        logger.info("Fetching %s for %s@%s", filename, pipeline, version)
        resp = await client.get(url)

    if resp.status_code == 404:
        raise ValueError(
            f"Schema '{filename}' not found for pipeline '{pipeline}' at version '{version}'. "
            f"Check that the pipeline name is correct and the version exists."
        )
    resp.raise_for_status()

    try:
        return resp.json()
    except Exception as exc:
        raise ValueError(
            f"Failed to parse {filename} for {pipeline}@{version}: {exc}"
        ) from exc
