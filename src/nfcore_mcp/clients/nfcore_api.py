"""Client for nf-core pipeline metadata.

Primary source: nf-co.re REST API v2 (richest metadata).
Fallback: GitHub search API (works from all network environments, including
cloud VMs where nf-co.re blocks non-browser traffic with 403).
"""

import logging
from typing import Any

import httpx

from nfcore_mcp.cache import cache_get, cache_set
from nfcore_mcp.clients.github import _github_headers

logger = logging.getLogger(__name__)

_NFCORE_API = "https://nf-co.re/api/v2"
_GITHUB_SEARCH = "https://api.github.com/search/repositories"
_GITHUB_REPO = "https://api.github.com/repos/nf-core"
_GITHUB_RELEASES = "https://api.github.com/repos/nf-core/{name}/releases/latest"  # kept for compat

from nfcore_mcp.clients.github import get_pipeline_release as _get_pipeline_release

_PIPELINES_TTL = 3600       # 1 hour
_PIPELINE_META_TTL = 3600   # 1 hour

# Repos in the nf-core org that are not pipelines
_NON_PIPELINE_REPOS = {
    "tools", "configs", "modules", "subworkflows", "cookiecutter", "website",
    "logos", "nf-validation", "nf-schema", "test-datasets", "TEMPLATE",
    "nf-core.github.io", "nf-co.re", "nf-neuro",
}


async def get_pipelines() -> list[dict[str, Any]]:
    """Return all nf-core pipelines with summary metadata.

    Tries nf-co.re API first; falls back to GitHub search API if unavailable.
    Responses are cached for 1 hour.
    """
    cached = cache_get("nfcore_api:pipelines", _PIPELINES_TTL)
    if cached is not None:
        return cached

    pipelines = await _fetch_via_nfcore_api() or await _fetch_via_github()
    if pipelines:
        cache_set("nfcore_api:pipelines", pipelines)
    return pipelines


async def get_pipeline(name: str) -> dict[str, Any] | None:
    """Return detailed metadata for a single pipeline by name.

    Returns None if the pipeline does not exist. Resolution order:
    1. Per-pipeline cache
    2. nf-co.re API (richest data)
    3. Pipeline list cache (already fetched via GitHub search)
    4. GitHub single-repo API (last resort, may be rate-limited)
    Responses cached 1 hour.
    """
    key = f"nfcore_api:pipeline:{name.lower()}"
    cached = cache_get(key, _PIPELINE_META_TTL)
    if cached is not None:
        return cached

    detail = await _fetch_pipeline_nfcore(name)

    if detail is None:
        # Check if it's in the already-cached pipeline list (avoids a separate API call)
        detail = await _lookup_in_list_cache(name)

    if detail is None:
        detail = await _fetch_pipeline_github(name)

    if detail is None:
        return None

    cache_set(key, detail)
    return detail


async def _lookup_in_list_cache(name: str) -> dict[str, Any] | None:
    """Find a pipeline in the cached full list and resolve its real version."""
    cached_list = cache_get("nfcore_api:pipelines", _PIPELINES_TTL)
    if not cached_list:
        return None
    match = next((p for p in cached_list if p["name"].lower() == name.lower()), None)
    if not match:
        return None

    version = match.get("latest_version", "master")
    # GitHub search returns "master" as a placeholder — resolve to a real tag.
    if version == "master":
        version = await _resolve_version(match["name"])

    return _build_detail(
        name=match["name"],
        description=match.get("description") or "",
        topics=match.get("topics") or [],
        version=version,
    )


# ---------------------------------------------------------------------------
# nf-co.re API path
# ---------------------------------------------------------------------------

async def _fetch_via_nfcore_api() -> list[dict[str, Any]] | None:
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            logger.info("Fetching pipeline list from nf-co.re API")
            resp = await client.get(f"{_NFCORE_API}/pipelines", params={"limit": 200})
            if resp.status_code in (401, 403, 429):
                logger.warning("nf-co.re API returned %s, will use GitHub fallback", resp.status_code)
                return None
            resp.raise_for_status()
            return _normalise_nfcore_list(resp.json())
    except httpx.HTTPError as exc:
        logger.warning("nf-co.re API unavailable (%s), using GitHub fallback", exc)
        return None


async def _fetch_pipeline_nfcore(name: str) -> dict[str, Any] | None:
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.get(f"{_NFCORE_API}/pipelines/{name.lower()}")
            if resp.status_code == 404:
                return None
            if resp.status_code in (401, 403):
                return None
            resp.raise_for_status()
            raw = resp.json()
            if isinstance(raw, list):
                raw = raw[0] if raw else None
            if raw is None:
                return None
            return _normalise_nfcore_detail(raw)
    except httpx.HTTPError:
        return None


# ---------------------------------------------------------------------------
# GitHub API fallback
# ---------------------------------------------------------------------------

async def _fetch_via_github() -> list[dict[str, Any]]:
    """Fetch all nf-core pipelines via GitHub search (topic:pipeline org:nf-core)."""
    results: list[dict[str, Any]] = []
    page = 1
    async with httpx.AsyncClient(timeout=30) as client:
        while True:
            logger.info("Fetching pipeline list from GitHub search (page %d)", page)
            resp = await client.get(
                _GITHUB_SEARCH,
                params={
                    "q": "org:nf-core topic:pipeline",
                    "per_page": 100,
                    "page": page,
                },
                headers=_github_headers(),
            )
            resp.raise_for_status()
            data = resp.json()
            items = data.get("items", [])
            if not items:
                break
            results.extend(_normalise_github_repo(r) for r in items
                           if r["name"].lower() not in _NON_PIPELINE_REPOS
                           and not r.get("archived"))
            if len(items) < 100:
                break
            page += 1

    return results


async def _fetch_pipeline_github(name: str) -> dict[str, Any] | None:
    """Fetch a single pipeline from GitHub API by repo name.

    Returns None on 404 (repo doesn't exist) or 403 (rate-limited — treat
    as unknown, since real pipelines are already in the list cache).
    """
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.get(
                f"{_GITHUB_REPO}/{name.lower()}",
                headers=_github_headers(),
            )
            if resp.status_code in (404, 403):
                return None
            resp.raise_for_status()
            repo = resp.json()

            version = await _fetch_latest_release(client, name.lower())
            topics_resp = await client.get(
                f"{_GITHUB_REPO}/{name.lower()}/topics",
                headers={**_github_headers(), "Accept": "application/vnd.github.mercy-preview+json"},
            )
            topics = topics_resp.json().get("names", []) if topics_resp.status_code == 200 else []

        return _normalise_github_detail(repo, version, topics)
    except httpx.HTTPError:
        return None


async def _resolve_version(name: str) -> str:
    """Return the recommended stable version tag for a pipeline."""
    try:
        info = await _get_pipeline_release(name)
        return info.get("recommended", "master")
    except Exception:
        return "master"


async def _fetch_latest_release(client: httpx.AsyncClient, name: str) -> str:
    """Kept for _fetch_pipeline_github; delegates to the release classifier."""
    return await _resolve_version(name)


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def _normalise_nfcore_list(raw: Any) -> list[dict[str, Any]]:
    items: list[Any] = []
    if isinstance(raw, dict):
        for key in ("pipelines", "results", "data"):
            if key in raw and isinstance(raw[key], list):
                items = raw[key]
                break
        if not items:
            items = [raw]
    elif isinstance(raw, list):
        items = raw

    return [
        {
            "name": item.get("name", ""),
            "description": item.get("description", ""),
            "topics": _coerce_list(item.get("topics") or item.get("topics_tags") or []),
            "latest_version": _nfcore_version(item),
            "url": f"https://nf-co.re/{item.get('name', '')}",
        }
        for item in items
        if item.get("name")
    ]


def _normalise_nfcore_detail(item: dict[str, Any]) -> dict[str, Any]:
    name = item.get("name", "")
    version = _nfcore_version(item)
    return _build_detail(
        name=name,
        description=item.get("description", ""),
        topics=_coerce_list(item.get("topics") or item.get("topics_tags") or []),
        version=version,
    )


def _normalise_github_repo(repo: dict[str, Any]) -> dict[str, Any]:
    name = repo["name"]
    return {
        "name": name,
        "description": repo.get("description") or "",  # GitHub returns null for some repos
        "topics": repo.get("topics") or [],
        "latest_version": "master",  # full version lookup deferred to get_pipeline()
        "url": f"https://nf-co.re/{name}",
    }


def _normalise_github_detail(
    repo: dict[str, Any],
    version: str,
    topics: list[str],
) -> dict[str, Any]:
    name = repo["name"]
    return _build_detail(
        name=name,
        description=repo.get("description", ""),
        topics=topics or repo.get("topics", []),
        version=version,
    )


def _build_detail(
    name: str,
    description: str,
    topics: list[str],
    version: str,
) -> dict[str, Any]:
    # Use the version tag for schema URLs; fall back to master (NOT main)
    schema_ref = version if version != "master" else "master"
    return {
        "name": name,
        "description": description,
        "topics": topics,
        "latest_version": version,
        "url": f"https://nf-co.re/{name}",
        "docs_url": f"https://nf-co.re/{name}/docs/usage",
        "schema_url": (
            f"https://raw.githubusercontent.com/nf-core/{name}"
            f"/{schema_ref}/nextflow_schema.json"
        ),
        "input_schema_url": (
            f"https://raw.githubusercontent.com/nf-core/{name}"
            f"/{schema_ref}/assets/schema_input.json"
        ),
        "supported_profiles": ["docker", "singularity", "conda", "podman"],
        "required_params": ["input", "outdir"],
        "output_description": "",
    }


def _nfcore_version(item: dict[str, Any]) -> str:
    for key in ("latest_version", "version", "releases"):
        val = item.get(key)
        if val:
            if isinstance(val, list) and val:
                return str(val[0].get("tag_name", val[0]))
            return str(val)
    return "master"


def _coerce_list(val: Any) -> list[str]:
    if isinstance(val, list):
        return [str(v) for v in val]
    if isinstance(val, str):
        return [val] if val else []
    return []
