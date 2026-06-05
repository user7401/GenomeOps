"""Async client for the nf-co.re public REST API v2."""

import logging
from typing import Any

import httpx

from nfcore_mcp.cache import cache_get, cache_set

logger = logging.getLogger(__name__)

_BASE_URL = "https://nf-co.re/api/v2"
_PIPELINES_TTL = 3600       # 1 hour
_PIPELINE_META_TTL = 3600   # 1 hour


async def get_pipelines() -> list[dict[str, Any]]:
    """Return all nf-core pipelines with summary metadata.

    Responses are cached for 1 hour. Each entry contains:
    name, description, topics, latest_version, url.
    """
    cached = cache_get("nfcore_api:pipelines", _PIPELINES_TTL)
    if cached is not None:
        return cached

    async with httpx.AsyncClient(timeout=30) as client:
        logger.info("Fetching pipeline list from nf-co.re API")
        resp = await client.get(f"{_BASE_URL}/pipelines", params={"limit": 200})
        resp.raise_for_status()
        raw = resp.json()

    pipelines = _normalise_pipeline_list(raw)
    cache_set("nfcore_api:pipelines", pipelines)
    return pipelines


async def get_pipeline(name: str) -> dict[str, Any] | None:
    """Return detailed metadata for a single pipeline by name.

    Returns None if the pipeline does not exist.
    Schema fetched from nf-co.re/api/v2/pipelines/{name}.
    Responses cached 1 hour.
    """
    key = f"nfcore_api:pipeline:{name.lower()}"
    cached = cache_get(key, _PIPELINE_META_TTL)
    if cached is not None:
        return cached

    async with httpx.AsyncClient(timeout=30) as client:
        logger.info("Fetching pipeline metadata for '%s'", name)
        resp = await client.get(f"{_BASE_URL}/pipelines/{name.lower()}")
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        raw = resp.json()

    # The API may return a list with one item or a dict depending on endpoint shape
    if isinstance(raw, list):
        raw = raw[0] if raw else None
    if raw is None:
        return None

    pipeline = _normalise_pipeline_detail(raw)
    cache_set(key, pipeline)
    return pipeline


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _normalise_pipeline_list(raw: Any) -> list[dict[str, Any]]:
    items: list[Any] = []
    if isinstance(raw, dict):
        # v2 API wraps results in {"pipelines": [...]} or {"results": [...]}
        for key in ("pipelines", "results", "data"):
            if key in raw and isinstance(raw[key], list):
                items = raw[key]
                break
        if not items:
            # Fallback: maybe the dict IS a single pipeline
            items = [raw]
    elif isinstance(raw, list):
        items = raw

    out = []
    for item in items:
        out.append({
            "name": item.get("name", ""),
            "description": item.get("description", ""),
            "topics": _coerce_list(item.get("topics") or item.get("topics_tags") or []),
            "latest_version": _latest_version(item),
            "url": f"https://nf-co.re/{item.get('name', '')}",
        })
    return out


def _normalise_pipeline_detail(item: dict[str, Any]) -> dict[str, Any]:
    name = item.get("name", "")
    return {
        "name": name,
        "description": item.get("description", ""),
        "topics": _coerce_list(item.get("topics") or item.get("topics_tags") or []),
        "latest_version": _latest_version(item),
        "url": f"https://nf-co.re/{name}",
        "docs_url": f"https://nf-co.re/{name}/docs/usage",
        "schema_url": (
            f"https://raw.githubusercontent.com/nf-core/{name}/"
            f"{_latest_version(item)}/nextflow_schema.json"
        ),
        "input_schema_url": (
            f"https://raw.githubusercontent.com/nf-core/{name}/"
            f"{_latest_version(item)}/assets/schema_input.json"
        ),
        "supported_profiles": ["docker", "singularity", "conda", "podman"],
        "required_params": ["input", "outdir"],
        "output_description": item.get("output_description", ""),
        "extra": {k: v for k, v in item.items()
                  if k not in {"name", "description", "topics", "topics_tags"}},
    }


def _latest_version(item: dict[str, Any]) -> str:
    for key in ("latest_version", "version", "releases"):
        val = item.get(key)
        if val:
            if isinstance(val, list) and val:
                return str(val[0].get("tag_name", val[0]))
            return str(val)
    return "main"


def _coerce_list(val: Any) -> list[str]:
    if isinstance(val, list):
        return [str(v) for v in val]
    if isinstance(val, str):
        return [val] if val else []
    return []
