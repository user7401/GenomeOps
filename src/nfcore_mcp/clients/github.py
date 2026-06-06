"""GitHub raw-content client for fetching nf-core pipeline schemas."""

import logging
from typing import Any

import httpx

from nfcore_mcp.cache import cache_get, cache_set

logger = logging.getLogger(__name__)

_SCHEMA_TTL = 86400  # 24 hours — immutable once a version is tagged
_RAW_BASE = "https://raw.githubusercontent.com/nf-core"


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
