"""Discovery tools: list and describe nf-core pipelines."""

import difflib
import logging
from typing import Any

from nfcore_mcp.clients.nfcore_api import get_pipelines, get_pipeline

logger = logging.getLogger(__name__)


async def list_pipelines(topic: str | None = None) -> dict[str, Any]:
    """List available nf-core pipelines, optionally filtered by topic or data type.

    Use this tool first when the user wants to run a bioinformatics pipeline
    but hasn't specified which one. Filter by topic to narrow results.

    Topics include: RNA-seq, ChIP-seq, ATAC-seq, variant-calling, methylation,
    amplicon, single-cell, proteomics, metagenomics, genome, Hi-C, nanopore.

    Returns a list of matching pipelines with names, descriptions, and versions.
    Always follow up with get_pipeline_info before proceeding to samplesheet generation.

    Args:
        topic: Optional keyword to filter pipelines (e.g. "RNA-seq", "variant calling").
               Matched case-insensitively against pipeline name, description, and topics.

    Returns:
        {
            "pipelines": [{"name", "description", "topics", "latest_version", "url"}],
            "total": int
        }
    """
    logger.info("Tool call: list_pipelines(topic=%r)", topic)
    try:
        all_pipelines = await get_pipelines()
    except Exception as exc:
        logger.warning("Failed to fetch pipeline list: %s", exc)
        return {
            "error": True,
            "code": "API_ERROR",
            "message": f"Could not fetch pipeline list from nf-co.re: {exc}",
        }

    if topic:
        needle = topic.lower()
        filtered = [
            p for p in all_pipelines
            if needle in p["name"].lower()
            or needle in (p.get("description") or "").lower()
            or any(needle in t.lower() for t in (p.get("topics") or []))
        ]
    else:
        filtered = all_pipelines

    return {"pipelines": filtered, "total": len(filtered)}


async def get_pipeline_info(pipeline_name: str, version: str | None = None) -> dict[str, Any]:
    """Return detailed metadata for a specific nf-core pipeline.

    Use this after list_pipelines to understand what a pipeline expects
    as input, what profiles it supports, what parameters are required,
    and where to find its documentation and schemas.

    Always call this before generating a samplesheet or building a launch command,
    so you know the correct input format and required parameters.

    Args:
        pipeline_name: The nf-core pipeline name (e.g. "rnaseq", "sarek", "chipseq").
        version: Optional version string (e.g. "3.14.0"). Defaults to latest.

    Returns:
        Full pipeline metadata including:
        - description, topics, latest_version, url
        - docs_url, schema_url, input_schema_url
        - supported_profiles: list of execution environments
        - required_params: parameters that must be supplied
        - output_description
    """
    logger.info("Tool call: get_pipeline_info(pipeline_name=%r, version=%r)", pipeline_name, version)

    try:
        info = await get_pipeline(pipeline_name)
    except Exception as exc:
        logger.warning("API error for pipeline %r: %s", pipeline_name, exc)
        return {
            "error": True,
            "code": "API_ERROR",
            "message": f"Could not fetch pipeline info: {exc}",
        }

    if info is None:
        suggestion = await _suggest_pipeline(pipeline_name)
        msg = f"Pipeline '{pipeline_name}' not found."
        if suggestion:
            msg += f" Did you mean '{suggestion}'?"
        return {"error": True, "code": "PIPELINE_NOT_FOUND", "message": msg}

    if version:
        # Override version-specific URLs
        info["schema_url"] = (
            f"https://raw.githubusercontent.com/nf-core/{pipeline_name}"
            f"/{version}/nextflow_schema.json"
        )
        info["input_schema_url"] = (
            f"https://raw.githubusercontent.com/nf-core/{pipeline_name}"
            f"/{version}/assets/schema_input.json"
        )

    return info


async def _suggest_pipeline(name: str) -> str | None:
    try:
        all_pipelines = await get_pipelines()
        names = [p["name"] for p in all_pipelines]
        matches = difflib.get_close_matches(name.lower(), names, n=1, cutoff=0.6)
        return matches[0] if matches else None
    except Exception:
        return None
