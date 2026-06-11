"""Version resolution tool: get the latest stable release for a pipeline."""

import logging
from typing import Any

from genomeops_mcp.clients.github import get_pipeline_release
from genomeops_mcp.clients.nfcore_api import get_pipelines

logger = logging.getLogger(__name__)


async def get_latest_version(pipeline_name: str) -> dict[str, Any]:
    """Return the latest stable release version for an nf-core pipeline.

    Queries the GitHub Releases API and classifies each release as stable or
    development. If the most recent release is a dev/pre-release version
    (tag contains 'dev', 'alpha', 'beta', 'rc', or GitHub marks it prerelease),
    a warning is returned and review_required is set to true so the human can
    decide whether to proceed with the dev version or use the last stable release.

    Always call this before get_samplesheet_schema or generate_launch_command to
    ensure schema URLs point to a pinned, immutable version tag rather than the
    moving 'master' branch.

    Args:
        pipeline_name: nf-core pipeline name (e.g. "rnaseq", "sarek").

    Returns:
        {
            "pipeline": str,
            "recommended_version": str,   # use this for all subsequent tool calls
            "stable_version": str | None, # latest non-dev release tag
            "latest_tag": str,            # newest tag regardless of stability
            "is_dev": bool,               # True when latest is a dev/pre-release
            "dev_warning": str | None,    # set when is_dev=True, human must review
            "review_required": bool,      # True when is_dev=True
        }
    """
    logger.info("Tool call: get_latest_version(%r)", pipeline_name)

    # Fast-fail if the pipeline doesn't exist at all
    try:
        all_pipelines = await get_pipelines()
        known = {p["name"].lower() for p in all_pipelines}
        if pipeline_name.lower() not in known and all_pipelines:
            import difflib
            suggestion = difflib.get_close_matches(
                pipeline_name.lower(), known, n=1, cutoff=0.6
            )
            msg = f"Pipeline '{pipeline_name}' not found."
            if suggestion:
                msg += f" Did you mean '{suggestion[0]}'?"
            return {"error": True, "code": "PIPELINE_NOT_FOUND", "message": msg}
    except Exception:
        pass  # proceed even if list fetch fails

    try:
        info = await get_pipeline_release(pipeline_name)
    except Exception as exc:
        logger.warning("Could not fetch release info for %r: %s", pipeline_name, exc)
        return {
            "error": True,
            "code": "RELEASE_FETCH_ERROR",
            "message": f"Could not fetch release information for '{pipeline_name}': {exc}",
        }

    return {
        "pipeline": info["pipeline"],
        "recommended_version": info["recommended"],
        "stable_version": info["stable_version"],
        "latest_tag": info["latest_tag"],
        "is_dev": info["is_dev"],
        "dev_warning": info["dev_warning"],
        "review_required": info["is_dev"],
    }
