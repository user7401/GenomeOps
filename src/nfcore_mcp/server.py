"""nfcore-mcp FastMCP server — registers all tools and exposes the entrypoint."""

import logging

from fastmcp import FastMCP

from nfcore_mcp.tools.discovery import list_pipelines, get_pipeline_info
from nfcore_mcp.tools.samplesheet import (
    get_samplesheet_schema,
    validate_samplesheet,
    generate_samplesheet,
)
from nfcore_mcp.tools.parameters import get_parameters, suggest_parameters
from nfcore_mcp.tools.configuration import (
    analyze_pipeline_schema,
    configure_parameters,
)
from nfcore_mcp.tools.results import generate_launch_command, parse_run_summary
from nfcore_mcp.tools.feasibility import check_feasibility
from nfcore_mcp.tools.versioning import get_latest_version

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

mcp = FastMCP(
    name="nfcore-mcp",
    instructions="""
You are connected to the nf-core bioinformatics pipeline ecosystem.

Typical workflow:
0. Use check_feasibility first when a user arrives with files and a goal but
   hasn't chosen a pipeline — it matches, audits gaps, and prescribes next steps.
1. Use list_pipelines to discover pipelines for a given data type or topic.
2. Use get_pipeline_info to understand what a pipeline expects as input.
   IMPORTANT: After get_pipeline_info, always call get_latest_version to pin
   to a stable release tag. Never use 'master' in production runs.
   If get_latest_version returns is_dev=true, stop and ask the user before
   proceeding — they must decide whether to accept the dev version.
3. Use get_samplesheet_schema to learn the required CSV format.
4. Use generate_samplesheet to draft a samplesheet from file paths.
5. Use validate_samplesheet to check for errors before running.
6. Use analyze_pipeline_schema to map the pipeline's FULL decision tree — every
   branch point (aligner/tool choices, skippable stages) and every parameter,
   classified by how hard it is to set. Do this BEFORE configuring; the
   samplesheet columns are only a fraction of what a pipeline exposes.
7. Use configure_parameters to resolve the non-obvious parameters. Pass any
   values the user wants to set themselves via user_choices (they may set some,
   all, or none). Pass mode="expert" ONLY when the user has asked the agent to
   propose values — expert mode uses AI to suggest context-dependent values from
   the experiment description. Parameters in needs_user_input (especially tier
   "expert_required") MUST be put to the user; never guess them.
8. Use get_parameters / suggest_parameters for a flat list or one-shot AI
   recommendations when the structured framework above is more than you need.
9. Use generate_launch_command to assemble the nextflow run command.
10. After a run completes, use parse_run_summary to assess QC and results.

Always validate samplesheets before generating launch commands.
Always present review_required outputs to the user for confirmation
before executing any pipeline command. Never silently set an expert_required
parameter — surface it for a human decision.
""",
)

mcp.tool(check_feasibility)
mcp.tool(get_latest_version)
mcp.tool(list_pipelines)
mcp.tool(get_pipeline_info)
mcp.tool(get_samplesheet_schema)
mcp.tool(validate_samplesheet)
mcp.tool(generate_samplesheet)
mcp.tool(analyze_pipeline_schema)
mcp.tool(configure_parameters)
mcp.tool(get_parameters)
mcp.tool(suggest_parameters)
mcp.tool(generate_launch_command)
mcp.tool(parse_run_summary)


def main() -> None:
    logger.info("Starting nfcore-mcp server")
    mcp.run()


if __name__ == "__main__":
    main()
