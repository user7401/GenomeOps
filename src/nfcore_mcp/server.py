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
from nfcore_mcp.tools.results import generate_launch_command, parse_run_summary

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
1. Use list_pipelines to discover pipelines for a given data type or topic.
2. Use get_pipeline_info to understand what a pipeline expects as input.
3. Use get_samplesheet_schema to learn the required CSV format.
4. Use generate_samplesheet to draft a samplesheet from file paths.
5. Use validate_samplesheet to check for errors before running.
6. Use get_parameters to understand available configuration options.
7. Use suggest_parameters to get AI-assisted parameter recommendations.
8. Use generate_launch_command to assemble the nextflow run command.
9. After a run completes, use parse_run_summary to assess QC and results.

Always validate samplesheets before generating launch commands.
Always present review_required outputs to the user for confirmation
before executing any pipeline command.
""",
)

mcp.tool(list_pipelines)
mcp.tool(get_pipeline_info)
mcp.tool(get_samplesheet_schema)
mcp.tool(validate_samplesheet)
mcp.tool(generate_samplesheet)
mcp.tool(get_parameters)
mcp.tool(suggest_parameters)
mcp.tool(generate_launch_command)
mcp.tool(parse_run_summary)


def main() -> None:
    logger.info("Starting nfcore-mcp server")
    mcp.run()


if __name__ == "__main__":
    main()
