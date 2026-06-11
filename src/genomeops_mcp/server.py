"""GenomeOps FastMCP server — registers all tools and exposes the entrypoint."""

import logging

from fastmcp import FastMCP

from genomeops_mcp.tools.discovery import list_pipelines, get_pipeline_info
from genomeops_mcp.tools.versioning import get_latest_version
from genomeops_mcp.tools.samplesheet import (
    get_samplesheet_schema,
    validate_samplesheet,
    generate_samplesheet,
)
from genomeops_mcp.tools.launch import generate_launch_command
from genomeops_mcp.tools.execution import (
    check_execution_environment,
    setup_environment,
    run_pipeline,
    get_run_status,
    list_runs,
    stop_pipeline,
    get_failure_logs,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

mcp = FastMCP(
    name="genomeops-mcp",
    instructions="""
You are connected to the nf-core bioinformatics pipeline ecosystem.

GenomeOps is faithful plumbing for an EXPERT who wants to run an nf-core pipeline
on their data. It does NOT choose a pipeline for the user, propose or set
parameters, or interpret results — those are the bioinformatician's decisions.
Your job is to make running the pipeline easy, not to make the science calls.

Typical workflow:
1. Use list_pipelines to discover pipelines for a given data type or topic, and
   get_pipeline_info to see what a pipeline expects.
   IMPORTANT: After get_pipeline_info, call get_latest_version to pin to a stable
   release tag. Never use 'master'/'main' in production runs. If get_latest_version
   returns is_dev=true, stop and ask the user before proceeding.
2. Use get_samplesheet_schema to show the required CSV format for the pipeline.
3. Use generate_samplesheet to draft a samplesheet from file paths (mechanical
   sample-name/R1-R2 pairing only; experiment-specific columns are left blank for
   the user to fill in). Use validate_samplesheet to check the user's samplesheet
   against the pipeline's own schema before running.
4. The user supplies the parameters they want. The server does not suggest or set
   parameter values — pass the user's chosen params through to the tools below as-is.
5. Use check_execution_environment to verify Nextflow + a container engine are
   installed and to pick a viable -profile. nf-core builds per-process environments
   automatically; the user does NOT set up conda per tool.
6. Use setup_environment to cache the pipeline and resolve its config (a dry run)
   before spending compute.
7. Use generate_launch_command to assemble the nextflow run command for review.
8. Use run_pipeline to launch. It is gated: with confirm=false it only previews the
   validated command; it runs ONLY with confirm=true, which consumes compute. Always
   show the preview and get explicit user approval before calling with confirm=true.
9. Use get_run_status to poll a launched run; use list_runs to see every run launched
   here. If a run fails, use get_failure_logs to surface the raw error output (the
   failed task's .command.err / .command.sh / exit code) for the user to read.
10. Use stop_pipeline to terminate a run; like run_pipeline it is gated and only sends
    the signal with confirm=true.

Always validate samplesheets before generating launch commands.
Always present review_required outputs to the user for confirmation before executing
any pipeline command. Never call run_pipeline with confirm=true without explicit user
approval of the previewed command.
""",
)

mcp.tool(list_pipelines)
mcp.tool(get_pipeline_info)
mcp.tool(get_latest_version)
mcp.tool(get_samplesheet_schema)
mcp.tool(validate_samplesheet)
mcp.tool(generate_samplesheet)
mcp.tool(generate_launch_command)
mcp.tool(check_execution_environment)
mcp.tool(setup_environment)
mcp.tool(run_pipeline)
mcp.tool(get_run_status)
mcp.tool(list_runs)
mcp.tool(stop_pipeline)
mcp.tool(get_failure_logs)


def main() -> None:
    logger.info("Starting GenomeOps MCP server")
    mcp.run()


if __name__ == "__main__":
    main()
