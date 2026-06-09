"""GenomeOps FastMCP server — registers all tools and exposes the entrypoint."""

import logging

from fastmcp import FastMCP

from genomeops_mcp.tools.discovery import list_pipelines, get_pipeline_info
from genomeops_mcp.tools.samplesheet import (
    get_samplesheet_schema,
    validate_samplesheet,
    generate_samplesheet,
)
from genomeops_mcp.tools.parameters import get_parameters, suggest_parameters
from genomeops_mcp.tools.configuration import (
    analyze_pipeline_schema,
    configure_parameters,
)
from genomeops_mcp.tools.results import generate_launch_command, parse_run_summary, inventory_results
from genomeops_mcp.tools.execution import (
    check_execution_environment,
    setup_environment,
    estimate_resources,
    run_pipeline,
    get_run_status,
    list_runs,
    stop_pipeline,
    diagnose_run_failure,
    diagnose_resume,
    generate_methods_note,
)
from genomeops_mcp.tools.feasibility import check_feasibility
from genomeops_mcp.tools.versioning import get_latest_version

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

mcp = FastMCP(
    name="genomeops-mcp",
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
9. Use generate_launch_command to assemble the nextflow run command for review.
10. Use check_execution_environment to verify Nextflow + a container engine are
    installed and to pick a viable -profile. nf-core builds per-process
    environments automatically; the user does NOT set up conda per tool.
11. Use setup_environment to cache the pipeline and resolve its config (a dry
    run) before spending compute.
11b. Use estimate_resources to size the job against this machine BEFORE launching:
    it proposes CPU/memory/disk/walltime from the workload + pipeline profile,
    returns a verdict (sufficient/marginal/insufficient), and suggests --max_*
    overrides. If the verdict is marginal/insufficient, surface it to the user
    before run_pipeline. With no ANTHROPIC_API_KEY it returns a heuristic plus the
    raw signals in reasoning_inputs for you to reason over yourself.
12. Use run_pipeline to launch the analysis. It is gated: with confirm=false it
    only previews the validated command; it runs ONLY with confirm=true, which
    consumes compute. Always show the preview and get explicit user approval
    before calling with confirm=true.
13. Use get_run_status to poll a launched run, then parse_run_summary on its
    outdir once it completes. Use list_runs to see every run launched here.
14. If get_run_status reports status="failed", use diagnose_run_failure to get the
    root cause (it reads the failed task's work dir) and concrete fixes.
15. Before re-launching with resume=true, use diagnose_resume to confirm the cache
    and inputs are intact — a changed input or missing work dir silently forces a
    full re-run.
16. Use stop_pipeline to terminate a run; like run_pipeline it is gated and only
    sends the signal with confirm=true.
17. Use generate_methods_note after a run to draft a citable Methods paragraph
    from the recorded provenance (pipeline, version, exact parameters, citations).

Always validate samplesheets before generating launch commands.
Always present review_required outputs to the user for confirmation
before executing any pipeline command. Never silently set an expert_required
parameter — surface it for a human decision. Never call run_pipeline with
confirm=true without explicit user approval of the previewed command.
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
mcp.tool(check_execution_environment)
mcp.tool(setup_environment)
mcp.tool(estimate_resources)
mcp.tool(run_pipeline)
mcp.tool(get_run_status)
mcp.tool(list_runs)
mcp.tool(stop_pipeline)
mcp.tool(diagnose_run_failure)
mcp.tool(diagnose_resume)
mcp.tool(generate_methods_note)
mcp.tool(parse_run_summary)
mcp.tool(inventory_results)


def main() -> None:
    logger.info("Starting GenomeOps MCP server")
    mcp.run()


if __name__ == "__main__":
    main()
