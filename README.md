# nfcore-mcp

> **⚠️ Status: developmental / pre-release.** This project is under active development. Tool names, schemas, and behavior may change without notice. Execution tools run real subprocesses (Nextflow, container engines) and can consume real compute — review every gated output carefully before approving anything, and expect rough edges.

An MCP (Model Context Protocol) server that exposes the [nf-core](https://nf-co.re) bioinformatics pipeline ecosystem as structured, agent-consumable tools. Any AI agent — Claude, GPT, Cursor — can go from "I have these files and this goal" all the way through pipeline discovery, samplesheet generation, parameter configuration, and **launching and monitoring an actual analysis run**, without hallucinating nf-core-specific knowledge.

The server is built around a human-in-the-loop (HITL) model: every step that involves a judgment call, a cost, or an irreversible action returns a structured "review required" signal so a human stays in control of what actually runs.

## Quickstart

```bash
# Run directly with uvx (no install required)
uvx nfcore-mcp

# Or install with uv
uv pip install nfcore-mcp
nfcore-mcp
```

### Add to Claude Desktop

Edit `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) or `%APPDATA%\Claude\claude_desktop_config.json` (Windows):

```json
{
  "mcpServers": {
    "nfcore": {
      "command": "uvx",
      "args": ["nfcore-mcp"]
    }
  }
}
```

### Add to Claude Code (CLI)

```bash
claude mcp add nfcore -- uvx nfcore-mcp
```

Or add to `.claude/mcp.json` in your project:

```json
{
  "mcpServers": {
    "nfcore": {
      "command": "uvx",
      "args": ["nfcore-mcp"]
    }
  }
}
```

### Enable AI parameter suggestions

```bash
export ANTHROPIC_API_KEY=sk-ant-...
uvx nfcore-mcp
```

## Tool Reference

### Discovery & matching

| Tool | Inputs | Outputs |
|------|--------|---------|
| `check_feasibility` | `goal: str`, `file_paths: list[str]` | Reads actual file content (magic bytes, gzip, first lines) via an LLM to identify formats — even with scrambled filenames — and matches them + the stated goal against candidate pipelines, surfacing gaps |
| `list_pipelines` | `topic?: str` | List of pipelines with name, description, version |
| `get_pipeline_info` | `pipeline_name: str`, `version?: str` | Full metadata, profiles, schema URLs |
| `get_latest_version` | `pipeline_name: str` | Latest stable release tag; flags dev/pre-releases (`is_dev`) so agents stop and ask before using one |

### Samplesheet

| Tool | Inputs | Outputs |
|------|--------|---------|
| `get_samplesheet_schema` | `pipeline_name: str`, `version?: str` | Column definitions, types, allowed values |
| `generate_samplesheet` | `pipeline_name: str`, `file_paths: list[str]` | CSV string + review warnings |
| `validate_samplesheet` | `pipeline_name: str`, `samplesheet_content: str` | `valid: bool`, plain-English errors with row numbers |

### Parameters

| Tool | Inputs | Outputs |
|------|--------|---------|
| `analyze_pipeline_schema` | `pipeline_name: str`, `version: str` | Maps the pipeline's full decision tree — every branch point and parameter classified by an LLM into tiers (`provided_input` → `expert_required`), with extracted expert guidance and conditional dependencies. Cached 24h per version. |
| `configure_parameters` | `pipeline_name`, `version`, `data_summary?`, `experiment_description?`, `user_choices?`, `mode?` | Resolves non-obvious parameters: auto-derives what it can, lets the user set any subset themselves via `user_choices`, and (in `mode="expert"`) uses an LLM to propose context-dependent values — while always escalating `expert_required` parameters back to the human rather than guessing |
| `get_parameters` | `pipeline_name: str`, `version?: str`, `group?: str` | Grouped parameter definitions |
| `suggest_parameters` | `pipeline_name: str`, `experiment_description: str` | AI-suggested params + justifications |

### Launch & execution

| Tool | Inputs | Outputs |
|------|--------|---------|
| `generate_launch_command` | `pipeline_name`, `version`, `profile`, `samplesheet_path`, `outdir`, `params?` | Validated `nextflow run` command string |
| `check_execution_environment` | `profile?: str` | Probes for Nextflow, Java, and container/conda engines (Docker, Singularity, Apptainer, Podman, Conda, Mamba), checks the Docker daemon, recommends a viable `-profile`, and reports what's missing with install hints |
| `setup_environment` | `pipeline_name`, `version`, `profile` | Runs `nextflow pull` + `nextflow config` to cache the pipeline and resolve its configuration as a dry run, before any compute is spent |
| `run_pipeline` | `pipeline_name`, `version`, `profile`, `samplesheet_path`, `outdir`, `params?`, `run_name?`, `confirm: bool` | **Gated.** With `confirm=false` (default) it only previews the validated command and changes nothing. It launches a detached background `nextflow run` only when called with `confirm=true` — which must follow explicit human approval of the preview |
| `get_run_status` | `run_name: str` | Polls a launched run: process liveness, log tail, and a parsed status (`running` / `completed` / `failed` / `unknown`) with suggested next steps |

### Results

| Tool | Inputs | Outputs |
|------|--------|---------|
| `parse_run_summary` | `results_dir: str` | Per-sample QC (from MultiQC), flagged samples with reasons, and overall run status |

## Typical Agent Workflow

```
check_feasibility(goal, file_paths)              # match files+goal → candidate pipeline(s)
  → get_pipeline_info("rnaseq")
  → get_latest_version("rnaseq")                  # pin to a stable tag, never "master"
  → analyze_pipeline_schema("rnaseq", "3.14.0")   # map every decision point & parameter
  → get_samplesheet_schema("rnaseq")
  → generate_samplesheet("rnaseq", ["/data/sample_R1.fastq.gz", ...])
  → validate_samplesheet("rnaseq", <csv>)         # fix any errors
  → configure_parameters(...)                     # resolve params; ask human for expert_required
  → generate_launch_command("rnaseq", ...)        # human reviews the command
  → check_execution_environment()                 # confirm Nextflow + a container engine
  → setup_environment("rnaseq", "3.14.0", "docker")
  → run_pipeline(..., confirm=False)              # preview only
  → run_pipeline(..., confirm=True)               # human has approved — launch for real
  → get_run_status("myrun")                       # poll until completed/failed
  → parse_run_summary("./results")
```

## Human-in-the-Loop (HITL)

Tools that generate executable content, propose AI-derived values, or gate compute always signal that a human should be in the loop — via `review_required: true`, `requires_confirmation: true`, or an explicit `confirm` parameter:

- `check_feasibility` — pipeline match is a recommendation, not a decision
- `get_latest_version` — if the latest release is a dev/pre-release, the agent must stop and ask before using it
- `analyze_pipeline_schema` — surfaces decision points (e.g. aligner choice) for the human to choose, or to delegate to expert mode
- `generate_samplesheet` — strandedness and similar fields cannot be inferred reliably
- `configure_parameters` — `expert_required` parameters are *never* silently set; they're always returned for a human decision
- `suggest_parameters` — AI suggestions need domain expert review
- `generate_launch_command` — pipelines consume real compute resources
- `run_pipeline` — hard-gated behind `confirm=true`; with `confirm=false` it only previews and launches nothing

These are not blocking terminal prompts — they're structured signals in the JSON response. The calling agent is expected to surface them to the human, wait for a reply, and only then proceed.

## Development

```bash
git clone https://github.com/user7401/nfcore-for-agents
cd nfcore-for-agents
uv sync --extra dev
uv run pytest -v
```

## License

MIT
