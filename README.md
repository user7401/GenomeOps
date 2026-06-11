# GenomeOps

> **⚠️ Status: developmental / pre-release.** This project is under active development. Tool names, schemas, and behavior may change without notice. Execution tools run real subprocesses (Nextflow, container engines) and can consume real compute — review every gated output carefully before approving anything, and expect rough edges.
>
> *Unofficial, community-built project. Not affiliated with, endorsed by, or maintained by the nf-core community or Seqera/Nextflow.*

GenomeOps is an MCP (Model Context Protocol) server that makes it **easy for a bioinformatician to run an [nf-core](https://nf-co.re) / Nextflow pipeline on their data**. It is faithful plumbing, not a co-scientist: it discovers pipelines, shows a pipeline's own input format, validates your samplesheet against that pipeline's own schema, checks and prepares the execution environment, assembles and launches the run, monitors it, and hands back raw logs when something fails.

**What it deliberately does *not* do.** It does not choose a pipeline for you, and it does not touch the parameter section — no suggesting, classifying, or auto-filling of pipeline parameters. It does not interpret your results, flag QC, or write your Methods. Those are your decisions as the domain expert; GenomeOps just removes the friction of *running* the pipeline you've chosen, with the parameters you've set.

Coverage of "every pipeline" comes from nf-core's standardization, not bespoke per-pipeline code: every nf-core pipeline ships a `nextflow_schema.json` and a samplesheet schema, so the same thin, schema-driven plumbing works uniformly across the whole catalog — with zero embedded opinions.

The server is built around a human-in-the-loop (HITL) model: every step that generates an executable command or spends compute returns a structured "review required" signal so you stay in control of what actually runs.

## Quickstart

```bash
# Run directly with uvx (no install required)
uvx genomeops-mcp

# Or install with uv
uv pip install genomeops-mcp
genomeops-mcp
```

### Add to Claude Desktop

Edit `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) or `%APPDATA%\Claude\claude_desktop_config.json` (Windows):

```json
{
  "mcpServers": {
    "genomeops": {
      "command": "uvx",
      "args": ["genomeops-mcp"]
    }
  }
}
```

### Add to Claude Code (CLI)

```bash
claude mcp add genomeops -- uvx genomeops-mcp
```

Or add to `.claude/mcp.json` in your project:

```json
{
  "mcpServers": {
    "genomeops": {
      "command": "uvx",
      "args": ["genomeops-mcp"]
    }
  }
}
```

GenomeOps needs no API keys. It calls no LLM of its own — the reasoning lives in the agent and the expertise lives in you.

## Tool Reference

### Discovery

| Tool | Inputs | Outputs |
|------|--------|---------|
| `list_pipelines` | `topic?: str` | List of nf-core pipelines with name, description, version |
| `get_pipeline_info` | `pipeline_name: str`, `version?: str` | Full metadata, profiles, schema URLs |
| `get_latest_version` | `pipeline_name: str` | Latest stable release tag; flags dev/pre-releases (`is_dev`) so the agent stops and asks before using one |

### Samplesheet (your input data)

| Tool | Inputs | Outputs |
|------|--------|---------|
| `get_samplesheet_schema` | `pipeline_name: str`, `version?: str` | The pipeline's own column definitions, types, allowed values |
| `generate_samplesheet` | `pipeline_name: str`, `file_paths: list[str]` | CSV draft from mechanical sample-name/R1–R2 pairing only. Experiment-specific columns (e.g. strandedness) are left **blank** for you to fill — the server does not guess them |
| `validate_samplesheet` | `pipeline_name: str`, `samplesheet_content: str` | `valid: bool`, plain-English errors with row numbers, checked against the pipeline's own schema |

### Launch & execution

You choose the parameters. Pass them through `params` and they are written verbatim — GenomeOps never inspects, suggests, or alters a parameter value.

| Tool | Inputs | Outputs |
|------|--------|---------|
| `generate_launch_command` | `pipeline_name`, `version`, `profile`, `samplesheet_path`, `outdir`, `params?` | Validated `nextflow run` command string (params passed through as-is) |
| `check_execution_environment` | `profile?: str` | Probes for Nextflow, Java, and container/conda engines (Docker, Singularity, Apptainer, Podman, Conda, Mamba), checks the Docker daemon, recommends a viable `-profile`, and reports what's missing with install hints |
| `setup_environment` | `pipeline_name`, `version`, `profile` | Runs `nextflow pull` + `nextflow config` to cache the pipeline and resolve its configuration as a dry run, before any compute is spent |
| `run_pipeline` | `pipeline_name`, `version`, `profile`, `samplesheet_path`, `outdir`, `params?`, `run_name?`, `confirm: bool` | **Gated.** With `confirm=false` (default) it only previews the validated command and changes nothing. It launches a detached background `nextflow run` only when called with `confirm=true` — which must follow explicit human approval of the preview. Parameters are written to a `-params-file`, and a provenance manifest is recorded |
| `get_run_status` | `run_name: str` | Polls a launched run: process liveness, log tail, and a parsed status (`running` / `completed` / `failed` / `unknown`) |
| `list_runs` | — | Lists every run launched through the server (newest first), refreshing liveness and resolving terminal status from the log |
| `stop_pipeline` | `run_name`, `confirm: bool` | **Gated** like `run_pipeline`: previews unless `confirm=true`, then sends SIGTERM to the detached process group. The run can be continued later with `resume=true` |
| `get_failure_logs` | `run_name: str` | For a failed run, locates the failed process and its work dir and returns the **raw** `.command.err`, `.command.sh`, exit code, and run-log tail — the evidence, gathered in one place, for you to read. It does not classify the cause or prescribe fixes |

## Typical Agent Workflow

```
list_pipelines("rna")                              # discover
  → get_pipeline_info("rnaseq")
  → get_latest_version("rnaseq")                    # pin to a stable tag, never "master"
  → get_samplesheet_schema("rnaseq", "3.14.0")      # see the required input format
  → generate_samplesheet("rnaseq", ["/data/sample_R1.fastq.gz", ...])  # mechanical draft
  → validate_samplesheet("rnaseq", <csv>)           # you fill in blanks, then check
  # → you choose the parameters yourself
  → check_execution_environment()                   # confirm Nextflow + a container engine
  → setup_environment("rnaseq", "3.14.0", "docker") # cache + resolve config (dry run)
  → generate_launch_command("rnaseq", ..., params=<your params>)   # human reviews the command
  → run_pipeline(..., confirm=False)                # preview command + params
  → run_pipeline(..., confirm=True)                 # human has approved — launch for real
  → get_run_status("myrun")                         # poll until completed/failed
  → get_failure_logs("myrun")                       # IF failed: raw error evidence to read
```

## Human-in-the-Loop (HITL)

Tools that generate an executable command or gate compute always signal that a human should be in the loop — via `review_required: true`, `requires_confirmation: true`, or an explicit `confirm` parameter:

- `get_latest_version` — if the latest release is a dev/pre-release, the agent must stop and ask before using it
- `generate_samplesheet` — experiment-specific fields are left blank for you; the server does not guess them
- `validate_samplesheet` — surfaces errors against the pipeline's own schema; you decide the fix
- `generate_launch_command` — pipelines consume real compute resources
- `run_pipeline` — hard-gated behind `confirm=true`; with `confirm=false` it only previews and launches nothing
- `stop_pipeline` — hard-gated behind `confirm=true`; previews what would be terminated otherwise
- `get_failure_logs` — returns the raw evidence for you to interpret, not a diagnosis to accept

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
