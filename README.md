# nfcore-mcp

An MCP (Model Context Protocol) server that exposes the [nf-core](https://nf-co.re) bioinformatics pipeline ecosystem as structured, agent-consumable tools. Any AI agent — Claude, GPT, Cursor — can discover pipelines, build valid samplesheets, suggest parameters, and parse results without hallucinating nf-core-specific knowledge.

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

| Tool | Inputs | Outputs |
|------|--------|---------|
| `list_pipelines` | `topic?: str` | List of pipelines with name, description, version |
| `get_pipeline_info` | `pipeline_name: str`, `version?: str` | Full metadata, profiles, schema URLs |
| `get_samplesheet_schema` | `pipeline_name: str`, `version?: str` | Column definitions, types, allowed values |
| `validate_samplesheet` | `pipeline_name: str`, `samplesheet_content: str` | `valid: bool`, plain-English errors with row numbers |
| `generate_samplesheet` | `pipeline_name: str`, `file_paths: list[str]` | CSV string + review warnings |
| `get_parameters` | `pipeline_name: str`, `version?: str`, `group?: str` | Grouped parameter definitions |
| `suggest_parameters` | `pipeline_name: str`, `experiment_description: str` | AI-suggested params + justifications |
| `generate_launch_command` | `pipeline_name`, `version`, `profile`, `samplesheet_path`, `outdir`, `params?` | `nextflow run` command string |
| `parse_run_summary` | `results_dir: str` | Per-sample QC, flagged samples, run status |

## Typical Agent Workflow

```
list_pipelines(topic="RNA-seq")
  → get_pipeline_info("rnaseq")
  → get_samplesheet_schema("rnaseq")
  → generate_samplesheet("rnaseq", ["/data/sample_R1.fastq.gz", ...])
  → validate_samplesheet("rnaseq", <csv>)          # fix any errors
  → get_parameters("rnaseq")
  → suggest_parameters("rnaseq", "Human paired-end stranded RNA-seq...")
  → generate_launch_command("rnaseq", ...)          # human reviews before running
  → parse_run_summary("./results")
```

## Human-in-the-Loop (HITL)

Tools that generate executable content always return `review_required: true`. This signals to agents that a human should approve before acting:

- `generate_samplesheet` — strandedness cannot be inferred reliably
- `suggest_parameters` — AI suggestions need domain expert review
- `generate_launch_command` — pipelines consume real compute resources

## Development

```bash
git clone https://github.com/user7401/nfcore-for-agents
cd nfcore-for-agents
uv sync --extra dev
uv run pytest -v
```

## License

MIT
