# Design: GenomeOps

## 1. The Problem (and what it is *not*)

A bioinformatician who wants to run an nf-core pipeline on their data has to do a lot of mechanical, error-prone bookkeeping: find the pipeline, pin a stable version, read the samplesheet format out of `assets/schema_input.json`, hand-write a CSV, get the required columns right, confirm the machine has Nextflow + a container engine, assemble a `nextflow run` command, launch it, and babysit it.

That bookkeeping is the friction GenomeOps removes. **The science is not friction, and GenomeOps does not touch it.** Choosing the pipeline, setting the parameters, judging the QC, and interpreting the results are the expert's job — they are the reason the expert is in the loop at all. An earlier version of this project tried to do those things too (it classified and proposed parameters, picked pipelines from your files, flagged QC, wrote Methods prose). That overreached: it took agency and learning away from the very people it was meant to help. This version is deliberately scoped back to **faithful plumbing**.

### The boundary

| GenomeOps does | GenomeOps does **not** |
|---|---|
| Discover pipelines and pin versions | Choose a pipeline for you from your data |
| Show a pipeline's own samplesheet/parameter schema | Suggest, classify, or auto-fill parameter values |
| Pair R1/R2 and draft a samplesheet skeleton | Guess experiment-specific fields (e.g. strandedness) |
| Validate your samplesheet against the pipeline's own schema | Decide what's wrong with your experiment |
| Check/prepare the environment, assemble, launch, monitor | Interpret QC, flag samples, write your Methods |
| Surface raw failure logs | Diagnose the root cause or prescribe a fix |

If a step requires domain judgment, GenomeOps surfaces the facts and stops. The agent puts those facts to the expert; the expert decides.

## 2. Why nf-core, and "a case for every pipeline"

GenomeOps does not hard-code per-pipeline logic. Coverage of the whole catalog falls out of nf-core's standardization: every nf-core pipeline publishes a `nextflow_schema.json` and an `assets/schema_input.json`, and is launched the same way (`nextflow run nf-core/<name> -r <version> -profile <engine> -params-file ...`). One thin, schema-driven implementation therefore supports *every* nf-core pipeline uniformly — and because it only ever reflects each pipeline's own published schema, it injects no opinions of its own.

## 3. The Agent-Native Interface

GenomeOps turns nf-core's raw data sources and the Nextflow CLI into **mechanical tools** an agent can call:

| Raw interface | Agent-native tool |
|---|---|
| Browse nf-co.re pipeline list | `list_pipelines(topic="rna")` |
| Read docs to understand inputs | `get_pipeline_info("rnaseq")` |
| Find the latest stable tag | `get_latest_version("rnaseq")` |
| Parse `schema_input.json` manually | `get_samplesheet_schema("rnaseq")` → structured field summary |
| Write samplesheet skeleton from scratch | `generate_samplesheet("rnaseq", file_paths)` → CSV draft + blanks for you to fill |
| Hope the samplesheet is correct | `validate_samplesheet("rnaseq", csv)` → row-level errors in plain English |
| Check Nextflow/engine install by hand | `check_execution_environment()` |
| Cache + resolve config before launch | `setup_environment("rnaseq", version, profile)` |
| Assemble command from memory | `generate_launch_command(...)` → validated command, your params passed through |
| Run and babysit `nextflow run` | `run_pipeline(...)` (gated), `get_run_status`, `list_runs`, `stop_pipeline` |
| Dig through work dirs after a failure | `get_failure_logs("myrun")` → raw `.command.err`/`.command.sh`/exit code |

## 4. Before vs After

### Scenario: Agent building a samplesheet for RNA-seq

**Without GenomeOps:**
```
User: Create a samplesheet for rnaseq with these files: sample_A_R1.fastq.gz, sample_A_R2.fastq.gz

Agent: Here is your samplesheet:
sample,fastq_1,fastq_2,strandedness
sample_A,sample_A_R1.fastq.gz,sample_A_R2.fastq.gz,auto

# Pipeline fails at launch:
# ERROR: 'auto' is not a valid value for strandedness. Must be: forward, reverse, unstranded
# The agent had no access to the schema and hallucinated the field value.
```

**With GenomeOps:**
```
Agent calls: get_samplesheet_schema("rnaseq")
→ learns the columns and that strandedness must be one of: forward, reverse, unstranded

Agent calls: generate_samplesheet("rnaseq", ["sample_A_R1.fastq.gz", "sample_A_R2.fastq.gz"])
→ returns the CSV with sample/R1/R2 paired, and strandedness LEFT BLANK
  + warning: "Strandedness left blank — set it yourself; it depends on your library prep, not the filename."

Expert fills in strandedness (a fact only they know), then:
Agent calls: validate_samplesheet("rnaseq", csv) → {valid: true, errors: []}
```

The difference from the earlier version: GenomeOps no longer fills in a default like `unstranded`. Quietly defaulting strandedness can silently invert reads and produce systematically wrong differential-expression results — exactly the kind of call that belongs to the expert, not the tool.

### Scenario: Choosing parameters

GenomeOps has no `suggest_parameters` / `configure_parameters` tool by design. The expert decides `--genome`, `--aligner`, and the rest. Whatever they pass via `params` is written to the `-params-file` verbatim:

```
Expert chooses: --genome GRCh38 --aligner star_salmon
Agent calls: generate_launch_command("rnaseq", "3.14.0", "docker", "samplesheet.csv", "results",
                                     params={"--genome": "GRCh38", "--aligner": "star_salmon"})
→ returns the exact nextflow run command, params passed through unchanged, review_required: true
```

## 5. HITL Checkpoints

Tools that produce an executable command or spend compute always return `review_required: true` (or require an explicit `confirm`):

- **`generate_samplesheet`** — the skeleton is mechanical; experiment-specific fields are blank and must be filled by someone who knows the library prep.
- **`generate_launch_command`** — pipelines consume real compute and may incur cloud costs; review the command, version, params, and output directory before running.
- **`run_pipeline` / `stop_pipeline`** — hard-gated behind `confirm=true`; with `confirm=false` they only preview and change nothing.
- **`get_failure_logs`** — returns the raw error evidence for the expert to interpret; it does not decide the cause.

These are structured signals in the JSON response, not blocking prompts. The calling agent surfaces them to the human and waits.

## 6. Caching Strategy

| Data | TTL | Rationale |
|---|---|---|
| Pipeline list | 1 hour | Changes rarely; new pipelines release monthly |
| Pipeline detail | 1 hour | Same |
| Samplesheet schema | 24 hours | Immutable per version tag |

Cache lives at `~/.cache/genomeops-mcp/`. Clear with `cache_clear()` from `genomeops_mcp.cache`.

## 7. Non-goals

To keep the boundary in section 1 honest, these are explicitly out of scope:

- Picking a pipeline from the user's files/goal.
- Any parameter suggestion, classification, or auto-fill.
- QC interpretation or sample flagging.
- Failure root-cause diagnosis or prescribed fixes (raw logs only).
- Drafting Methods text or interpreting outputs.
- Embedding an LLM in the server — the reasoning belongs to the calling agent, the expertise to the user.
