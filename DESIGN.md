# Design: GenomeOps

## 1. The Problem

A bioinformatician today who wants to run an nf-core pipeline must:

1. Browse nf-co.re to find the right pipeline for their data type
2. Open the pipeline docs to find the samplesheet format
3. Manually inspect `assets/schema_input.json` to understand field types and allowed values
4. Write a CSV samplesheet by hand, hoping they got strandedness, column order, and required fields right
5. Browse another docs page to find which `--genome` flag to pass
6. Scan `nextflow_schema.json` (often 1,000+ lines of JSON) to find the right aligner flag
7. Assemble a `nextflow run` command from memory or examples
8. Debug schema validation errors at pipeline startup

This is fine for experts who run the same pipeline weekly. For occasional users, new team members, or AI agents trying to help — it's an opaque, error-prone process entirely dependent on navigating dense documentation.

## 2. The Old Interface

- **nf-co.re website**: Human-readable docs, not machine-parseable
- **Raw JSON schemas**: `nextflow_schema.json` contains 800–2000 lines of nested JSON. The structure is well-defined but requires schema traversal logic to extract useful information
- **CLI flags**: `nextflow run nf-core/rnaseq --help` prints hundreds of flags with no guidance on which matter for a given experiment
- **No validation before launch**: Schema validation happens at pipeline startup. Mistakes in samplesheets cause failures after potentially pulling gigabytes of containers

## 3. The Agent-Native Interface

GenomeOps turns these raw data sources into **semantic tools**:

| Raw interface | Agent-native tool |
|---|---|
| Browse nf-co.re pipeline list | `list_pipelines(topic="RNA-seq")` |
| Read docs to understand inputs | `get_pipeline_info("rnaseq")` |
| Parse `schema_input.json` manually | `get_samplesheet_schema("rnaseq")` → structured field summary |
| Write samplesheet from scratch | `generate_samplesheet("rnaseq", file_paths)` → CSV + warnings |
| Hope samplesheet is correct | `validate_samplesheet("rnaseq", csv)` → row-level errors in plain English |
| Scan `nextflow_schema.json` | `get_parameters("rnaseq", group="alignment")` → grouped, typed param list |
| Guess which params to set | `suggest_parameters("rnaseq", experiment_description)` → justified suggestions |
| Assemble command from memory | `generate_launch_command(...)` → validated command string |
| Check results manually | `parse_run_summary("./results")` → per-sample QC + flagged samples |

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
# Agent had no access to the actual schema, hallucinated the field value
```

**With GenomeOps:**
```
Agent calls: get_samplesheet_schema("rnaseq")
→ learns strandedness must be one of: forward, reverse, unstranded

Agent calls: generate_samplesheet("rnaseq", ["sample_A_R1.fastq.gz", "sample_A_R2.fastq.gz"])
→ returns CSV with strandedness="unstranded" + warning: "Verify strandedness matches your library prep"

Agent calls: validate_samplesheet("rnaseq", csv)
→ returns {valid: true, errors: []}

Agent presents CSV to user with review_required: true and strandedness warning
```

### Scenario: Agent choosing parameters for DE analysis

**Without GenomeOps:**
```
User: Run rnaseq for differential expression in human fibroblasts

Agent: Use these flags: --genome hg38 --aligner star --pseudo_aligner salmon
# Wrong: hg38 is not a valid iGenomes key (correct: GRCh38)
# Wrong: --aligner star doesn't exist (correct: star_salmon, star_rsem, hisat2)
```

**With GenomeOps:**
```
Agent calls: suggest_parameters("rnaseq", "human fibroblasts, DE analysis, paired-end")
→ returns:
{
  "suggested_params": {"--genome": "GRCh38", "--aligner": "star_salmon"},
  "justifications": {
    "--genome": "Human data; GRCh38 is the current reference in iGenomes",
    "--aligner": "STAR+Salmon is recommended for DE analysis"
  },
  "review_required": true
}
# All values validated against the actual schema enums
```

## 5. HITL Checkpoints

Three tools always return `review_required: true`:

**`generate_samplesheet`**
Strandedness cannot be determined from a filename. Swap forward/reverse strandedness and you silently double-count reads and get systematically wrong differential expression results. A human with knowledge of the library prep protocol must verify this.

**`suggest_parameters`**
AI suggestions are based on language patterns in the experiment description matched against parameter descriptions. They are starting points — a human domain expert must verify that the suggested genome build, aligner, and analysis options match the actual experimental design.

**`generate_launch_command`**
Pipelines consume real compute resources and may incur cloud costs. The assembled command should be reviewed for correctness, the right version, and the right output directory before execution.

## 6. Caching Strategy

| Data | TTL | Rationale |
|---|---|---|
| Pipeline list | 1 hour | Changes rarely; new pipelines release monthly |
| Pipeline detail | 1 hour | Same |
| Samplesheet schema | 24 hours | Immutable per version tag |
| Nextflow schema | 24 hours | Immutable per version tag |

Cache lives at `~/.cache/genomeops-mcp/`. Clear with `cache_clear()` from `genomeops_mcp.cache`.

## 7. Roadmap

**After v0.4:**
- `v0.5`: Nextflow Tower / Seqera Platform integration — submit runs directly via the Tower API
- `v0.6`: Local samplesheet file ingestion — pass a file path instead of CSV string
- `v0.7`: Pipeline comparison tool — given a data type, compare multiple pipelines' features
- `v0.8`: Cost estimation — estimate cloud compute cost given sample count and genome size
- `v0.9`: nf-core configs integration — expose institutional profiles from `nf-core/configs`
- `v1.0`: nf-test integration — run pipeline unit tests and report results
