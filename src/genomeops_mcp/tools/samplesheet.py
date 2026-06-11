"""Samplesheet tools: schema inspection, validation, and generation."""

import csv
import io
import logging
import re
from pathlib import Path
from typing import Any

import jsonschema

from genomeops_mcp.clients.github import get_samplesheet_schema as _fetch_schema

logger = logging.getLogger(__name__)

# Common FASTQ filename patterns for R1/R2 detection
_R1_PATTERNS = [
    r"_R1_001\.fastq\.gz$", r"_R1\.fastq\.gz$",
    r"_R1_001\.fq\.gz$",   r"_R1\.fq\.gz$",
    r"_1\.fastq\.gz$",      r"_1\.fq\.gz$",
]
_R2_PATTERNS = [
    r"_R2_001\.fastq\.gz$", r"_R2\.fastq\.gz$",
    r"_R2_001\.fq\.gz$",   r"_R2\.fq\.gz$",
    r"_2\.fastq\.gz$",      r"_2\.fq\.gz$",
]


async def get_samplesheet_schema(
    pipeline_name: str,
    version: str = "master",
) -> dict[str, Any]:
    """Retrieve and describe the samplesheet schema for an nf-core pipeline.

    Fetches assets/schema_input.json from GitHub and returns a structured,
    human- and agent-readable summary of each field — name, type, whether
    required, allowed values, and a plain-English description.

    Use this before calling validate_samplesheet or generate_samplesheet to
    understand what columns the CSV must contain.

    Args:
        pipeline_name: nf-core pipeline name (e.g. "rnaseq", "sarek").
        version: Pipeline version or branch (default "main").

    Returns:
        {
            "pipeline": str,
            "version": str,
            "fields": [{"name", "type", "required", "description",
                        "allowed_values", "default"}]
        }
    """
    logger.info("Tool call: get_samplesheet_schema(%r, %r)", pipeline_name, version)
    try:
        schema = await _fetch_schema(pipeline_name, version)
    except ValueError as exc:
        return {"error": True, "code": "SCHEMA_NOT_FOUND", "message": str(exc)}
    except Exception as exc:
        return {"error": True, "code": "FETCH_ERROR", "message": f"Could not fetch schema: {exc}"}

    fields = _parse_schema_fields(schema)
    return {
        "pipeline": pipeline_name,
        "version": version,
        "fields": fields,
    }


async def validate_samplesheet(
    pipeline_name: str,
    samplesheet_content: str,
    version: str = "master",
) -> dict[str, Any]:
    """Validate a samplesheet CSV against the pipeline's input schema.

    Parses the CSV content in-memory and checks every row against the JSON
    schema for that pipeline. Returns plain-English error messages — never
    raw jsonschema output — with row numbers and fix suggestions.

    Use this before passing a samplesheet to generate_launch_command to catch
    format errors early.

    Args:
        pipeline_name: nf-core pipeline name (e.g. "rnaseq").
        samplesheet_content: Full CSV content as a string (including header row).
        version: Pipeline version or branch (default "main").

    Returns:
        {
            "valid": bool,
            "errors": [{"row", "column", "message", "fix_suggestion"}]
        }
    """
    logger.info("Tool call: validate_samplesheet(%r)", pipeline_name)
    try:
        schema = await _fetch_schema(pipeline_name, version)
    except ValueError as exc:
        return {"error": True, "code": "SCHEMA_NOT_FOUND", "message": str(exc)}
    except Exception as exc:
        return {"error": True, "code": "FETCH_ERROR", "message": f"Could not fetch schema: {exc}"}

    items_schema = schema.get("items", schema)
    required_cols: list[str] = items_schema.get("required", [])
    properties: dict[str, Any] = items_schema.get("properties", {})

    try:
        reader = csv.DictReader(io.StringIO(samplesheet_content.strip()))
        rows = list(reader)
    except Exception as exc:
        return {
            "valid": False,
            "errors": [{"row": 0, "column": "", "message": f"CSV parse error: {exc}", "fix_suggestion": ""}],
        }

    errors = []
    headers = reader.fieldnames or []

    # Check required columns exist
    for col in required_cols:
        if col not in headers:
            errors.append({
                "row": 0,
                "column": col,
                "message": f"Required column '{col}' is missing from the samplesheet header.",
                "fix_suggestion": f"Add a '{col}' column to your CSV.",
            })

    # Validate each row
    for row_idx, row in enumerate(rows, start=2):
        for col, prop in properties.items():
            value = row.get(col, "")
            col_errors = _validate_cell(row_idx, col, value, prop, col in required_cols)
            errors.extend(col_errors)

    return {"valid": len(errors) == 0, "errors": errors}


async def generate_samplesheet(
    pipeline_name: str,
    file_paths: list[str],
    version: str = "master",
) -> dict[str, Any]:
    """Generate a samplesheet CSV from a list of input file paths.

    Infers sample names by stripping standard FASTQ suffixes (_R1_001.fastq.gz, etc.)
    and pairs R1/R2 files automatically. This is purely mechanical bookkeeping:
    fields that depend on the experiment (e.g. strandedness) are left blank for you
    to fill in — the server does not guess them.

    Always sets review_required=true because sample metadata cannot be reliably
    inferred from file names alone.

    Args:
        pipeline_name: nf-core pipeline name (e.g. "rnaseq").
        file_paths: List of file path strings (local or cloud paths).
        version: Pipeline version or branch (default "main").

    Returns:
        {
            "samplesheet": str,        # CSV content ready to write to a file
            "warnings": [str],         # Things a human should verify
            "review_required": true,
            "review_hint": str
        }
    """
    logger.info("Tool call: generate_samplesheet(%r, %d files)", pipeline_name, len(file_paths))
    try:
        schema = await _fetch_schema(pipeline_name, version)
    except ValueError as exc:
        return {"error": True, "code": "SCHEMA_NOT_FOUND", "message": str(exc)}
    except Exception as exc:
        return {"error": True, "code": "FETCH_ERROR", "message": f"Could not fetch schema: {exc}"}

    items_schema = schema.get("items", schema)
    properties: dict[str, Any] = items_schema.get("properties", {})
    required_cols: list[str] = items_schema.get("required", [])

    r1_files, r2_files, other_files = _classify_files(file_paths)
    rows, warnings = _build_rows(r1_files, r2_files, other_files, properties, required_cols)

    columns = list(properties.keys()) if properties else _default_columns(pipeline_name)
    csv_out = _render_csv(rows, columns, required_cols, properties)

    return {
        "samplesheet": csv_out,
        "warnings": warnings,
        "review_required": True,
        "review_hint": (
            "Fill in any experiment-specific columns left blank (e.g. strandedness). "
            "Check sample names are correct. "
            "Confirm paired-end pairing is as expected."
        ),
    }


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _parse_schema_fields(schema: dict[str, Any]) -> list[dict[str, Any]]:
    items = schema.get("items", schema)
    props: dict[str, Any] = items.get("properties", {})
    required: list[str] = items.get("required", [])
    fields = []
    for name, prop in props.items():
        fields.append({
            "name": name,
            "type": prop.get("type", "string"),
            "required": name in required,
            "description": prop.get("description", prop.get("help_text", "")),
            "allowed_values": prop.get("enum", []),
            "default": prop.get("default"),
            "format": prop.get("format"),
        })
    return fields


def _validate_cell(
    row: int,
    col: str,
    value: str,
    prop: dict[str, Any],
    required: bool,
) -> list[dict[str, Any]]:
    errors = []

    if not value:
        if required:
            errors.append({
                "row": row,
                "column": col,
                "message": f"Required column '{col}' is empty in row {row}.",
                "fix_suggestion": f"Provide a value for '{col}'.",
            })
        return errors

    allowed = prop.get("enum", [])
    if allowed and value not in allowed:
        errors.append({
            "row": row,
            "column": col,
            "message": (
                f"Value '{value}' is not allowed for column '{col}'. "
                f"Must be one of: {', '.join(str(v) for v in allowed)}"
            ),
            "fix_suggestion": (
                f"Change '{value}' to one of: {', '.join(str(v) for v in allowed)}"
            ),
        })

    expected_type = prop.get("type", "string")
    if expected_type == "integer":
        try:
            int(value)
        except ValueError:
            errors.append({
                "row": row,
                "column": col,
                "message": f"Column '{col}' expects an integer, got '{value}'.",
                "fix_suggestion": f"Replace '{value}' with a whole number.",
            })
    elif expected_type == "number":
        try:
            float(value)
        except ValueError:
            errors.append({
                "row": row,
                "column": col,
                "message": f"Column '{col}' expects a number, got '{value}'.",
                "fix_suggestion": f"Replace '{value}' with a numeric value.",
            })

    return errors


def _classify_files(
    file_paths: list[str],
) -> tuple[list[str], list[str], list[str]]:
    r1, r2, other = [], [], []
    for fp in file_paths:
        if any(re.search(p, fp) for p in _R1_PATTERNS):
            r1.append(fp)
        elif any(re.search(p, fp) for p in _R2_PATTERNS):
            r2.append(fp)
        else:
            other.append(fp)
    return sorted(r1), sorted(r2), sorted(other)


def _strip_read_suffix(path: str) -> str:
    for pattern in _R1_PATTERNS + _R2_PATTERNS:
        clean = re.sub(pattern, "", Path(path).name)
        if clean != Path(path).name:
            return clean
    # Strip common generic suffixes
    name = Path(path).stem
    for suffix in (".fastq", ".fq", ".bam", ".cram", ".vcf"):
        name = name.replace(suffix, "")
    return name


def _build_rows(
    r1_files: list[str],
    r2_files: list[str],
    other_files: list[str],
    properties: dict[str, Any],
    required_cols: list[str],
) -> tuple[list[dict[str, str]], list[str]]:
    warnings = []

    # Build R2 lookup keyed by stripped name
    r2_lookup: dict[str, str] = {}
    for fp in r2_files:
        key = _strip_read_suffix(fp)
        r2_lookup[key] = fp

    rows = []
    for fp in r1_files:
        sample_name = _strip_read_suffix(fp)
        r2 = r2_lookup.get(sample_name, "")
        if not r2:
            warnings.append(f"No R2 file found for sample '{sample_name}' (R1: {fp}).")

        row: dict[str, str] = {"sample": sample_name, "fastq_1": fp, "fastq_2": r2}

        if "strandedness" in properties or "strandedness" in required_cols:
            # Strandedness is a library-prep fact, not something to guess from a
            # filename. Leave it blank for the expert to fill in.
            row["strandedness"] = ""
            warnings.append(
                "Strandedness left blank — set it yourself for every sample "
                "(it depends on your library preparation protocol, not the filename)."
            )

        rows.append(row)

    # Handle non-FASTQ files (BAM, CRAM, VCF, etc.)
    for fp in other_files:
        sample_name = Path(fp).stem
        row = {"sample": sample_name}
        # Infer column from file extension
        ext = Path(fp).suffix.lower()
        if ext in {".bam", ".cram"}:
            row["bam"] = fp
        elif ext in {".vcf", ".vcf.gz"}:
            row["vcf"] = fp
        else:
            row["input"] = fp
        rows.append(row)

    if not rows:
        warnings.append("No recognised input files found. Check file extensions.")

    return rows, list(dict.fromkeys(warnings))  # deduplicate while preserving order


def _default_columns(pipeline_name: str) -> list[str]:
    defaults = {
        "rnaseq": ["sample", "fastq_1", "fastq_2", "strandedness"],
        "sarek": ["patient", "sample", "status", "sex", "bam", "bai"],
        "chipseq": ["sample", "fastq_1", "fastq_2", "antibody", "control"],
    }
    return defaults.get(pipeline_name.lower(), ["sample", "fastq_1", "fastq_2"])


def _render_csv(
    rows: list[dict[str, str]],
    columns: list[str],
    required_cols: list[str],
    properties: dict[str, Any],
) -> str:
    # Use required columns plus any populated optional ones
    populated_extras = set()
    for row in rows:
        populated_extras.update(row.keys())

    final_cols = [c for c in columns if c in required_cols or c in populated_extras]
    # Ensure any populated columns not in schema order appear at the end
    for col in sorted(populated_extras):
        if col not in final_cols:
            final_cols.append(col)

    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=final_cols, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({col: row.get(col, "") for col in final_cols})
    return out.getvalue()
