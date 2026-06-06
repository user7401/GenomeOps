"""Parameter tools: inspect and suggest nf-core pipeline parameters."""

import logging
import os
from typing import Any

from nfcore_mcp.clients.github import get_nextflow_schema

logger = logging.getLogger(__name__)

# Nextflow schema uses these format hints for path-type parameters
_PATH_FORMATS = {"file-path", "directory-path", "path"}


async def get_parameters(
    pipeline_name: str,
    version: str = "master",
    group: str | None = None,
) -> dict[str, Any]:
    """Retrieve and describe all parameters for an nf-core pipeline.

    Parses the pipeline's nextflow_schema.json and returns every parameter
    with its type, default value, description, allowed values, and whether
    it is required. Parameters are grouped by schema section (e.g.
    "Input/output options", "Reference genome options").

    Use this to understand what flags to pass when building a launch command,
    or before calling suggest_parameters.

    Args:
        pipeline_name: nf-core pipeline name (e.g. "rnaseq", "sarek").
        version: Pipeline version or branch (default "main").
        group: Optional section name to filter to (case-insensitive substring match).
               E.g. "reference genome" returns only genome-related parameters.

    Returns:
        {
            "pipeline": str,
            "version": str,
            "groups": {
                "<group_name>": [
                    {
                        "name": str,
                        "type": str,
                        "default": any,
                        "description": str,
                        "required": bool,
                        "allowed_values": [str],
                        "is_path": bool,
                        "group": str
                    }
                ]
            }
        }
    """
    logger.info("Tool call: get_parameters(%r, %r, group=%r)", pipeline_name, version, group)
    try:
        schema = await get_nextflow_schema(pipeline_name, version)
    except ValueError as exc:
        return {"error": True, "code": "SCHEMA_NOT_FOUND", "message": str(exc)}
    except Exception as exc:
        return {"error": True, "code": "FETCH_ERROR", "message": f"Could not fetch schema: {exc}"}

    grouped = _parse_parameters(schema)

    if group:
        needle = group.lower()
        grouped = {k: v for k, v in grouped.items() if needle in k.lower()}

    return {
        "pipeline": pipeline_name,
        "version": version,
        "groups": grouped,
    }


async def suggest_parameters(
    pipeline_name: str,
    experiment_description: str,
    version: str = "master",
) -> dict[str, Any]:
    """Suggest pipeline parameters based on a plain-language experiment description.

    Reads the pipeline's full parameter schema, then uses the Anthropic API
    (Claude) to reason over the schema and the experiment description, returning
    a set of recommended parameters with plain-English justifications.

    Always returns review_required=true. Never use suggested parameters without
    human review — they are starting points, not ground truth.

    Requires ANTHROPIC_API_KEY to be set in the environment.

    Args:
        pipeline_name: nf-core pipeline name (e.g. "rnaseq").
        experiment_description: Natural-language description of the experiment,
            e.g. "Human RNA-seq, paired-end, stranded, comparing treated vs control
            fibroblasts, want differential expression with DESeq2."
        version: Pipeline version or branch (default "main").

    Returns:
        {
            "suggested_params": {"--param": value},
            "justifications": {"--param": "reason"},
            "review_required": true,
            "warning": str
        }
    """
    logger.info("Tool call: suggest_parameters(%r)", pipeline_name)

    try:
        schema = await get_nextflow_schema(pipeline_name, version)
    except ValueError as exc:
        return {"error": True, "code": "SCHEMA_NOT_FOUND", "message": str(exc)}
    except Exception as exc:
        return {"error": True, "code": "FETCH_ERROR", "message": f"Could not fetch schema: {exc}"}

    grouped = _parse_parameters(schema)
    schema_summary = _format_schema_for_prompt(grouped)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return {
            "error": True,
            "code": "NO_API_KEY",
            "message": (
                "ANTHROPIC_API_KEY environment variable not set. "
                "Set it to enable AI-powered parameter suggestions."
            ),
        }

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        result = await _call_claude(client, pipeline_name, experiment_description, schema_summary)
        return result
    except ImportError:
        return {
            "error": True,
            "code": "MISSING_DEPENDENCY",
            "message": "The 'anthropic' package is required for suggest_parameters.",
        }
    except Exception as exc:
        logger.warning("Claude API call failed: %s", exc)
        return {
            "error": True,
            "code": "AI_ERROR",
            "message": f"Failed to get parameter suggestions: {exc}",
        }


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _parse_parameters(schema: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Extract parameters grouped by schema section."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    definitions = schema.get("definitions", schema.get("$defs", {}))
    required_top: list[str] = schema.get("required", [])

    # nextflow_schema.json groups parameters under top-level `allOf` references
    for section_ref in schema.get("allOf", []):
        raw_ref = section_ref.get("$ref", "")
        for prefix in ("#/definitions/", "#/$defs/"):
            if raw_ref.startswith(prefix):
                raw_ref = raw_ref[len(prefix):]
                break
        ref_key = raw_ref
        section = definitions.get(ref_key, {})
        group_name = section.get("title", ref_key) or ref_key
        props: dict[str, Any] = section.get("properties", {})
        required_in_group: list[str] = section.get("required", [])

        params = []
        for param_name, prop in props.items():
            params.append({
                "name": f"--{param_name}",
                "type": prop.get("type", "string"),
                "default": prop.get("default"),
                "description": prop.get("description", prop.get("help_text", "")),
                "required": param_name in required_top or param_name in required_in_group,
                "allowed_values": prop.get("enum", []),
                "is_path": prop.get("format", "") in _PATH_FORMATS,
                "group": group_name,
            })
        if params:
            grouped[group_name] = params

    # Fallback: flat properties at root level
    if not grouped and "properties" in schema:
        params = []
        for param_name, prop in schema["properties"].items():
            params.append({
                "name": f"--{param_name}",
                "type": prop.get("type", "string"),
                "default": prop.get("default"),
                "description": prop.get("description", prop.get("help_text", "")),
                "required": param_name in required_top,
                "allowed_values": prop.get("enum", []),
                "is_path": prop.get("format", "") in _PATH_FORMATS,
                "group": "Parameters",
            })
        grouped["Parameters"] = params

    return grouped


def _format_schema_for_prompt(grouped: dict[str, list[dict[str, Any]]]) -> str:
    lines = []
    for group_name, params in grouped.items():
        lines.append(f"\n## {group_name}")
        for p in params:
            req = " [REQUIRED]" if p["required"] else ""
            allowed = f" (allowed: {', '.join(str(v) for v in p['allowed_values'])})" if p["allowed_values"] else ""
            default = f" [default: {p['default']}]" if p["default"] is not None else ""
            lines.append(f"- {p['name']}{req}{default}: {p['description']}{allowed}")
    return "\n".join(lines)


async def _call_claude(
    client: Any,
    pipeline_name: str,
    description: str,
    schema_summary: str,
) -> dict[str, Any]:
    import json as _json

    prompt = f"""You are a bioinformatics expert helping configure an nf-core pipeline.

Pipeline: nf-core/{pipeline_name}

Experiment description:
{description}

Available parameters:
{schema_summary}

Based on the experiment description, suggest the most important parameters to set.
Focus on parameters that differ from defaults and are relevant to this experiment.
Do NOT suggest --input or --outdir (those are user-provided separately).

Respond with ONLY valid JSON in this exact format:
{{
  "suggested_params": {{
    "--param_name": "value or true/false"
  }},
  "justifications": {{
    "--param_name": "one-sentence reason based on the experiment"
  }}
}}"""

    message = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = message.content[0].text.strip()
    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    parsed = _json.loads(raw)
    return {
        "suggested_params": parsed.get("suggested_params", {}),
        "justifications": parsed.get("justifications", {}),
        "review_required": True,
        "warning": "Always review suggested parameters before launching a pipeline",
    }
