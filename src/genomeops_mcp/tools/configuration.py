"""Parameter configuration framework.

Two tools that together let an agent reason about *every* knob a pipeline
exposes — not just the columns that happen to appear in a sample samplesheet:

  analyze_pipeline_schema  — map the full decision tree (branch points, all
                             analysis paths) and classify every parameter into
                             a difficulty tier. Deterministic, no LLM, no key.

  configure_parameters     — the "expert mode" engine. Given what the user is
                             bringing (data) and any choices they want to make
                             themselves, it auto-derives what the data reveals,
                             optionally asks Claude to propose reasoned values
                             for context-dependent parameters, and flags the
                             expert-only knobs for mandatory human review.

Design principle: nothing high-stakes is ever set silently. The user can pin
some, all, or none of the parameters; everything else is either a safe default,
a data-derived fact, an explicitly-reviewed expert suggestion, or an open
question handed back for a decision.
"""

import logging
import os
import re
from typing import Any

from genomeops_mcp.cache import cache_get, cache_set
from genomeops_mcp.clients.github import get_nextflow_schema

logger = logging.getLogger(__name__)

# Classification of an immutable, version-pinned schema is itself immutable —
# cache it as long as the schema (24h covers the moving 'master' branch).
_CLASSIFY_TTL = 86400

# ---------------------------------------------------------------------------
# Tiers
# ---------------------------------------------------------------------------

PROVIDED_INPUT = "provided_input"        # user brings/selects: input, outdir, genome, fasta...
AUTO_DERIVABLE = "auto_derivable"        # inferable from data structure (e.g. single vs paired)
SAFE_DEFAULT = "safe_default"            # nf-core default is fine; leave it alone
CONTEXT_DEPENDENT = "context_dependent"  # threshold/tool-choice — depends on the experiment
EXPERT_REQUIRED = "expert_required"      # no good default without domain/published knowledge

_TIER_ORDER = [
    PROVIDED_INPUT,
    AUTO_DERIVABLE,
    CONTEXT_DEPENDENT,
    EXPERT_REQUIRED,
    SAFE_DEFAULT,
]

_TIER_LABELS = {
    PROVIDED_INPUT: "You provide this (your data or a reference selection).",
    AUTO_DERIVABLE: "Derivable from the structure of your files.",
    SAFE_DEFAULT: "nf-core's default is sensible; leave unless you have a reason.",
    CONTEXT_DEPENDENT: "Depends on your experiment — choose a value or let expert mode propose one.",
    EXPERT_REQUIRED: "No safe default. Requires domain knowledge — must be reviewed by a human.",
}

# Nextflow schema format hints for path-type parameters
_PATH_FORMATS = {"file-path", "directory-path", "path", "file-path-pattern"}

# ---------------------------------------------------------------------------
# Name-based heuristics (ordered classification rules use these)
# ---------------------------------------------------------------------------

# Core inputs the user always supplies or selects.
_PROVIDED_INPUT_NAMES = {
    "input", "outdir", "genome", "fasta", "gtf", "gff", "gff3",
    "transcript_fasta", "additional_fasta", "gene_bed",
}
_INDEX_NAME_RE = re.compile(r"(_index$|^index$|bwa$|^star$|salmon$|hisat2$|bowtie2?$|bismark)", re.I)

# Read pairing is the only thing we can genuinely read off the data structure
# (presence of R2 mate files). Values like read_length / fragment_size are
# experiment facts the user knows, not structural facts we can derive, so they
# are deliberately NOT here — they route to context_dependent/expert questions
# instead of pretending to be auto-derivable and silently resolving to nothing.
_AUTO_DERIVABLE_RE = re.compile(
    r"(single_end|paired_end|^paired$)", re.I
)

# Resource / reporting / boilerplate — always safe to leave at default.
_BOILERPLATE_RE = re.compile(
    r"(max_cpus|max_memory|max_time|publish_dir_mode|^email|multiqc|monochrome|"
    r"plaintext|hook_url|validate_params|^version$|^help$|config_profile|"
    r"custom_config|igenomes_base|igenomes_ignore|tracedir|show_hidden|"
    r"^outdir_mode|enable_conda|singularity_pull|max_multiqc)",
    re.I,
)

# Numeric/threshold names whose "right" value depends on the experiment.
_THRESHOLD_RE = re.compile(
    r"(min_|max_|_min$|_max$|cutoff|threshold|fdr|pval|qval|_length|quality|"
    r"_score|depth|coverage|mapq|fragment|tss|gsize|clip_r|three_prime|"
    r"window|bin_size|fraction|percent|ratio|distance|extend|broad_cutoff|"
    r"macs|narrow|trim|_size$|_count$|num_)",
    re.I,
)

# Boolean toggles that switch a stage of the pipeline on/off (decision points).
_TOGGLE_PREFIX_RE = re.compile(
    r"^(skip_|run_|with_|without_|save_|remove_|only_|use_|do_|enable_|disable_|no_)",
    re.I,
)

# Description-level signal that an undefaulted string is really a file/reference
# input the user supplies. Many nf-core schemas (notably sarek) type reference
# files as bare strings with no `format`, but their descriptions follow a
# convention ("Path to ...", "... indices", a named file format). Reading the
# description — itself a fact in the schema — lets the heuristic recognise these
# as provided inputs instead of conservatively escalating them to expert.
_PATHISH_DESC_RE = re.compile(
    r"^\s*path to\b|\bindices?\b|panel-of-normals|\bvcf\b|\bbed\b|\bfasta\b|\.gz\b",
    re.I,
)


# ===========================================================================
# Tool 1 — analyze_pipeline_schema
# ===========================================================================

async def analyze_pipeline_schema(
    pipeline_name: str,
    version: str = "master",
) -> dict[str, Any]:
    """Map the COMPLETE decision tree of an nf-core pipeline before configuring it.

    Unlike get_samplesheet_schema (which only describes the input CSV columns),
    this reads the full nextflow_schema.json and exposes every analysis path the
    pipeline can take: which aligner/quantifier/peak-caller it will use, which
    stages can be skipped, and every numeric knob that changes the result.

    The schema is flattened deterministically, then each parameter is classified
    by an LLM that reads the actual description and help_text fields — extracting
    any expert guidance embedded there (e.g. "typically 2.7e9 for human") and the
    conditional relationships between parameters (e.g. an index that is only
    needed for a particular aligner). Because a version-pinned schema is
    immutable, this classification is cached for 24h, so it is a one-time cost
    per pipeline version. With no ANTHROPIC_API_KEY set it falls back to a fast
    name-based heuristic.

    Call this immediately after a pipeline is chosen and pinned to a version, so
    the agent — or the user — has the full menu of choices before launching
    expert mode (configure_parameters).

    Args:
        pipeline_name: nf-core pipeline name (e.g. "rnaseq", "chipseq").
        version: Pinned version tag from get_latest_version (default "master").

    Returns:
        {
          "pipeline": str,
          "version": str,
          "classification_method": "llm" | "llm-cached" | "heuristic",
          "parameter_count": int,
          "decision_points": [          # branch points that change the analysis path
            {"param","kind","options","default","stage","tier","affects"}
          ],
          "tool_choices": int,          # mutually-exclusive method selections (aligner, trimmer)
          "optional_stages": int,       # on/off toggles for skippable/optional stages
          "conditional_dependencies": [ # params only relevant under a condition
            {"param","relevant_when"}
          ],
          "tiers": {                    # every parameter, bucketed by difficulty
            "<tier>": {
              "label": str,
              "count": int,
              "params": [ {param record + tier/rationale/expert_guidance/relevant_when} ]
            }
          },
          "summary": str,
          "next_steps": [str]
        }
    """
    logger.info("Tool call: analyze_pipeline_schema(%r, %r)", pipeline_name, version)

    try:
        schema = await get_nextflow_schema(pipeline_name, version)
    except ValueError as exc:
        return {"error": True, "code": "SCHEMA_NOT_FOUND", "message": str(exc)}
    except Exception as exc:
        return {"error": True, "code": "FETCH_ERROR", "message": f"Could not fetch schema: {exc}"}

    params = _parse_rich(schema)
    method = await _classify_all(params, pipeline_name, version)

    decision_points = _decision_points(params)
    # Characterise the decision space in terms a human can act on. The naive
    # product of every option count is meaningless (40 boolean toggles alone is
    # ~2^40 "paths"), so instead report the two distinct kinds of choice:
    #   tool_choices    — mutually-exclusive method selections (aligner, trimmer)
    #   optional_stages — on/off toggles for skippable/optional stages
    tool_choices = sum(1 for dp in decision_points if dp["kind"] == "choice")
    optional_stages = sum(1 for dp in decision_points if dp["kind"] == "toggle")

    tiers: dict[str, Any] = {}
    for tier in _TIER_ORDER:
        members = [p for p in params if p["tier"] == tier]
        if members:
            tiers[tier] = {
                "label": _TIER_LABELS[tier],
                "count": len(members),
                "params": members,
            }

    conditional = [
        {"param": p["name"], "relevant_when": p["relevant_when"]}
        for p in params if p.get("relevant_when")
    ]

    n_ctx = len(tiers.get(CONTEXT_DEPENDENT, {}).get("params", []))
    n_expert = len(tiers.get(EXPERT_REQUIRED, {}).get("params", []))
    summary = (
        f"nf-core/{pipeline_name}@{version} exposes {len(params)} parameters: "
        f"{tool_choices} tool/method choice(s) and {optional_stages} optional "
        f"stage toggle(s). {n_ctx} parameter(s) depend on your experiment and "
        f"{n_expert} need expert review; the rest are inputs you provide, "
        f"data-derivable, or safe defaults."
    )

    return {
        "pipeline": pipeline_name,
        "version": version,
        "classification_method": method,
        "parameter_count": len(params),
        "decision_points": decision_points,
        "tool_choices": tool_choices,
        "optional_stages": optional_stages,
        "conditional_dependencies": conditional,
        "tiers": tiers,
        "summary": summary,
        "next_steps": [
            "Review decision_points — these change which tools run and the shape of the output.",
            "To let the user pick: present context_dependent and expert_required params for choices.",
            f"To auto-configure: call configure_parameters('{pipeline_name}', ...) with the "
            f"experiment description, the data summary, and any user_choices.",
        ],
    }


# ===========================================================================
# Tool 2 — configure_parameters (expert mode)
# ===========================================================================

async def configure_parameters(
    pipeline_name: str,
    experiment_description: str = "",
    version: str = "master",
    data_summary: dict[str, Any] | None = None,
    user_choices: dict[str, Any] | None = None,
    mode: str = "interactive",
) -> dict[str, Any]:
    """Resolve a pipeline's non-obvious parameters, honouring user choices first.

    This is the configuration engine for users who lack deep proficiency with a
    pipeline's many knobs. It classifies every parameter, then for each one
    decides where the value should come from:

      1. user_choices         — anything the user pinned themselves wins, always.
      2. data-derived         — facts read from the data structure (e.g. paired-end).
      3. expert suggestion    — in mode="expert", Claude proposes reasoned values
                                for context-dependent params (with justifications).
      4. needs_user_input     — context-dependent params left unresolved + ALL
                                expert-required params, returned as targeted
                                questions with options/ranges. Never set silently.
      5. safe defaults        — left at nf-core's default value.

    The user may pin some, all, or none of the parameters via user_choices.
    Whatever they don't pin is filled by the data, by expert mode (if requested),
    or returned as an explicit question.

    Args:
        pipeline_name: nf-core pipeline name.
        experiment_description: Plain-language experiment context. Used by
            expert mode to reason about thresholds and tool choices. Optional in
            interactive mode.
        version: Pinned version tag (default "master").
        data_summary: Optional dict from check_feasibility's "data_summary"
            (paired_end, has_reference_genome, sample_count, ...). Drives
            auto-derivation.
        user_choices: Optional {"--param": value} the user wants to set
            themselves. These are accepted verbatim and override every other
            source. Bare names ("aligner") and flag form ("--aligner") both work.
        mode: "interactive" (default) returns questions for the user to answer;
            "expert" additionally calls Claude to propose values for
            context-dependent parameters (requires ANTHROPIC_API_KEY).

    Returns:
        {
          "pipeline","version","mode",
          "resolved_params": {"--param": value},   # ready to feed generate_launch_command
          "param_sources": {"--param": "user"|"derived"|"expert"|"default"},
          "auto_derived":   [ {param,value,evidence} ],
          "expert_suggested":[ {param,value,justification,confidence} ],
          "needs_user_input":[ {param,tier,question,options,suggested_range,default,why_it_matters} ],
          "using_defaults": [ {param,default} ],
          "unresolved_count": int,
          "review_required": true,
          "warning": str
        }
    """
    logger.info(
        "Tool call: configure_parameters(%r, mode=%r, user_choices=%d)",
        pipeline_name, mode, len(user_choices or {}),
    )

    try:
        schema = await get_nextflow_schema(pipeline_name, version)
    except ValueError as exc:
        return {"error": True, "code": "SCHEMA_NOT_FOUND", "message": str(exc)}
    except Exception as exc:
        return {"error": True, "code": "FETCH_ERROR", "message": f"Could not fetch schema: {exc}"}

    data_summary = data_summary or {}
    user_choices = _normalise_choices(user_choices or {})

    params = _parse_rich(schema)
    await _classify_all(params, pipeline_name, version)
    by_name: dict[str, dict[str, Any]] = {p["raw_name"]: p for p in params}

    resolved: dict[str, Any] = {}
    sources: dict[str, str] = {}
    auto_derived: list[dict[str, Any]] = []
    expert_suggested: list[dict[str, Any]] = []
    needs_input: list[dict[str, Any]] = []
    using_defaults: list[dict[str, Any]] = []

    # 1. User choices win unconditionally (some, all, or none).
    consumed: set[str] = set()
    for raw_name, value in user_choices.items():
        flag = f"--{raw_name}"
        resolved[flag] = value
        sources[flag] = "user"
        consumed.add(raw_name)

    # 2. Walk the remaining parameters and route by tier.
    for p in params:
        raw = p["raw_name"]
        if raw in consumed:
            continue
        flag = p["name"]
        tier = p["tier"]

        if tier == AUTO_DERIVABLE:
            derived = _derive_from_data(p, data_summary)
            if derived is not None:
                resolved[flag] = derived["value"]
                sources[flag] = "derived"
                auto_derived.append({"param": flag, **derived})
            else:
                needs_input.append(_question_for(p))
            continue

        if tier == SAFE_DEFAULT:
            if p["default"] is not None:
                using_defaults.append({"param": flag, "default": p["default"]})
            continue

        if tier == PROVIDED_INPUT:
            # input/outdir/genome — the user supplies these elsewhere.
            if raw in {"input", "outdir"} or p["required"]:
                needs_input.append(_question_for(p))
            elif raw in _PROVIDED_INPUT_NAMES:
                # Reference genome/annotation params (genome, fasta, gtf, ...)
                # are arguably the most consequential choice in the run — a
                # wrong or missing reference silently invalidates everything
                # downstream. Never let these fall through to a default or
                # vanish; always force an explicit human decision.
                needs_input.append(_question_for(p))
            elif p["default"] is not None:
                using_defaults.append({"param": flag, "default": p["default"]})
            continue

        if tier == CONTEXT_DEPENDENT:
            needs_input.append(_question_for(p))
            continue

        if tier == EXPERT_REQUIRED:
            needs_input.append(_question_for(p))
            continue

    # 3. Expert mode: ask Claude to propose values for the context-dependent
    #    questions (never for expert_required — those stay for the human).
    expert_error: str | None = None
    if mode == "expert":
        ctx_questions = [q for q in needs_input if q["tier"] == CONTEXT_DEPENDENT]
        if ctx_questions:
            proposed, expert_error = await _expert_fill(
                pipeline_name, experiment_description, data_summary,
                ctx_questions, by_name,
            )
            if proposed:
                still_open = []
                proposed_names = {pr["param"] for pr in proposed}
                for q in needs_input:
                    if q["param"] in proposed_names:
                        continue  # moved to resolved
                    still_open.append(q)
                for pr in proposed:
                    resolved[pr["param"]] = pr["value"]
                    sources[pr["param"]] = "expert"
                    expert_suggested.append(pr)
                needs_input = still_open

    # 4. Drop questions whose relevant_when condition is now contradicted by
    #    an already-resolved value (e.g. don't ask about a STAR-only knob once
    #    --aligner=salmon has been pinned, derived, or expert-proposed).
    suppressed = [
        q for q in needs_input
        if not _relevant_when_satisfied(q.get("relevant_when"), resolved, by_name)
    ]
    if suppressed:
        needs_input = [q for q in needs_input if q not in suppressed]

    warning_bits = [
        "All values are starting points and must be reviewed before launching."
    ]
    if expert_suggested:
        warning_bits.append(
            f"{len(expert_suggested)} value(s) were proposed by AI from your experiment "
            f"description — verify each against your protocol."
        )
    if any(q["tier"] == EXPERT_REQUIRED for q in needs_input):
        warning_bits.append(
            "Some parameters require domain expertise and were intentionally left "
            "unset — do not guess these."
        )
    if mode == "expert" and expert_error:
        warning_bits.append(f"Expert mode partially unavailable: {expert_error}")

    return {
        "pipeline": pipeline_name,
        "version": version,
        "mode": mode,
        "resolved_params": resolved,
        "param_sources": sources,
        "auto_derived": auto_derived,
        "expert_suggested": expert_suggested,
        "needs_user_input": needs_input,
        "using_defaults": using_defaults,
        "unresolved_count": len(needs_input),
        "review_required": True,
        "warning": " ".join(warning_bits),
    }


# ===========================================================================
# Schema parsing (rich — captures hidden/min/max beyond get_parameters)
# ===========================================================================

def _parse_rich(schema: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten nextflow_schema.json into rich parameter records."""
    out: list[dict[str, Any]] = []
    definitions = schema.get("definitions", schema.get("$defs", {}))
    required_top: list[str] = schema.get("required", [])

    sections: list[tuple[str, dict[str, Any]]] = []
    for section_ref in schema.get("allOf", []):
        raw_ref = section_ref.get("$ref", "")
        for prefix in ("#/definitions/", "#/$defs/"):
            if raw_ref.startswith(prefix):
                raw_ref = raw_ref[len(prefix):]
                break
        section = definitions.get(raw_ref, {})
        sections.append((section.get("title", raw_ref) or raw_ref, section))

    # Fallback: flat properties at the root.
    if not sections and "properties" in schema:
        sections.append(("Parameters", schema))

    for group_name, section in sections:
        props: dict[str, Any] = section.get("properties", {})
        required_in_group: list[str] = section.get("required", [])
        for raw_name, prop in props.items():
            out.append({
                "name": f"--{raw_name}",
                "raw_name": raw_name,
                "type": prop.get("type", "string"),
                "default": prop.get("default"),
                "description": prop.get("description", "") or "",
                "help_text": prop.get("help_text", "") or "",
                "required": raw_name in required_top or raw_name in required_in_group,
                "allowed_values": prop.get("enum", []) or [],
                "minimum": prop.get("minimum"),
                "maximum": prop.get("maximum"),
                "is_path": prop.get("format", "") in _PATH_FORMATS,
                "hidden": bool(prop.get("hidden", False)),
                "group": group_name,
            })
    return out


# ===========================================================================
# Classification
# ===========================================================================

async def _classify_all(
    params: list[dict[str, Any]],
    pipeline: str,
    version: str,
) -> str:
    """Classify every parameter in place; return the method used.

    Prefers an LLM that reads each parameter's description and help_text
    (capturing embedded expert guidance and conditional dependencies). Caches
    the result for 24h per (pipeline, version). Falls back to the name-based
    heuristic when no API key is set or the call fails.

    Adds these keys to every param record: tier, rationale, expert_guidance,
    relevant_when.
    """
    key = f"param_classification:{pipeline}:{version}"
    cached = cache_get(key, _CLASSIFY_TTL)
    if cached is not None:
        _apply_classification(params, cached.get("classifications", {}))
        return f"{cached.get('method', 'llm')}-cached"

    llm = await _llm_classify(params, pipeline)
    if llm:
        cache_set(key, {"method": "llm", "classifications": llm})
        _apply_classification(params, llm)
        return "llm"

    for p in params:
        tier, rationale = _classify(p)
        p["tier"] = tier
        p["rationale"] = rationale
        p["expert_guidance"] = None
        p["relevant_when"] = None
    return "heuristic"


def _apply_classification(
    params: list[dict[str, Any]],
    classifications: dict[str, Any],
) -> None:
    """Merge an LLM classification onto param records; heuristic-fill any gaps."""
    for p in params:
        c = classifications.get(p["raw_name"])
        if isinstance(c, dict) and c.get("tier") in _TIER_ORDER:
            p["tier"] = c["tier"]
            p["rationale"] = c.get("rationale", "")
            p["expert_guidance"] = c.get("expert_guidance")
            p["relevant_when"] = c.get("relevant_when")
        else:
            tier, rationale = _classify(p)
            p["tier"] = tier
            p["rationale"] = rationale
            p["expert_guidance"] = None
            p["relevant_when"] = None


async def _llm_classify(
    params: list[dict[str, Any]],
    pipeline: str,
) -> dict[str, Any] | None:
    """Classify parameters by having Claude read their descriptions/help_text.

    Returns {raw_name: {tier, rationale, expert_guidance, relevant_when}} or
    None on any failure (no key, import error, parse error, empty result).
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    try:
        import anthropic
    except ImportError:
        return None

    lines = []
    for p in params:
        bits = [f"type={p['type']}"]
        if p["allowed_values"]:
            bits.append(f"enum={p['allowed_values']}")
        if p["minimum"] is not None or p["maximum"] is not None:
            bits.append(f"range={p['minimum']}..{p['maximum']}")
        bits.append(f"default={p['default']!r}")
        meta = ", ".join(bits)
        desc = p["description"] or "(no description)"
        help_t = f" HELP: {p['help_text']}" if p["help_text"] else ""
        lines.append(f"- {p['raw_name']} ({meta}): {desc}{help_t}")
    param_block = "\n".join(lines)

    prompt = f"""You are a bioinformatics expert auditing the parameters of nf-core/{pipeline}.

Classify EVERY parameter below into exactly one tier:

- provided_input: a core input or reference the user supplies/selects (samplesheet,
  output dir, genome key, FASTA/GTF, prebuilt index).
- auto_derivable: can be inferred from the structure of the user's files
  (e.g. single- vs paired-end).
- safe_default: resource/reporting/boilerplate or any knob whose shipped default
  is fine for almost everyone.
- context_dependent: a tool/method choice or numeric threshold whose right value
  depends on the experiment, but for which a sensible starting value or default
  exists.
- expert_required: no safe default exists and choosing a value needs domain or
  published knowledge (e.g. effective genome size, statistical model priors).

Read the description and HELP text carefully. If the text states or implies a
recommended value or rule of thumb (e.g. "typically 2.7e9 for human"), copy that
into expert_guidance. If a parameter is only relevant under a condition (e.g.
only used with a particular aligner, or only when another flag is set), state
that briefly in relevant_when; otherwise null.

Parameters:
{param_block}

Respond with ONLY valid JSON:
{{
  "classifications": {{
    "<param_name_without_dashes>": {{
      "tier": "<one of the five tiers>",
      "rationale": "<short reason>",
      "expert_guidance": "<recommended value/rule of thumb, or null>",
      "relevant_when": "<condition this applies under, or null>"
    }}
  }}
}}"""

    try:
        import json as _json
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model="claude-opus-4-8",
            max_tokens=16000,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = message.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        parsed = _json.loads(raw.strip())
        classifications = parsed.get("classifications", {})
        valid = {
            k: v for k, v in classifications.items()
            if isinstance(v, dict) and v.get("tier") in _TIER_ORDER
        }
        return valid or None
    except Exception as exc:  # noqa: BLE001 — degrade gracefully to heuristic
        logger.warning("LLM classification failed for %s: %s", pipeline, exc)
        return None


def _classify(p: dict[str, Any]) -> tuple[str, str]:
    """Heuristic fallback: assign a tier + rationale from the parameter's name.

    Used only when the LLM classifier is unavailable. Less accurate than reading
    the descriptions, but fast and dependency-free.
    """
    raw = p["raw_name"]
    name_l = raw.lower()
    is_numeric = p["type"] in ("integer", "number")
    has_enum = bool(p["allowed_values"])
    has_default = p["default"] is not None

    # 1. Core provided inputs / references / indexes.
    if name_l in _PROVIDED_INPUT_NAMES or (p["is_path"] and _INDEX_NAME_RE.search(name_l)):
        return PROVIDED_INPUT, "Core input or reference you supply or select."
    if name_l == "genome":
        return PROVIDED_INPUT, "iGenomes reference selection — you choose this for your organism."

    # 2. Derivable from the data structure.
    if _AUTO_DERIVABLE_RE.search(name_l):
        return AUTO_DERIVABLE, "Can be inferred from how your files are structured (read pairing)."

    # 3. Boilerplate / resource / reporting knobs.
    if _BOILERPLATE_RE.search(name_l):
        return SAFE_DEFAULT, "Resource, reporting, or boilerplate option — default is fine."

    # 4. Hidden advanced knobs almost always ship a safe default.
    if p["hidden"] and has_default:
        return SAFE_DEFAULT, "Hidden advanced option with a vetted default."

    # 5. Tool/method choices (enum) are decision points that depend on the experiment.
    if has_enum:
        return CONTEXT_DEPENDENT, (
            f"Choice of method/tool ({', '.join(str(v) for v in p['allowed_values'][:4])}"
            f"{'…' if len(p['allowed_values']) > 4 else ''}) — depends on your experiment."
        )

    # 6. Boolean toggles are off-by-default in nf-core (an absent flag is
    #    false), so they always have a safe default even when the schema omits
    #    an explicit `default: false`. They must never be escalated to
    #    expert_required — leaving a stage toggle at its default needs no domain
    #    expertise. analyze_pipeline_schema separately surfaces meaningful
    #    toggles as decision_points for the user to browse.
    if p["type"] == "boolean":
        return SAFE_DEFAULT, "Boolean toggle — defaults to off; flip it only if you need that stage."

    # 7. Numeric thresholds whose right value depends on the experiment.
    #    With a default or an explicit range the user has a starting point
    #    (context-dependent); with neither, the value must be known up front
    #    (e.g. MACS effective genome size) — that is an expert decision.
    if is_numeric and _THRESHOLD_RE.search(name_l):
        if has_default or p["minimum"] is not None or p["maximum"] is not None:
            return CONTEXT_DEPENDENT, "Numeric threshold whose ideal value depends on your data/experiment."
        return EXPERT_REQUIRED, "Numeric threshold with no default or range — needs an expert value."

    # 8. Undefaulted string whose description reads like a file/reference path.
    #    nf-core often types reference files as bare strings without a `format`;
    #    the description still names them ("Path to dbsnp file", "BWA indices").
    if (
        not has_default
        and p["type"] == "string"
        and not has_enum
        and _PATHISH_DESC_RE.search(p["description"] or "")
    ):
        return PROVIDED_INPUT, "Reference/file input you supply (recognised from its description)."

    # 9. Open-ended values with no default and no obvious heuristic.
    if not has_default and not p["is_path"] and p["type"] != "boolean":
        # Short/empty descriptions on undefaulted params are the hardest cases.
        return EXPERT_REQUIRED, "No default and no general heuristic — needs domain expertise."

    # 10. Numeric with a default but no threshold semantics — usually safe.
    if is_numeric and has_default:
        return SAFE_DEFAULT, "Numeric option with a sensible default."

    # 11. Booleans / defaulted strings fall through to safe default.
    if has_default:
        return SAFE_DEFAULT, "Has a sensible default."

    # 12. Undefaulted path that isn't a core input → optional input.
    if p["is_path"]:
        return PROVIDED_INPUT, "Optional file input you may supply."

    return EXPERT_REQUIRED, "Unclassified with no default — review before setting."


def _decision_points(params: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Identify parameters that branch the analysis into different paths.

    A decision point is either an enum tool/method choice, or a boolean that
    toggles a stage. Booleans with a recognisable prefix (skip_/save_/run_…)
    are obvious toggles, but plenty of branch flags (e.g. narrow_peak) lack one,
    so any non-boilerplate, non-hidden, non-data-derived boolean counts too.
    """
    points: list[dict[str, Any]] = []
    for p in params:
        raw = p["raw_name"]
        if raw in _PROVIDED_INPUT_NAMES:
            continue
        if p["allowed_values"]:
            points.append({
                "param": p["name"],
                "kind": "choice",
                "options": list(p["allowed_values"]),
                "default": p["default"],
                "stage": p["group"],
                "tier": p["tier"],
                "affects": p["description"] or "Selects which tool/method runs.",
            })
        elif (
            p["type"] == "boolean"
            and not _BOILERPLATE_RE.search(raw)
            and not p["hidden"]
            and not _AUTO_DERIVABLE_RE.search(raw)
        ):
            points.append({
                "param": p["name"],
                "kind": "toggle",
                "options": [True, False],
                "default": p["default"] if p["default"] is not None else False,
                "stage": p["group"],
                "tier": p["tier"],
                "affects": p["description"] or "Turns a pipeline stage on/off.",
            })
    return points


# ===========================================================================
# Data-derivation & question building
# ===========================================================================

def _derive_from_data(
    p: dict[str, Any],
    data: dict[str, Any],
) -> dict[str, Any] | None:
    """Return {'value','evidence'} if the parameter can be read off the data."""
    raw = p["raw_name"].lower()
    if not data:
        return None

    if "single_end" in raw and "paired_end" in data:
        paired = bool(data.get("paired_end"))
        return {
            "value": not paired,
            "evidence": (
                "Detected paired R1/R2 files" if paired
                else "No R2 mate files detected — treating as single-end"
            ),
        }
    if "paired_end" in raw and "paired_end" in data:
        return {
            "value": bool(data.get("paired_end")),
            "evidence": "R1/R2 mate pairing detected from file names"
            if data.get("paired_end") else "No paired mates detected",
        }
    return None


def _question_for(p: dict[str, Any]) -> dict[str, Any]:
    """Build a targeted, answerable question for an unresolved parameter."""
    label = p["raw_name"].replace("_", " ")
    why = p["description"] or p["help_text"] or "No description provided in the schema."

    options = list(p["allowed_values"]) if p["allowed_values"] else None
    suggested_range = None
    if p["minimum"] is not None or p["maximum"] is not None:
        suggested_range = {"min": p["minimum"], "max": p["maximum"]}

    if options:
        question = (
            f"Which {label} should nf-core use? "
            f"Options: {', '.join(str(o) for o in options)}."
            + (f" Default: {p['default']}." if p["default"] is not None else "")
        )
    elif p["tier"] == EXPERT_REQUIRED:
        question = (
            f"'{p['name']}' has no safe default and needs an expert decision. "
            f"What value applies to your study?"
        )
    elif p["type"] in ("integer", "number"):
        rng = ""
        if suggested_range:
            rng = f" (allowed {suggested_range['min']}–{suggested_range['max']})"
        question = (
            f"What value for {label}{rng}? This depends on your experiment."
            + (f" Default: {p['default']}." if p["default"] is not None else "")
        )
    else:
        question = (
            f"What value for {label}?"
            + (f" Default: {p['default']}." if p["default"] is not None else "")
        )

    return {
        "param": p["name"],
        "tier": p["tier"],
        "question": question,
        "options": options,
        "suggested_range": suggested_range,
        "default": p["default"],
        "why_it_matters": why,
        "expert_guidance": p.get("expert_guidance"),
        "relevant_when": p.get("relevant_when"),
    }


def _relevant_when_satisfied(
    relevant_when: str | None,
    resolved: dict[str, Any],
    by_name: dict[str, dict[str, Any]],
) -> bool:
    """Best-effort check of whether a free-text relevant_when condition still holds.

    relevant_when is short prose from the LLM classifier, e.g. "Only used when
    --aligner is star_salmon" or "--fasta is not set". We can't truly parse
    arbitrary conditions, so we only suppress a question when we're confident
    it's now moot: an already-resolved parameter is named in the text, that
    parameter has enumerable allowed values, the text names one or more of
    those values, and the resolved value isn't among the named ones. Anything
    we can't confidently evaluate is treated as still relevant — we'd rather
    over-ask than silently hide a question that still applies.
    """
    if not relevant_when:
        return True
    text_l = relevant_when.lower()
    for p in by_name.values():
        flag = p["name"]
        if flag not in resolved:
            continue
        raw_l = p["raw_name"].lower()
        if raw_l not in text_l and flag.lower() not in text_l:
            continue
        candidates = [str(v).lower() for v in (p.get("allowed_values") or [])]
        if not candidates:
            continue
        mentioned = [c for c in candidates if c and c in text_l]
        if mentioned and str(resolved[flag]).lower() not in mentioned:
            return False
    return True


def _normalise_choices(choices: dict[str, Any]) -> dict[str, Any]:
    """Accept both '--param' and 'param' keys; return bare names."""
    out: dict[str, Any] = {}
    for k, v in choices.items():
        out[k[2:] if k.startswith("--") else k] = v
    return out


# ===========================================================================
# Expert mode — Claude-assisted value proposal
# ===========================================================================

async def _expert_fill(
    pipeline_name: str,
    experiment_description: str,
    data_summary: dict[str, Any],
    questions: list[dict[str, Any]],
    by_name: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], str | None]:
    """Use Claude to propose values for context-dependent parameters.

    Returns (proposals, error). proposals is a list of
    {param, value, justification, confidence}. On any failure returns ([], msg)
    so the caller falls back to returning the questions unanswered.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return [], "ANTHROPIC_API_KEY not set"
    if not experiment_description.strip():
        return [], "no experiment_description provided for expert reasoning"

    try:
        import anthropic
    except ImportError:
        return [], "anthropic package not installed"

    lines = []
    for q in questions:
        p = by_name.get(q["param"][2:].replace("-", "_")) or by_name.get(q["param"][2:], {})
        constraint = ""
        if q["options"]:
            constraint = f" allowed: {q['options']}"
        elif q["suggested_range"]:
            constraint = f" range: {q['suggested_range']['min']}–{q['suggested_range']['max']}"
        default = p.get("default") if p else q["default"]
        lines.append(f"- {q['param']} (default {default}){constraint}: {q['why_it_matters']}")
    param_block = "\n".join(lines)

    prompt = f"""You are a bioinformatics expert configuring nf-core/{pipeline_name}.

Experiment:
{experiment_description}

Detected data characteristics:
{data_summary}

Propose values ONLY for the parameters below, and ONLY when the experiment gives
you genuine grounds to deviate from or confirm the default. If you cannot justify
a value from the information given, OMIT that parameter — do not guess.

Parameters needing a decision:
{param_block}

Respond with ONLY valid JSON:
{{
  "proposals": [
    {{"param": "--name", "value": <value>, "justification": "<one sentence tied to the experiment>", "confidence": "high|medium|low"}}
  ]
}}"""

    try:
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model="claude-opus-4-8",
            max_tokens=1500,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = message.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        import json as _json
        parsed = _json.loads(raw.strip())
        proposals = []
        valid_flags = {q["param"] for q in questions}
        for item in parsed.get("proposals", []):
            flag = item.get("param", "")
            if flag in valid_flags and "value" in item:
                proposals.append({
                    "param": flag,
                    "value": item["value"],
                    "justification": item.get("justification", ""),
                    "confidence": item.get("confidence", "medium"),
                })
        return proposals, None
    except Exception as exc:  # noqa: BLE001 — degrade gracefully
        logger.warning("Expert fill failed: %s", exc)
        return [], f"AI call failed: {exc}"
