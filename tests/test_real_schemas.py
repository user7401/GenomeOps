"""Validation against REAL nf-core schemas (committed fixtures).

These tests are deliberately anchored to authentic `nextflow_schema.json` files
we did not author (see tests/fixtures/schemas/SOURCES.md). Synthetic schemas can
make the parser/classifier look correct by construction; real schemas cannot.

They verify two distinct layers:
  * Facts  — _parse_rich must extract every parameter and preserve its default
             and enum exactly (no loss, no corruption). This is the
             deterministic transport layer and must be perfect.
  * Judgment — the heuristic classifier must produce sane tiers (e.g. boolean
             toggles are never escalated to expert_required, core inputs are
             provided_input, the expert fraction stays sane). This is the
             fallback used when no LLM key is present.
"""

import json
from pathlib import Path

import httpx
import pytest
import respx

from genomeops_mcp.cache import cache_clear
from genomeops_mcp.tools.configuration import (
    configure_parameters,
    _parse_rich,
    _classify,
    PROVIDED_INPUT,
    CONTEXT_DEPENDENT,
    EXPERT_REQUIRED,
    _TIER_ORDER,
)

FIXTURES = Path(__file__).parent / "fixtures" / "schemas"

# (fixture file, pipeline, version, convention)
SCHEMAS = [
    ("rnaseq_3.14.0.json", "rnaseq", "3.14.0", "definitions"),
    ("sarek_3.5.1.json", "sarek", "3.5.1", "$defs"),
    ("demo_1.0.1.json", "demo", "1.0.1", "$defs"),
    ("chipseq_2.0.0.json", "chipseq", "2.0.0", "definitions"),
]


@pytest.fixture(autouse=True)
def clear_cache():
    cache_clear()
    yield
    cache_clear()


def _load(fname: str) -> dict:
    return json.loads((FIXTURES / fname).read_text())


def _source_params(schema: dict) -> dict[str, dict]:
    """Flatten the raw schema the way nf-core authors it (ground truth)."""
    src: dict[str, dict] = {}
    defs = schema.get("definitions", schema.get("$defs", {}))
    for _group, gdef in defs.items():
        for pname, pdef in gdef.get("properties", {}).items():
            src[pname] = pdef
    for pname, pdef in schema.get("properties", {}).items():  # top-level, if any
        src[pname] = pdef
    return src


# ---------------------------------------------------------------------------
# Facts layer — parsing must be lossless and exact
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fname,_p,_v,_conv", SCHEMAS)
def test_parse_extracts_every_parameter(fname, _p, _v, _conv):
    schema = _load(fname)
    src = _source_params(schema)
    parsed = {p["raw_name"] for p in _parse_rich(schema)}
    missing = set(src) - parsed
    assert not missing, f"{fname}: parser dropped {missing}"
    assert len(parsed) == len(src), f"{fname}: count mismatch"


@pytest.mark.parametrize("fname,_p,_v,_conv", SCHEMAS)
def test_parse_preserves_defaults_and_enums(fname, _p, _v, _conv):
    schema = _load(fname)
    src = _source_params(schema)
    for p in _parse_rich(schema):
        s = src[p["raw_name"]]
        assert s.get("default") == p["default"], (
            f"{fname}:{p['raw_name']} default corrupted "
            f"{s.get('default')!r} -> {p['default']!r}"
        )
        assert list(s.get("enum", [])) == list(p["allowed_values"] or []), (
            f"{fname}:{p['raw_name']} enum corrupted"
        )


def test_both_schema_conventions_are_covered():
    conventions = set()
    for fname, _p, _v, _conv in SCHEMAS:
        schema = _load(fname)
        conventions.add("definitions" if "definitions" in schema else "$defs")
        assert _parse_rich(schema), f"{fname}: parsed empty"
    assert conventions == {"definitions", "$defs"}, (
        f"both conventions must be represented, got {conventions}"
    )


@pytest.mark.parametrize("fname,_p,_v,_conv", SCHEMAS)
def test_every_param_gets_a_valid_tier(fname, _p, _v, _conv):
    schema = _load(fname)
    for p in _parse_rich(schema):
        tier, _ = _classify(p)
        assert tier in _TIER_ORDER, f"{fname}:{p['raw_name']} -> invalid tier {tier}"


# ---------------------------------------------------------------------------
# Judgment layer — heuristic classification sanity on real schemas
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fname,_p,_v,_conv", SCHEMAS)
def test_no_boolean_toggle_is_expert_required(fname, _p, _v, _conv):
    """Regression: undefaulted boolean toggles (skip_*/save_*/with_*) must never
    be escalated to expert_required. Real schemas omit `default: false`, which
    previously dumped ~30 booleans per pipeline into the expert tier."""
    schema = _load(fname)
    offenders = []
    for p in _parse_rich(schema):
        tier, _ = _classify(p)
        if p["type"] == "boolean" and tier == EXPERT_REQUIRED:
            offenders.append(p["raw_name"])
    assert not offenders, f"{fname}: booleans wrongly marked expert: {offenders}"


@pytest.mark.parametrize("fname", ["rnaseq_3.14.0.json", "sarek_3.5.1.json"])
def test_core_inputs_are_provided_input(fname):
    schema = _load(fname)
    by = {p["raw_name"]: _classify(p)[0] for p in _parse_rich(schema)}
    assert by.get("input") == PROVIDED_INPUT
    assert by.get("outdir") == PROVIDED_INPUT


def test_rnaseq_references_are_provided_input():
    by = {p["raw_name"]: _classify(p)[0] for p in _parse_rich(_load("rnaseq_3.14.0.json"))}
    for ref in ("genome", "fasta", "gtf"):
        assert by.get(ref) == PROVIDED_INPUT, f"{ref} should be provided_input"


def test_rnaseq_tool_choices_are_decision_points():
    """aligner/trimmer are enum tool choices — they must surface as
    context_dependent decisions, not get buried in safe_default."""
    by = {p["raw_name"]: _classify(p)[0] for p in _parse_rich(_load("rnaseq_3.14.0.json"))}
    assert by.get("aligner") == CONTEXT_DEPENDENT
    assert by.get("trimmer") == CONTEXT_DEPENDENT


@pytest.mark.parametrize("fname,max_frac", [
    ("rnaseq_3.14.0.json", 0.20),
    ("sarek_3.5.1.json", 0.25),
    ("demo_1.0.1.json", 0.10),
])
def test_expert_fraction_is_sane(fname, max_frac):
    """Guards against the over-escalation regression: most parameters of a real
    pipeline are NOT expert decisions. If this fraction balloons, the classifier
    is dumping ordinary knobs into expert_required again."""
    params = _parse_rich(_load(fname))
    expert = sum(1 for p in params if _classify(p)[0] == EXPERT_REQUIRED)
    frac = expert / len(params)
    assert frac <= max_frac, f"{fname}: {frac:.0%} expert_required exceeds {max_frac:.0%}"


def test_file_path_pattern_format_recognized():
    """sarek's known_indels uses format 'file-path-pattern' — it must be parsed
    as a path and classified as a provided input, not an opaque string."""
    by = {p["raw_name"]: p for p in _parse_rich(_load("sarek_3.5.1.json"))}
    ki = by.get("known_indels")
    assert ki is not None
    assert ki["is_path"] is True
    assert _classify(ki)[0] == PROVIDED_INPUT


# ---------------------------------------------------------------------------
# End-to-end on a real schema (no API key → heuristic path), network mocked
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_configure_parameters_end_to_end_real_rnaseq(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    schema = _load("rnaseq_3.14.0.json")
    url = "https://raw.githubusercontent.com/nf-core/rnaseq/3.14.0/nextflow_schema.json"

    with respx.mock(assert_all_called=False) as mock:
        mock.get(url).mock(return_value=httpx.Response(200, json=schema))
        result = await configure_parameters("rnaseq", version="3.14.0")

    assert "error" not in result
    assert result["review_required"] is True
    # The gap-#3 fix on REAL data: the most consequential reference choices are
    # always surfaced, never silently dropped or defaulted.
    asked = {q["param"] for q in result["needs_user_input"]}
    assert "--input" in asked
    assert "--outdir" in asked
    assert "--genome" in asked
    assert "--fasta" in asked
    # The bulk of a real pipeline is handled for the user: the questions put to
    # them must be a clear minority of the ~109 parameters, not most of them.
    total_params = len(_parse_rich(schema))
    assert len(result["needs_user_input"]) < total_params * 0.4
    assert result["using_defaults"], "some params should resolve to explicit defaults"
