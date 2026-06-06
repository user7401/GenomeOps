"""Tests for the parameter configuration framework.

Covers analyze_pipeline_schema (decision-tree mapping + tier classification)
and configure_parameters (user-choice precedence, data derivation, expert mode,
and the needs_user_input elicitation path).
"""

import httpx
import pytest
import respx

from nfcore_mcp.cache import cache_clear
from nfcore_mcp.tools.configuration import (
    analyze_pipeline_schema,
    configure_parameters,
    _classify,
    _parse_rich,
    PROVIDED_INPUT,
    AUTO_DERIVABLE,
    SAFE_DEFAULT,
    CONTEXT_DEPENDENT,
    EXPERT_REQUIRED,
)


# A schema exercising every tier and both kinds of decision point.
RICH_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema",
    "title": "nf-core/demo pipeline parameters",
    "definitions": {
        "input_output_options": {
            "title": "Input/output options",
            "type": "object",
            "required": ["input", "outdir"],
            "properties": {
                "input": {"type": "string", "format": "file-path",
                          "description": "Samplesheet CSV."},
                "outdir": {"type": "string", "format": "directory-path",
                           "description": "Output dir."},
            },
        },
        "reference_genome_options": {
            "title": "Reference genome options",
            "type": "object",
            "properties": {
                "genome": {"type": "string", "enum": ["GRCh38", "GRCm38"],
                           "description": "iGenomes key."},
                "fasta": {"type": "string", "format": "file-path",
                          "description": "Genome FASTA."},
            },
        },
        "read_options": {
            "title": "Read options",
            "type": "object",
            "properties": {
                "single_end": {"type": "boolean", "default": False,
                               "description": "Single-end reads."},
            },
        },
        "alignment_options": {
            "title": "Alignment options",
            "type": "object",
            "properties": {
                "aligner": {"type": "string", "enum": ["star_salmon", "hisat2"],
                            "default": "star_salmon", "description": "Aligner."},
                "skip_markduplicates": {"type": "boolean", "default": False,
                                        "description": "Skip dedup."},
                "min_mapped_reads": {"type": "integer", "default": 5,
                                     "minimum": 0, "maximum": 100,
                                     "description": "Minimum mapped read %."},
            },
        },
        "peak_options": {
            "title": "Peak calling options",
            "type": "object",
            "properties": {
                "macs_gsize": {"type": "number",
                               "description": "Effective genome size for MACS2."},
                "narrow_peak": {"type": "boolean", "default": False,
                                "description": "Call narrow peaks."},
            },
        },
        "generic_options": {
            "title": "Generic options",
            "type": "object",
            "properties": {
                "max_cpus": {"type": "integer", "default": 16,
                             "description": "Max CPUs.", "hidden": True},
                "publish_dir_mode": {"type": "string", "default": "copy",
                                     "description": "Publish mode.", "hidden": True},
                "email": {"type": "string", "description": "Notification email."},
            },
        },
    },
    "allOf": [
        {"$ref": "#/definitions/input_output_options"},
        {"$ref": "#/definitions/reference_genome_options"},
        {"$ref": "#/definitions/read_options"},
        {"$ref": "#/definitions/alignment_options"},
        {"$ref": "#/definitions/peak_options"},
        {"$ref": "#/definitions/generic_options"},
    ],
}

SCHEMA_URL = "https://raw.githubusercontent.com/nf-core/demo/master/nextflow_schema.json"


@pytest.fixture(autouse=True)
def clear_cache():
    cache_clear()
    yield
    cache_clear()


@pytest.fixture
def mock_demo_schema():
    with respx.mock(assert_all_called=False) as mock:
        mock.get(SCHEMA_URL).mock(return_value=httpx.Response(200, json=RICH_SCHEMA))
        mock.get(
            "https://raw.githubusercontent.com/nf-core/notexist/master/nextflow_schema.json"
        ).mock(return_value=httpx.Response(404))
        yield mock


def _by_name(params):
    return {p["raw_name"]: p for p in params}


# ---------------------------------------------------------------------------
# Classification — pure, no network
# ---------------------------------------------------------------------------

def test_classify_covers_every_tier():
    params = _parse_rich(RICH_SCHEMA)
    tiers = {p["raw_name"]: _classify(p)[0] for p in params}

    assert tiers["input"] == PROVIDED_INPUT
    assert tiers["outdir"] == PROVIDED_INPUT
    assert tiers["genome"] == PROVIDED_INPUT
    assert tiers["fasta"] == PROVIDED_INPUT
    assert tiers["single_end"] == AUTO_DERIVABLE
    assert tiers["aligner"] == CONTEXT_DEPENDENT          # enum tool choice
    assert tiers["min_mapped_reads"] == CONTEXT_DEPENDENT  # numeric threshold
    assert tiers["macs_gsize"] == EXPERT_REQUIRED         # numeric, no default
    assert tiers["max_cpus"] == SAFE_DEFAULT              # boilerplate / hidden
    assert tiers["publish_dir_mode"] == SAFE_DEFAULT


def test_parse_rich_captures_min_max_hidden():
    by = _by_name(_parse_rich(RICH_SCHEMA))
    assert by["min_mapped_reads"]["minimum"] == 0
    assert by["min_mapped_reads"]["maximum"] == 100
    assert by["max_cpus"]["hidden"] is True
    assert by["aligner"]["allowed_values"] == ["star_salmon", "hisat2"]


# ---------------------------------------------------------------------------
# analyze_pipeline_schema
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_analyze_returns_decision_points(mock_demo_schema):
    result = await analyze_pipeline_schema("demo")
    params_by = {dp["param"] for dp in result["decision_points"]}

    # enum choices and skip toggles are branch points; genome (provided input) is not
    assert "--aligner" in params_by
    assert "--skip_markduplicates" in params_by
    assert "--narrow_peak" in params_by
    assert "--genome" not in params_by

    aligner = next(dp for dp in result["decision_points"] if dp["param"] == "--aligner")
    assert aligner["kind"] == "choice"
    assert aligner["options"] == ["star_salmon", "hisat2"]

    skip = next(dp for dp in result["decision_points"] if dp["param"] == "--skip_markduplicates")
    assert skip["kind"] == "toggle"
    assert skip["options"] == [True, False]


@pytest.mark.asyncio
async def test_analyze_path_estimate_is_product(mock_demo_schema):
    result = await analyze_pipeline_schema("demo")
    # aligner(2) * skip_markduplicates(2) * narrow_peak(2) = 8
    assert result["path_count_estimate"] == 8


@pytest.mark.asyncio
async def test_analyze_buckets_params_by_tier(mock_demo_schema):
    result = await analyze_pipeline_schema("demo")
    assert CONTEXT_DEPENDENT in result["tiers"]
    assert EXPERT_REQUIRED in result["tiers"]
    expert_names = {p["name"] for p in result["tiers"][EXPERT_REQUIRED]["params"]}
    assert "--macs_gsize" in expert_names


@pytest.mark.asyncio
async def test_analyze_schema_not_found(mock_demo_schema):
    result = await analyze_pipeline_schema("notexist")
    assert result.get("error") is True
    assert result["code"] == "SCHEMA_NOT_FOUND"


# ---------------------------------------------------------------------------
# configure_parameters — user choices, derivation, elicitation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_user_choices_win(mock_demo_schema):
    result = await configure_parameters(
        "demo",
        user_choices={"aligner": "hisat2", "--min_mapped_reads": 10},
    )
    assert result["resolved_params"]["--aligner"] == "hisat2"
    assert result["resolved_params"]["--min_mapped_reads"] == 10
    assert result["param_sources"]["--aligner"] == "user"
    # a user-pinned context param must not also be asked about
    asked = {q["param"] for q in result["needs_user_input"]}
    assert "--aligner" not in asked
    assert "--min_mapped_reads" not in asked


@pytest.mark.asyncio
async def test_auto_derives_single_end_from_data(mock_demo_schema):
    result = await configure_parameters(
        "demo",
        data_summary={"paired_end": True},
    )
    assert result["resolved_params"].get("--single_end") is False
    assert result["param_sources"]["--single_end"] == "derived"
    derived = {d["param"] for d in result["auto_derived"]}
    assert "--single_end" in derived


@pytest.mark.asyncio
async def test_context_and_expert_become_questions(mock_demo_schema):
    result = await configure_parameters("demo")
    asked = {q["param"]: q for q in result["needs_user_input"]}

    # context-dependent numeric threshold is asked with its range
    assert "--min_mapped_reads" in asked
    assert asked["--min_mapped_reads"]["suggested_range"] == {"min": 0, "max": 100}

    # enum choice surfaces its options
    assert asked["--aligner"]["options"] == ["star_salmon", "hisat2"]

    # expert-required param is flagged and never auto-set
    assert "--macs_gsize" in asked
    assert asked["--macs_gsize"]["tier"] == EXPERT_REQUIRED
    assert "--macs_gsize" not in result["resolved_params"]


@pytest.mark.asyncio
async def test_safe_defaults_listed_not_asked(mock_demo_schema):
    result = await configure_parameters("demo")
    default_params = {d["param"] for d in result["using_defaults"]}
    asked = {q["param"] for q in result["needs_user_input"]}
    assert "--max_cpus" in default_params
    assert "--max_cpus" not in asked


@pytest.mark.asyncio
async def test_required_inputs_are_asked(mock_demo_schema):
    result = await configure_parameters("demo")
    asked = {q["param"] for q in result["needs_user_input"]}
    assert "--input" in asked
    assert "--outdir" in asked


@pytest.mark.asyncio
async def test_review_required_always_true(mock_demo_schema):
    result = await configure_parameters("demo")
    assert result["review_required"] is True


@pytest.mark.asyncio
async def test_expert_mode_without_key_degrades(mock_demo_schema, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    result = await configure_parameters(
        "demo",
        experiment_description="Human ChIP-seq for H3K27ac",
        mode="expert",
    )
    # No key → no expert suggestions, questions remain, warning explains why
    assert result["expert_suggested"] == []
    assert any(q["param"] == "--min_mapped_reads" for q in result["needs_user_input"])
    assert "partially unavailable" in result["warning"]


@pytest.mark.asyncio
async def test_expert_mode_fills_context_params(mock_demo_schema, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    class _FakeMessage:
        content = [type("C", (), {"text":
            '{"proposals": [{"param": "--min_mapped_reads", "value": 20, '
            '"justification": "Low-input ChIP needs a higher mapped-read floor", '
            '"confidence": "medium"}]}'})()]

    class _FakeClient:
        def __init__(self, *a, **k):
            self.messages = self

        def create(self, *a, **k):
            return _FakeMessage()

    import anthropic
    monkeypatch.setattr(anthropic, "Anthropic", _FakeClient)

    result = await configure_parameters(
        "demo",
        experiment_description="Low-input ChIP-seq, H3K27ac, human",
        mode="expert",
    )

    assert result["resolved_params"].get("--min_mapped_reads") == 20
    assert result["param_sources"]["--min_mapped_reads"] == "expert"
    suggested = {s["param"] for s in result["expert_suggested"]}
    assert "--min_mapped_reads" in suggested
    # expert mode must NOT have filled the expert_required param
    assert "--macs_gsize" not in result["resolved_params"]
    assert any(q["param"] == "--macs_gsize" for q in result["needs_user_input"])


@pytest.mark.asyncio
async def test_configure_schema_not_found(mock_demo_schema):
    result = await configure_parameters("notexist")
    assert result.get("error") is True
    assert result["code"] == "SCHEMA_NOT_FOUND"
