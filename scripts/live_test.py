"""
Live integration test against real nf-co.re API and GitHub.
Run with: uv run python scripts/live_test.py
"""
import asyncio
import json
import sys
import traceback


async def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


async def ok(label: str, value=None):
    msg = f"  [PASS] {label}"
    if value is not None:
        preview = str(value)[:120]
        msg += f"\n         {preview}"
    print(msg)


async def fail(label: str, exc=None, value=None):
    msg = f"  [FAIL] {label}"
    if value is not None:
        msg += f"\n         {str(value)[:200]}"
    if exc is not None:
        msg += f"\n         {type(exc).__name__}: {exc}"
    print(msg)


async def test_list_pipelines():
    await section("1. list_pipelines() — real API")
    from nfcore_mcp.tools.discovery import list_pipelines

    try:
        result = await list_pipelines()
        if result.get("error"):
            await fail("list_pipelines() returned error", value=result)
            return result
        total = result["total"]
        await ok(f"Returned {total} pipelines")
        first = result["pipelines"][0]
        missing_keys = [k for k in ("name", "description", "topics", "latest_version", "url") if k not in first]
        if missing_keys:
            await fail(f"Missing keys in pipeline: {missing_keys}", value=first)
        else:
            await ok("Pipeline dict has all required keys", value=first)
        return result
    except Exception as e:
        await fail("list_pipelines() raised exception", exc=e)
        traceback.print_exc()
        return {}


async def test_list_pipelines_filtered(all_pipelines):
    await section("2. list_pipelines(topic='RNA-seq') — filter")
    from nfcore_mcp.tools.discovery import list_pipelines

    try:
        result = await list_pipelines(topic="RNA-seq")
        if result.get("error"):
            await fail("Filtered list returned error", value=result)
            return
        names = [p["name"] for p in result["pipelines"]]
        if "rnaseq" in names:
            await ok(f"rnaseq in filtered results ({result['total']} pipelines)", value=names[:5])
        else:
            await fail("rnaseq NOT found in RNA-seq filter", value=names[:10])
    except Exception as e:
        await fail("list_pipelines(topic=...) raised exception", exc=e)


async def test_get_pipeline_info():
    await section("3. get_pipeline_info('rnaseq') — real API")
    from nfcore_mcp.tools.discovery import get_pipeline_info

    try:
        result = await get_pipeline_info("rnaseq")
        if result.get("error"):
            await fail("get_pipeline_info returned error", value=result)
            return result
        for key in ("name", "description", "latest_version", "url", "docs_url",
                    "schema_url", "input_schema_url", "supported_profiles", "required_params"):
            if key in result:
                await ok(f"Has '{key}'", value=result[key])
            else:
                await fail(f"Missing key '{key}'", value=list(result.keys()))
        return result
    except Exception as e:
        await fail("get_pipeline_info raised exception", exc=e)
        traceback.print_exc()
        return {}


async def test_pipeline_not_found():
    await section("4. get_pipeline_info('rnseq') — typo correction")
    from nfcore_mcp.tools.discovery import get_pipeline_info

    try:
        result = await get_pipeline_info("rnseq")
        if result.get("error") and result.get("code") == "PIPELINE_NOT_FOUND":
            if "rnaseq" in result.get("message", "").lower():
                await ok("Typo corrected — suggested 'rnaseq'", value=result["message"])
            else:
                await fail("PIPELINE_NOT_FOUND but no suggestion", value=result["message"])
        else:
            await fail("Expected PIPELINE_NOT_FOUND error", value=result)
    except Exception as e:
        await fail("Typo test raised exception", exc=e)


async def test_samplesheet_schema(pipeline_info: dict):
    await section("5. get_samplesheet_schema('rnaseq') — real GitHub")
    from nfcore_mcp.tools.samplesheet import get_samplesheet_schema

    version = pipeline_info.get("latest_version", "main")
    try:
        result = await get_samplesheet_schema("rnaseq", version)
        if result.get("error"):
            await fail("get_samplesheet_schema returned error", value=result)
            # Try with 'main' as fallback
            result = await get_samplesheet_schema("rnaseq", "main")
            if result.get("error"):
                await fail("Fallback to 'main' also failed", value=result)
                return result

        fields = result.get("fields", [])
        await ok(f"Returned {len(fields)} fields", value=[f["name"] for f in fields])
        required = [f for f in fields if f["required"]]
        await ok(f"{len(required)} required fields", value=[f["name"] for f in required])
        for f in fields:
            if f.get("allowed_values"):
                await ok(f"Field '{f['name']}' has enum values", value=f["allowed_values"])
        return result
    except Exception as e:
        await fail("get_samplesheet_schema raised exception", exc=e)
        traceback.print_exc()
        return {}


async def test_validate_samplesheet(schema_result: dict):
    await section("6. validate_samplesheet — valid + invalid CSV")
    from nfcore_mcp.tools.samplesheet import validate_samplesheet

    # Get required columns from real schema
    fields = schema_result.get("fields", [])
    required = [f["name"] for f in fields if f["required"]]
    print(f"  Required columns from real schema: {required}")

    # Build a valid-looking samplesheet based on real schema
    if "strandedness" in required:
        valid_csv = "sample,fastq_1,fastq_2,strandedness\nctrl_1,/data/ctrl_1_R1.fastq.gz,/data/ctrl_1_R2.fastq.gz,forward\n"
        bad_csv = "sample,fastq_1,fastq_2,strandedness\nctrl_1,/data/ctrl_1_R1.fastq.gz,/data/ctrl_1_R2.fastq.gz,fwd\n"
    else:
        valid_csv = "sample,fastq_1,fastq_2\nctrl_1,/data/ctrl_1_R1.fastq.gz,/data/ctrl_1_R2.fastq.gz\n"
        bad_csv = valid_csv  # can't test bad values without enum

    try:
        result = await validate_samplesheet("rnaseq", valid_csv)
        if result.get("error"):
            await fail("validate_samplesheet (valid CSV) returned error", value=result)
        elif result.get("valid"):
            await ok("Valid CSV accepted", value=result)
        else:
            await fail("Valid CSV rejected — unexpected errors", value=result["errors"])
    except Exception as e:
        await fail("validate_samplesheet (valid) raised exception", exc=e)
        traceback.print_exc()

    if "strandedness" in required:
        try:
            result = await validate_samplesheet("rnaseq", bad_csv)
            if result.get("error"):
                await fail("validate_samplesheet (bad CSV) returned error", value=result)
            elif not result.get("valid") and result.get("errors"):
                err = result["errors"][0]
                await ok("Invalid value correctly rejected", value=err)
                if "fix_suggestion" in err and err["fix_suggestion"]:
                    await ok("fix_suggestion populated", value=err["fix_suggestion"])
                else:
                    await fail("fix_suggestion empty or missing", value=err)
            else:
                await fail("Bad CSV was not rejected", value=result)
        except Exception as e:
            await fail("validate_samplesheet (bad) raised exception", exc=e)
            traceback.print_exc()


async def test_generate_samplesheet():
    await section("7. generate_samplesheet — file pairing")
    from nfcore_mcp.tools.samplesheet import generate_samplesheet

    files = [
        "/data/batch1/sample_A_R1_001.fastq.gz",
        "/data/batch1/sample_A_R2_001.fastq.gz",
        "/data/batch1/sample_B_R1_001.fastq.gz",
        "/data/batch1/sample_B_R2_001.fastq.gz",
        "/data/batch2/sample_C_R1.fastq.gz",
        "/data/batch2/sample_C_R2.fastq.gz",
    ]
    try:
        result = await generate_samplesheet("rnaseq", files)
        if result.get("error"):
            await fail("generate_samplesheet returned error", value=result)
            return
        csv = result.get("samplesheet", "")
        lines = csv.strip().split("\n")
        await ok(f"Generated {len(lines)-1} sample rows (+ header)", value=csv)
        if result.get("review_required"):
            await ok("review_required=true set")
        else:
            await fail("review_required not set")
        if result.get("warnings"):
            await ok(f"{len(result['warnings'])} warnings", value=result["warnings"])
    except Exception as e:
        await fail("generate_samplesheet raised exception", exc=e)
        traceback.print_exc()


async def test_get_parameters():
    await section("8. get_parameters('rnaseq') — real nextflow_schema.json")
    from nfcore_mcp.tools.parameters import get_parameters

    try:
        result = await get_parameters("rnaseq")
        if result.get("error"):
            await fail("get_parameters returned error", value=result)
            return result
        groups = result.get("groups", {})
        await ok(f"Returned {len(groups)} parameter groups", value=list(groups.keys()))
        total_params = sum(len(v) for v in groups.values())
        await ok(f"Total {total_params} parameters across all groups")

        all_params = [p for g in groups.values() for p in g]
        required = [p for p in all_params if p["required"]]
        await ok(f"{len(required)} required parameters", value=[p["name"] for p in required])

        with_enum = [p for p in all_params if p["allowed_values"]]
        if with_enum:
            await ok(f"{len(with_enum)} parameters have enum constraints", value=with_enum[0])

        # Check for known params
        names = [p["name"] for p in all_params]
        for expected in ("--input", "--outdir", "--genome"):
            if expected in names:
                await ok(f"Found expected param '{expected}'")
            else:
                await fail(f"Expected param '{expected}' not found", value=names[:20])

        return result
    except Exception as e:
        await fail("get_parameters raised exception", exc=e)
        traceback.print_exc()
        return {}


async def test_get_parameters_group_filter():
    await section("9. get_parameters(group='reference') — filter")
    from nfcore_mcp.tools.parameters import get_parameters

    try:
        result = await get_parameters("rnaseq", group="reference")
        if result.get("error"):
            await fail("get_parameters(group=reference) returned error", value=result)
            return
        groups = result.get("groups", {})
        if groups:
            await ok(f"Group filter returned {len(groups)} group(s)", value=list(groups.keys()))
        else:
            await fail("Group filter returned no groups — check group name matching")
    except Exception as e:
        await fail("get_parameters(group=...) raised exception", exc=e)


async def test_check_feasibility():
    await section("10. check_feasibility — real API + keyword scoring")
    from nfcore_mcp.tools.feasibility import check_feasibility

    cases = [
        (
            "differential gene expression in human cancer cell lines, paired-end stranded RNA-seq",
            [
                "/project/data/ctrl_rep1_R1_001.fastq.gz",
                "/project/data/ctrl_rep1_R2_001.fastq.gz",
                "/project/data/ctrl_rep2_R1_001.fastq.gz",
                "/project/data/ctrl_rep2_R2_001.fastq.gz",
                "/project/data/treat_rep1_R1_001.fastq.gz",
                "/project/data/treat_rep1_R2_001.fastq.gz",
            ],
            "human",
        ),
        (
            "somatic variant calling from matched tumour/normal whole genome sequencing data",
            [
                "/bam/tumor_sample.bam",
                "/bam/tumor_sample.bai",
                "/bam/normal_sample.bam",
                "/bam/normal_sample.bai",
            ],
            None,
        ),
        (
            "16S amplicon microbiome profiling of gut samples",
            [
                "/amplicon/sample1_R1.fastq.gz",
                "/amplicon/sample1_R2.fastq.gz",
                "/amplicon/sample2_R1.fastq.gz",
                "/amplicon/sample2_R2.fastq.gz",
            ],
            None,
        ),
    ]

    for goal, files, organism in cases:
        print(f"\n  Goal: {goal[:70]}...")
        try:
            result = await check_feasibility(goal, files, organism=organism)
            if result.get("error"):
                await fail("check_feasibility returned error", value=result)
                continue

            feasibility = result["overall_feasibility"]
            matched = result["matched_pipelines"]
            ds = result["data_summary"]

            print(f"  Data: {ds['data_type']}, {ds['sample_count']} samples, paired={ds['paired_end']}")
            print(f"  Feasibility: {feasibility} — {result['feasibility_summary']}")

            if matched:
                top = matched[0]
                await ok(
                    f"Top match: {top['name']} (confidence={top['confidence']})",
                    value=top["match_reason"],
                )
                if top["sufficient"]:
                    print(f"  Sufficient: {top['sufficient']}")
                if top["missing"]:
                    print(f"  Missing:    {top['missing']}")
                print(f"  Next steps: {top['next_steps'][:2]}")
            else:
                await fail("No pipelines matched")
        except Exception as e:
            await fail(f"check_feasibility raised exception for goal: {goal[:40]}", exc=e)
            traceback.print_exc()


async def test_sarek_schema():
    await section("11. get_samplesheet_schema('sarek') — different pipeline")
    from nfcore_mcp.tools.samplesheet import get_samplesheet_schema

    try:
        result = await get_samplesheet_schema("sarek")
        if result.get("error"):
            await fail("sarek schema returned error", value=result)
        else:
            fields = result.get("fields", [])
            await ok(f"sarek schema has {len(fields)} fields", value=[f["name"] for f in fields])
    except Exception as e:
        await fail("sarek schema raised exception", exc=e)


async def main():
    print("\nnfcore-mcp live integration test")
    print(f"Testing against real nf-co.re API + GitHub\n")

    from nfcore_mcp.cache import cache_clear
    cache_clear()

    all_pipelines = await test_list_pipelines()
    await test_list_pipelines_filtered(all_pipelines)
    pipeline_info = await test_get_pipeline_info()
    await test_pipeline_not_found()
    schema_result = await test_samplesheet_schema(pipeline_info)
    await test_validate_samplesheet(schema_result)
    await test_generate_samplesheet()
    await test_get_parameters()
    await test_get_parameters_group_filter()
    await test_check_feasibility()
    await test_sarek_schema()

    print(f"\n{'='*60}")
    print("  Done.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    asyncio.run(main())
