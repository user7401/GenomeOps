"""
Live smoke test against the real nf-co.re API and GitHub.

Covers only the faithful-plumbing tool surface (no parameter intelligence,
feasibility, QC, or methods tools — those were removed by design).

Run with: uv run python scripts/live_test.py
"""
import asyncio
import traceback


async def section(title: str):
    print(f"\n{'='*60}\n  {title}\n{'='*60}")


async def ok(label: str, value=None):
    msg = f"  [PASS] {label}"
    if value is not None:
        msg += f"\n         {str(value)[:120]}"
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
    from genomeops_mcp.tools.discovery import list_pipelines
    try:
        result = await list_pipelines()
        if result.get("error"):
            await fail("list_pipelines() returned error", value=result)
            return {}
        await ok(f"Returned {result['total']} pipelines")
        first = result["pipelines"][0]
        missing = [k for k in ("name", "description", "topics", "latest_version", "url") if k not in first]
        if missing:
            await fail(f"Missing keys in pipeline: {missing}", value=first)
        else:
            await ok("Pipeline dict has all required keys", value=first)
        return result
    except Exception as e:
        await fail("list_pipelines() raised exception", exc=e)
        traceback.print_exc()
        return {}


async def test_list_pipelines_filtered():
    await section("2. list_pipelines(topic='rna') — filter")
    from genomeops_mcp.tools.discovery import list_pipelines
    try:
        result = await list_pipelines(topic="rna")
        if result.get("error"):
            await fail("Filtered list returned error", value=result)
            return
        names = [p["name"] for p in result["pipelines"]]
        await ok(f"{result['total']} pipelines matched", value=names[:5])
    except Exception as e:
        await fail("filtered list raised exception", exc=e)


async def test_get_pipeline_info():
    await section("3. get_pipeline_info('rnaseq')")
    from genomeops_mcp.tools.discovery import get_pipeline_info
    try:
        result = await get_pipeline_info("rnaseq")
        if result.get("error"):
            await fail("get_pipeline_info returned error", value=result)
            return
        await ok("schema_url present", value=result.get("schema_url"))
    except Exception as e:
        await fail("get_pipeline_info raised exception", exc=e)


async def test_get_latest_version():
    await section("4. get_latest_version('rnaseq')")
    from genomeops_mcp.tools.versioning import get_latest_version
    try:
        result = await get_latest_version("rnaseq")
        if result.get("error"):
            await fail("get_latest_version returned error (release API may be blocked)", value=result)
            return
        await ok("resolved a version", value={k: result.get(k) for k in ("recommended", "stable", "is_dev")})
    except Exception as e:
        await fail("get_latest_version raised exception", exc=e)


async def test_get_samplesheet_schema():
    await section("5. get_samplesheet_schema('rnaseq')")
    from genomeops_mcp.tools.samplesheet import get_samplesheet_schema
    try:
        result = await get_samplesheet_schema("rnaseq")
        if result.get("error"):
            await fail("get_samplesheet_schema returned error", value=result)
            return None
        fields = [f["name"] for f in result["fields"]]
        await ok(f"{len(fields)} fields", value=fields)
        return result
    except Exception as e:
        await fail("get_samplesheet_schema raised exception", exc=e)
        return None


async def test_generate_and_validate_samplesheet():
    await section("6. generate_samplesheet + validate_samplesheet")
    from genomeops_mcp.tools.samplesheet import generate_samplesheet, validate_samplesheet
    files = ["/data/sampleA_R1.fastq.gz", "/data/sampleA_R2.fastq.gz"]
    try:
        gen = await generate_samplesheet("rnaseq", files)
        if gen.get("error"):
            await fail("generate_samplesheet returned error", value=gen)
            return
        if "unstranded" in gen["samplesheet"]:
            await fail("strandedness was guessed — it must be left blank", value=gen["samplesheet"])
        else:
            await ok("strandedness left blank (not guessed)")
        val = await validate_samplesheet("rnaseq", gen["samplesheet"])
        await ok(f"validation ran: valid={val.get('valid')}, {len(val.get('errors', []))} error(s)")
    except Exception as e:
        await fail("samplesheet round-trip raised exception", exc=e)


async def test_generate_launch_command():
    await section("7. generate_launch_command — params passed through verbatim")
    from genomeops_mcp.tools.launch import generate_launch_command
    try:
        result = await generate_launch_command(
            "rnaseq", "3.14.0", "docker", "samplesheet.csv", "results",
            params={"--genome": "GRCh38", "--aligner": "star_salmon"},
        )
        if result.get("error"):
            await fail("generate_launch_command returned error", value=result)
            return
        cmd = result["command"]
        if "--genome GRCh38" in cmd and "--aligner star_salmon" in cmd:
            await ok("expert params present verbatim in command")
        else:
            await fail("expert params not passed through", value=cmd)
    except Exception as e:
        await fail("generate_launch_command raised exception", exc=e)


async def test_check_execution_environment():
    await section("8. check_execution_environment() — read-only probe")
    from genomeops_mcp.tools.execution import check_execution_environment
    try:
        env = await check_execution_environment("docker")
        await ok("probed environment", value={k: env.get(k) for k in ("ready", "recommended_profile", "missing")})
    except Exception as e:
        await fail("check_execution_environment raised exception", exc=e)


async def main():
    print("\nGenomeOps live smoke test (faithful-plumbing surface)\n")
    await test_list_pipelines()
    await test_list_pipelines_filtered()
    await test_get_pipeline_info()
    await test_get_latest_version()
    await test_get_samplesheet_schema()
    await test_generate_and_validate_samplesheet()
    await test_generate_launch_command()
    await test_check_execution_environment()
    print("\nDone.\n")


if __name__ == "__main__":
    asyncio.run(main())
