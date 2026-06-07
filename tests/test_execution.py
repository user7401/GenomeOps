"""Tests for execution tools: preflight, setup, run, and status.

Subprocess interaction (_run_cmd, _launch_background, _pid_alive) is monkeypatched
so no real Nextflow or container engine is needed. The runs registry is redirected
to a tmp path.
"""

import json
from pathlib import Path

import pytest

from genomeops_mcp.tools import execution
from genomeops_mcp.tools.execution import (
    check_execution_environment,
    setup_environment,
    run_pipeline,
    get_run_status,
    _recommend_profile,
    _profile_available,
    _build_exec_args,
    _build_params,
    _write_params_file,
    _scan_log_for_status,
)


@pytest.fixture(autouse=True)
def runs_dir(tmp_path, monkeypatch):
    """Redirect the runs registry to a temp directory for every test."""
    d = tmp_path / "runs"
    d.mkdir()
    monkeypatch.setattr(execution, "_RUNS_DIR", d)
    return d


def _fake_run_cmd(available: dict[str, int]):
    """Build a fake _run_cmd that reports tools in `available` as present.

    available maps the first command token to a returncode (0 = installed).
    Tokens not present raise the 'command not found' path.
    """
    version_text = {
        "nextflow": "nextflow version 23.10.1.5891",
        "java": 'openjdk version "17.0.9"',
        "docker": "Docker version 24.0.7",
        "singularity": "singularity version 3.11.4",
        "apptainer": "apptainer version 1.2.5",
        "podman": "podman version 4.6.2",
        "conda": "conda 23.7.4",
        "mamba": "mamba 1.5.1",
    }

    async def fake(args, timeout=30, cwd=None):
        tool = args[0]
        # `docker info` daemon probe
        if tool == "docker" and len(args) > 1 and args[1] == "info":
            rc = 0 if available.get("docker_daemon", 1) == 0 else 1
            return {"found": True, "returncode": rc, "stdout": "", "stderr": ""}
        if tool in ("nextflow",) and len(args) > 1 and args[1] in ("pull", "config"):
            rc = available.get(f"nextflow_{args[1]}", 0)
            return {"found": True, "returncode": rc, "stdout": f"{args[1]} ok", "stderr": ""}
        if tool in available:
            return {"found": True, "returncode": available[tool],
                    "stdout": version_text.get(tool, ""), "stderr": ""}
        return {"found": False, "returncode": 127, "stdout": "", "stderr": "command not found"}

    return fake


# Full, healthy environment: nextflow + java + docker (daemon up)
HEALTHY = {"nextflow": 0, "java": 0, "docker": 0, "docker_daemon": 0}


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def test_recommend_prefers_docker_with_daemon():
    engines = {
        "docker": {"installed": True, "daemon_running": True},
        "singularity": {"installed": True},
    }
    assert _recommend_profile(engines) == "docker"


def test_recommend_skips_docker_without_daemon():
    engines = {
        "docker": {"installed": True, "daemon_running": False},
        "singularity": {"installed": True},
        "apptainer": {"installed": False},
        "podman": {"installed": False},
        "conda": {"installed": False},
        "mamba": {"installed": False},
    }
    assert _recommend_profile(engines) == "singularity"


def test_recommend_falls_back_to_conda():
    engines = {
        "docker": {"installed": False},
        "singularity": {"installed": False},
        "apptainer": {"installed": False},
        "podman": {"installed": False},
        "conda": {"installed": True},
        "mamba": {"installed": False},
    }
    assert _recommend_profile(engines) == "conda"


def test_recommend_none_when_no_engine():
    engines = {k: {"installed": False} for k in
               ("docker", "singularity", "apptainer", "podman", "conda", "mamba")}
    assert _recommend_profile(engines) is None


def test_profile_available_docker_needs_daemon():
    engines = {"docker": {"installed": True, "daemon_running": False}}
    assert _profile_available("docker", engines) is False
    engines["docker"]["daemon_running"] = True
    assert _profile_available("docker", engines) is True


def test_profile_available_institutional_assumed_true():
    assert _profile_available("slurm", {}) is True


def test_build_exec_args_uses_params_file(tmp_path):
    pf = tmp_path / "myrun.params.json"
    args = _build_exec_args(
        "rnaseq", "3.14.0", "docker", "myrun", pf, resume=True,
    )
    assert args[:3] == ["nextflow", "run", "nf-core/rnaseq"]
    assert "-r" in args and "3.14.0" in args
    assert "-name" in args and "myrun" in args
    assert "-resume" in args
    assert "-params-file" in args and str(pf) in args
    # Parameters are no longer inline flags — they live in the params file.
    assert "--genome" not in args


def test_build_params_merges_input_outdir_and_strips_dashes():
    params = _build_params(
        "ss.csv", "out",
        {"--genome": "GRCh38", "--save_reference": True, "--skip_qc": False},
    )
    assert params["input"] == "ss.csv"
    assert params["outdir"] == "out"
    assert params["genome"] == "GRCh38"
    assert params["save_reference"] is True
    # Unlike the CLI, an explicit false IS kept in a params file (overrides a default).
    assert params["skip_qc"] is False


def test_write_params_file_round_trips(tmp_path):
    pf = tmp_path / "x" / "run.params.json"
    _write_params_file({"input": "ss.csv", "outdir": "out", "genome": "GRCh38"}, pf)
    assert pf.exists()
    loaded = json.loads(pf.read_text())
    assert loaded == {"input": "ss.csv", "outdir": "out", "genome": "GRCh38"}


@pytest.mark.parametrize("text,expected", [
    ("Pipeline completed successfully", "completed"),
    ("  -[nf-core/rnaseq] Pipeline completed successfully -", "completed"),
    ("ERROR ~ Something broke", "failed"),
    ("Execution aborted due to an unexpected error", "failed"),
    ("Goodbye", "completed"),
    ("still going", "unknown"),
    ("", "unknown"),
])
def test_scan_log_for_status(text, expected):
    assert _scan_log_for_status(text) == expected


# ---------------------------------------------------------------------------
# check_execution_environment
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_check_env_ready(monkeypatch):
    monkeypatch.setattr(execution, "_run_cmd", _fake_run_cmd(HEALTHY))
    result = await check_execution_environment()
    assert result["ready"] is True
    assert result["nextflow"]["installed"] is True
    assert result["nextflow"]["version"] == "23.10.1"
    assert result["recommended_profile"] == "docker"
    assert result["missing"] == []


@pytest.mark.asyncio
async def test_check_env_missing_nextflow(monkeypatch):
    monkeypatch.setattr(execution, "_run_cmd",
                        _fake_run_cmd({"java": 0, "docker": 0, "docker_daemon": 0}))
    result = await check_execution_environment()
    assert result["ready"] is False
    assert "nextflow" in result["missing"]
    assert "nextflow" in result["install_hints"]


@pytest.mark.asyncio
async def test_check_env_docker_daemon_down(monkeypatch):
    # docker installed but daemon not running, nothing else available
    monkeypatch.setattr(execution, "_run_cmd",
                        _fake_run_cmd({"nextflow": 0, "java": 0, "docker": 0, "docker_daemon": 1}))
    result = await check_execution_environment()
    assert result["engines"]["docker"]["installed"] is True
    assert result["engines"]["docker"]["daemon_running"] is False
    assert result["recommended_profile"] is None
    assert result["ready"] is False


@pytest.mark.asyncio
async def test_check_env_requested_profile_unavailable(monkeypatch):
    monkeypatch.setattr(execution, "_run_cmd", _fake_run_cmd(HEALTHY))
    result = await check_execution_environment(profile="singularity")
    assert result["requested_profile_available"] is False


# ---------------------------------------------------------------------------
# setup_environment
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_setup_environment_success(monkeypatch):
    monkeypatch.setattr(execution, "_run_cmd", _fake_run_cmd(HEALTHY))
    result = await setup_environment("rnaseq", "3.14.0", "docker")
    assert result["ready"] is True
    step_names = [s["step"] for s in result["steps"]]
    assert step_names == ["pull", "config"]
    assert all(s["success"] for s in result["steps"])


@pytest.mark.asyncio
async def test_setup_environment_pull_failure_skips_config(monkeypatch):
    avail = dict(HEALTHY, nextflow_pull=1)
    monkeypatch.setattr(execution, "_run_cmd", _fake_run_cmd(avail))
    result = await setup_environment("rnaseq", "3.14.0", "docker")
    assert result["ready"] is False
    assert [s["step"] for s in result["steps"]] == ["pull"]  # config skipped


@pytest.mark.asyncio
async def test_setup_environment_not_ready(monkeypatch):
    monkeypatch.setattr(execution, "_run_cmd", _fake_run_cmd({"java": 0}))  # no nextflow
    result = await setup_environment("rnaseq", "3.14.0", "docker")
    assert result["ready"] is False
    assert result["steps"] == []


# ---------------------------------------------------------------------------
# run_pipeline — gating
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_without_confirm_previews_only(monkeypatch):
    monkeypatch.setattr(execution, "_run_cmd", _fake_run_cmd(HEALTHY))
    launched = {"called": False}

    def fake_launch(*a, **k):
        launched["called"] = True
        return 4242

    monkeypatch.setattr(execution, "_launch_background", fake_launch)

    result = await run_pipeline("rnaseq", "3.14.0", "docker", "ss.csv", "out")

    assert result["requires_confirmation"] is True
    assert "nextflow run" in result["command"]
    assert result["review_required"] is True
    assert launched["called"] is False  # nothing actually ran


@pytest.mark.asyncio
async def test_run_rejects_master_version(monkeypatch):
    monkeypatch.setattr(execution, "_run_cmd", _fake_run_cmd(HEALTHY))
    result = await run_pipeline("rnaseq", "main", "docker", "ss.csv", "out", confirm=True)
    assert result.get("error") is True
    assert result["code"] == "VERSION_REQUIRED"


@pytest.mark.asyncio
async def test_run_blocked_when_env_not_ready(monkeypatch):
    monkeypatch.setattr(execution, "_run_cmd", _fake_run_cmd({"java": 0}))  # no nextflow/engine
    result = await run_pipeline("rnaseq", "3.14.0", "docker", "ss.csv", "out", confirm=True)
    assert result.get("error") is True
    assert result["code"] == "ENV_NOT_READY"


@pytest.mark.asyncio
async def test_run_blocked_when_profile_unavailable(monkeypatch):
    monkeypatch.setattr(execution, "_run_cmd", _fake_run_cmd(HEALTHY))  # only docker
    result = await run_pipeline("rnaseq", "3.14.0", "singularity", "ss.csv", "out", confirm=True)
    assert result.get("error") is True
    assert result["code"] == "PROFILE_UNAVAILABLE"


@pytest.mark.asyncio
async def test_run_with_confirm_launches(monkeypatch, runs_dir):
    monkeypatch.setattr(execution, "_run_cmd", _fake_run_cmd(HEALTHY))
    captured = {}

    def fake_launch(args, log_path, work_dir):
        captured["args"] = args
        captured["log_path"] = log_path
        return 9999

    monkeypatch.setattr(execution, "_launch_background", fake_launch)

    result = await run_pipeline(
        "rnaseq", "3.14.0", "docker", "ss.csv", "out",
        params={"--genome": "GRCh38"}, confirm=True, run_name="testrun",
    )

    assert result["status"] == "launched"
    assert result["pid"] == 9999
    assert result["run_name"] == "testrun"
    assert captured["args"][:3] == ["nextflow", "run", "nf-core/rnaseq"]
    # Params are passed via -params-file, not inline flags
    assert "-params-file" in captured["args"]
    assert "--genome" not in captured["args"]

    # Params file was written with the merged parameters
    params_file = runs_dir / "testrun.params.json"
    assert params_file.exists()
    params = json.loads(params_file.read_text())
    assert params["input"] == "ss.csv"
    assert params["outdir"] == "out"
    assert params["genome"] == "GRCh38"

    # Registry file was written with provenance fields
    meta_file = runs_dir / "testrun.json"
    assert meta_file.exists()
    meta = json.loads(meta_file.read_text())
    assert meta["pid"] == 9999
    assert meta["status"] == "running"
    assert meta["params"]["genome"] == "GRCh38"
    assert meta["params_file"].endswith("testrun.params.json")
    assert meta["launched_with"] == "genomeops-mcp"
    assert meta["confirmed"] is True


# ---------------------------------------------------------------------------
# get_run_status
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_status_unknown_run():
    result = await get_run_status("does-not-exist")
    assert result.get("error") is True
    assert result["code"] == "RUN_NOT_FOUND"


@pytest.mark.asyncio
async def test_status_running(monkeypatch, runs_dir):
    log = runs_dir / "r1.log"
    log.write_text("executor > local\n[aa/bbccdd] process > FASTQC\n")
    execution._save_run("r1", {
        "run_name": "r1", "pid": 1234, "log_file": str(log),
        "outdir": "out", "status": "running", "started_at": 1000.0,
    })
    monkeypatch.setattr(execution, "_pid_alive", lambda pid: True)

    result = await get_run_status("r1")
    assert result["status"] == "running"
    assert result["alive"] is True
    assert len(result["log_tail"]) >= 1


@pytest.mark.asyncio
async def test_status_completed(monkeypatch, runs_dir):
    log = runs_dir / "r2.log"
    log.write_text("...\n-[nf-core/rnaseq] Pipeline completed successfully-\n")
    execution._save_run("r2", {
        "run_name": "r2", "pid": 1234, "log_file": str(log),
        "outdir": "out", "status": "running", "started_at": 1000.0,
    })
    monkeypatch.setattr(execution, "_pid_alive", lambda pid: False)

    result = await get_run_status("r2")
    assert result["status"] == "completed"
    assert result["alive"] is False
    assert any("parse_run_summary" in s for s in result["next_steps"])

    # Terminal status is persisted
    meta = json.loads((runs_dir / "r2.json").read_text())
    assert meta["status"] == "completed"


@pytest.mark.asyncio
async def test_status_failed(monkeypatch, runs_dir):
    log = runs_dir / "r3.log"
    log.write_text("...\nERROR ~ Error executing process > FASTQC\nExecution aborted\n")
    execution._save_run("r3", {
        "run_name": "r3", "pid": 1234, "log_file": str(log),
        "outdir": "out", "status": "running", "started_at": 1000.0,
    })
    monkeypatch.setattr(execution, "_pid_alive", lambda pid: False)

    result = await get_run_status("r3")
    assert result["status"] == "failed"
    assert any("resume" in s.lower() for s in result["next_steps"])


# ---------------------------------------------------------------------------
# generate_methods_note
# ---------------------------------------------------------------------------

from genomeops_mcp.tools.execution import generate_methods_note


@pytest.mark.asyncio
async def test_methods_note_unknown_run():
    result = await generate_methods_note("nope")
    assert result.get("error") is True
    assert result["code"] == "RUN_NOT_FOUND"


@pytest.mark.asyncio
async def test_methods_note_includes_facts_and_citations(runs_dir):
    execution._save_run("m1", {
        "run_name": "m1", "pipeline": "rnaseq", "version": "3.14.0",
        "profile": "docker", "outdir": "/no/such/dir",
        "params": {"input": "ss.csv", "outdir": "/no/such/dir", "genome": "GRCh38"},
        "params_file": str(runs_dir / "m1.params.json"),
        "engine_versions": {"nextflow": "23.10.1", "docker": "24.0.7"},
        "status": "completed",
    })

    result = await generate_methods_note("m1")
    assert "nf-core/rnaseq" in result["methods_text"]
    assert "3.14.0" in result["methods_text"]
    assert "docker" in result["methods_text"]
    assert "Nextflow v23.10.1" in result["methods_text"]
    # Non-default param surfaces; I/O paths do not
    assert "--genome GRCh38" in result["methods_text"]
    assert result["parameters_used"] == {"genome": "GRCh38"}
    # Canonical citations present
    dois = [c["doi"] for c in result["citations"]]
    assert "10.1038/s41587-020-0439-x" in dois  # nf-core
    assert "10.1038/nbt.3820" in dois           # Nextflow
    assert result["warnings"] == []             # pinned stable version → no warning
    assert result["review_required"] is True


@pytest.mark.asyncio
async def test_methods_note_warns_on_unpinned_version(runs_dir):
    execution._save_run("m2", {
        "run_name": "m2", "pipeline": "rnaseq", "version": "master",
        "profile": "docker", "outdir": "out",
        "params": {"input": "ss.csv", "outdir": "out"},
        "engine_versions": {},
        "status": "completed",
    })
    result = await generate_methods_note("m2")
    assert any("reproducib" in w.lower() for w in result["warnings"])
    assert "All parameters were left at the pipeline defaults." in result["methods_text"]


# ---------------------------------------------------------------------------
# list_runs
# ---------------------------------------------------------------------------

from genomeops_mcp.tools.execution import list_runs, stop_pipeline


@pytest.mark.asyncio
async def test_list_runs_empty():
    result = await list_runs()
    assert result == {"runs": [], "total": 0}


@pytest.mark.asyncio
async def test_list_runs_sorted_and_excludes_params_files(monkeypatch, runs_dir):
    execution._save_run("old", {
        "run_name": "old", "pid": 1, "pipeline": "rnaseq", "version": "3.14.0",
        "profile": "docker", "outdir": "o1", "status": "completed", "started_at": 100.0,
    })
    execution._save_run("new", {
        "run_name": "new", "pid": 2, "pipeline": "sarek", "version": "3.4.4",
        "profile": "docker", "outdir": "o2", "status": "completed", "started_at": 200.0,
    })
    # A stray params file must not be listed as a run
    (runs_dir / "new.params.json").write_text('{"input": "ss.csv"}')

    result = await list_runs()
    assert result["total"] == 2
    assert [r["run_name"] for r in result["runs"]] == ["new", "old"]  # newest first


@pytest.mark.asyncio
async def test_list_runs_refreshes_dead_running_status(monkeypatch, runs_dir):
    log = runs_dir / "z.log"
    log.write_text("...\n-[nf-core/rnaseq] Pipeline completed successfully-\n")
    execution._save_run("z", {
        "run_name": "z", "pid": 555, "pipeline": "rnaseq", "version": "3.14.0",
        "profile": "docker", "outdir": "o", "log_file": str(log),
        "status": "running", "started_at": 100.0,
    })
    monkeypatch.setattr(execution, "_pid_alive", lambda pid: False)

    result = await list_runs()
    assert result["runs"][0]["status"] == "completed"
    # Terminal status persisted back to the registry
    meta = json.loads((runs_dir / "z.json").read_text())
    assert meta["status"] == "completed"


# ---------------------------------------------------------------------------
# stop_pipeline — gating
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_stop_unknown_run():
    result = await stop_pipeline("ghost")
    assert result.get("error") is True
    assert result["code"] == "RUN_NOT_FOUND"


@pytest.mark.asyncio
async def test_stop_not_running(monkeypatch, runs_dir):
    execution._save_run("done", {
        "run_name": "done", "pid": 9, "status": "completed", "outdir": "o",
    })
    monkeypatch.setattr(execution, "_pid_alive", lambda pid: False)
    result = await stop_pipeline("done", confirm=True)
    assert result["already_stopped"] is True


@pytest.mark.asyncio
async def test_stop_without_confirm_previews_only(monkeypatch, runs_dir):
    execution._save_run("live", {
        "run_name": "live", "pid": 777, "status": "running", "outdir": "o",
    })
    monkeypatch.setattr(execution, "_pid_alive", lambda pid: True)
    killed = {"called": False}
    monkeypatch.setattr(execution, "_terminate_pid",
                        lambda pid: killed.__setitem__("called", True) or True)

    result = await stop_pipeline("live")  # confirm defaults to False
    assert result["requires_confirmation"] is True
    assert killed["called"] is False  # nothing was signalled


@pytest.mark.asyncio
async def test_stop_with_confirm_terminates(monkeypatch, runs_dir):
    execution._save_run("live2", {
        "run_name": "live2", "pid": 888, "status": "running", "outdir": "o",
    })
    monkeypatch.setattr(execution, "_pid_alive", lambda pid: True)
    captured = {}
    monkeypatch.setattr(execution, "_terminate_pid",
                        lambda pid: captured.__setitem__("pid", pid) or True)

    result = await stop_pipeline("live2", confirm=True)
    assert result["status"] == "stopped"
    assert result["signal_sent"] is True
    assert captured["pid"] == 888
    meta = json.loads((runs_dir / "live2.json").read_text())
    assert meta["status"] == "stopped"


# ---------------------------------------------------------------------------
# diagnose_run_failure
# ---------------------------------------------------------------------------

from genomeops_mcp.tools.execution import (
    diagnose_run_failure,
    _parse_failure,
    _classify_failure,
    _read_work_dir,
)

_SAMPLE_FAILURE_LOG = """\
executor >  local
[ab/cdef01] process > NFCORE_RNASEQ:RNASEQ:FASTQC (S1) [100%] 1 of 1, failed: 1
ERROR ~ Error executing process > 'NFCORE_RNASEQ:RNASEQ:FASTQC (S1)'

Caused by:
  Process `FASTQC (S1)` terminated with an error exit status (137)

Command executed:
  fastqc --threads 2 S1.fastq.gz

Command exit status:
  137

Command error:
  .command.sh: line 5:   123 Killed   fastqc --threads 2 S1.fastq.gz
  slurmstepd: error: Detected 1 oom-kill event

Work dir:
  /work/ab/cdef0123456789

Tip: you can replicate the issue by changing to the process work dir
"""


def test_parse_failure_extracts_fields():
    parsed = _parse_failure(_SAMPLE_FAILURE_LOG)
    assert "FASTQC" in parsed["failed_process"]
    assert parsed["exit_code"] == 137
    assert parsed["work_dir"] == "/work/ab/cdef0123456789"
    assert any("killed" in ln.lower() for ln in parsed["error_excerpt"])


@pytest.mark.parametrize("exit_code,text,expect", [
    (137, "oom-kill event", "memory"),
    (143, "time limit reached", "time"),
    (1, "No such file or directory", "missing"),
    (1, "Unable to find image 'x' locally", "container"),
    (None, "", "could not be determined"),
])
def test_classify_failure(exit_code, text, expect):
    cause, fixes = _classify_failure(exit_code, text)
    assert expect.lower() in cause.lower()
    assert isinstance(fixes, list) and fixes


def test_read_work_dir_reads_task_files(tmp_path):
    wd = tmp_path / "ab" / "cdef"
    wd.mkdir(parents=True)
    (wd / ".command.err").write_text("boom\nKilled\n")
    (wd / ".command.sh").write_text("fastqc S1.fastq.gz\n")
    (wd / ".exitcode").write_text("137\n")
    out = _read_work_dir(str(wd))
    assert "Killed" in out["command_err"]
    assert "fastqc" in out["command_sh"]
    assert out["exit_code"] == 137


def test_read_work_dir_missing_is_safe():
    assert _read_work_dir(None) == {"command_err": None, "command_sh": None, "exit_code": None}
    assert _read_work_dir("/no/such/path")["command_err"] is None


@pytest.mark.asyncio
async def test_diagnose_unknown_run():
    result = await diagnose_run_failure("ghost")
    assert result.get("error") is True
    assert result["code"] == "RUN_NOT_FOUND"


@pytest.mark.asyncio
async def test_diagnose_oom_from_log_heuristic(monkeypatch, runs_dir):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    log = runs_dir / "f1.log"
    log.write_text(_SAMPLE_FAILURE_LOG)
    execution._save_run("f1", {
        "run_name": "f1", "pid": 1, "pipeline": "rnaseq", "version": "3.14.0",
        "profile": "docker", "outdir": "o", "log_file": str(log), "status": "failed",
    })
    monkeypatch.setattr(execution, "_pid_alive", lambda pid: False)

    result = await diagnose_run_failure("f1")
    assert result["analysis_method"] == "heuristic"
    assert result["exit_code"] == 137
    assert "FASTQC" in result["failed_process"]
    assert "memory" in result["likely_cause"].lower()
    assert any("resume" in f.lower() for f in result["suggested_fixes"])
    assert result["review_required"] is True


@pytest.mark.asyncio
async def test_diagnose_prefers_work_dir_files(monkeypatch, runs_dir, tmp_path):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    wd = tmp_path / "ab" / "cdef0123456789"
    wd.mkdir(parents=True)
    (wd / ".command.err").write_text("No such file or directory: genome.fa\n")
    (wd / ".exitcode").write_text("1\n")
    (wd / ".command.sh").write_text("bwa index genome.fa\n")

    log = runs_dir / "f2.log"
    log.write_text(
        "ERROR ~ Error executing process > 'BWA_INDEX'\n"
        "Command exit status:\n  1\n"
        f"Work dir:\n  {wd}\n"
    )
    execution._save_run("f2", {
        "run_name": "f2", "pid": 1, "pipeline": "sarek", "version": "3.4.4",
        "profile": "docker", "outdir": "o", "log_file": str(log), "status": "failed",
    })
    monkeypatch.setattr(execution, "_pid_alive", lambda pid: False)

    result = await diagnose_run_failure("f2")
    assert result["exit_code"] == 1
    assert "missing" in result["likely_cause"].lower()
    assert "bwa index" in result["command_preview"]


# ---------------------------------------------------------------------------
# diagnose_resume
# ---------------------------------------------------------------------------

from genomeops_mcp.tools.execution import (
    diagnose_resume,
    _fingerprint_inputs,
    _compare_fingerprints,
    _fingerprint_file,
)


def test_fingerprint_inputs_includes_samplesheet_and_referenced_files(tmp_path):
    r1 = tmp_path / "S1_R1.fastq.gz"
    r1.write_text("@read\nACGT\n+\nFFFF\n")
    ss = tmp_path / "samplesheet.csv"
    ss.write_text(f"sample,fastq_1\nS1,{r1}\n")

    prints = _fingerprint_inputs(str(ss))
    assert str(ss) in prints
    assert str(r1) in prints
    assert prints[str(r1)]["size"] > 0


def test_compare_fingerprints_detects_changes(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("one")
    recorded = {str(f): _fingerprint_file(str(f))}

    assert _compare_fingerprints(recorded) == []  # unchanged

    f.write_text("a much longer content")  # size changes
    changed = _compare_fingerprints(recorded)
    assert changed and changed[0]["path"] == str(f)

    # Missing file is reported
    f.unlink()
    missing = _compare_fingerprints(recorded)
    assert "no longer exists" in missing[0]["reason"]


@pytest.mark.asyncio
async def test_diagnose_resume_unknown_run():
    result = await diagnose_resume("ghost")
    assert result.get("error") is True
    assert result["code"] == "RUN_NOT_FOUND"


@pytest.mark.asyncio
async def test_diagnose_resume_all_intact(tmp_path, runs_dir):
    launch = tmp_path / "launch"
    (launch / ".nextflow").mkdir(parents=True)
    (launch / "work").mkdir()
    ss = launch / "ss.csv"
    ss.write_text("sample,fastq_1\nS1,reads.fastq.gz\n")

    execution._save_run("res1", {
        "run_name": "res1", "pid": 1, "outdir": "o",
        "launch_dir": str(launch),
        "input_fingerprints": _fingerprint_inputs(str(ss)),
    })

    result = await diagnose_resume("res1")
    assert result["cache_present"] is True
    assert result["work_dir_present"] is True
    assert result["changed_inputs"] == []
    assert result["resume_will_reuse_cache"] is True


@pytest.mark.asyncio
async def test_diagnose_resume_flags_missing_cache_and_changed_input(tmp_path, runs_dir):
    launch = tmp_path / "launch"
    launch.mkdir()
    # no .nextflow, no work dir
    ss = launch / "ss.csv"
    ss.write_text("sample,fastq_1\nS1,reads.fastq.gz\n")
    fingerprints = _fingerprint_inputs(str(ss))

    # mutate the samplesheet after recording the fingerprint
    ss.write_text("sample,fastq_1\nS1,reads.fastq.gz\nS2,other.fastq.gz\n")

    execution._save_run("res2", {
        "run_name": "res2", "pid": 1, "outdir": "o",
        "launch_dir": str(launch),
        "input_fingerprints": fingerprints,
    })

    result = await diagnose_resume("res2")
    assert result["cache_present"] is False
    assert result["work_dir_present"] is False
    assert any(c["path"] == str(ss) for c in result["changed_inputs"])
    assert result["resume_will_reuse_cache"] is False
    assert result["issues"]


# ---------------------------------------------------------------------------
# estimate_resources
# ---------------------------------------------------------------------------

from genomeops_mcp.tools.execution import (
    estimate_resources,
    _gather_input_signals,
    _scan_samplesheet_inputs,
    _heuristic_estimate,
    _machine_verdict,
    _suggest_overrides,
    _probe_machine,
)


def test_probe_machine_reports_cpu_and_never_raises():
    m = _probe_machine()
    assert m["cpu_count"] >= 1
    # memory/disk may be None on exotic platforms, but the keys are always present
    assert "total_memory_gb" in m
    assert "available_memory_gb" in m
    assert "disk_free_gb" in m


def test_scan_samplesheet_counts_rows_and_sizes_files(tmp_path):
    r1 = tmp_path / "S1_R1.fastq.gz"
    r1.write_bytes(b"x" * 1000)
    r2 = tmp_path / "S2_R1.fastq.gz"
    r2.write_bytes(b"y" * 2000)
    ss = tmp_path / "samplesheet.csv"
    ss.write_text(f"sample,fastq_1\nS1,{r1}\nS2,{r2}\n")

    rows, total_bytes = _scan_samplesheet_inputs(str(ss))
    assert rows == 2
    assert total_bytes == 3000


def test_gather_input_signals_prefers_data_summary(tmp_path):
    signals = _gather_input_signals(
        samplesheet_path=None,
        file_paths=None,
        data_summary={"sample_count": 8, "paired_end": True, "total_input_gb": 12.5},
    )
    assert signals["sample_count"] == 8
    assert signals["paired_end"] is True
    assert signals["total_input_gb"] == 12.5
    assert signals["source"] == "data_summary"


def test_gather_input_signals_from_samplesheet(tmp_path):
    r1 = tmp_path / "a.fastq.gz"
    r1.write_bytes(b"z" * 5_000_000)
    ss = tmp_path / "ss.csv"
    ss.write_text(f"sample,fastq_1\nS1,{r1}\n")

    signals = _gather_input_signals(str(ss), None, None)
    assert signals["sample_count"] == 1
    assert signals["total_input_gb"] == round(5_000_000 / 1e9, 2)
    assert signals["source"] == "samplesheet"


def test_heuristic_estimate_uses_pipeline_profile():
    machine = {"cpu_count": 8, "total_memory_gb": 64, "disk_free_gb": 500}
    signals = {"sample_count": 4, "total_input_gb": 10, "paired_end": True}
    est = _heuristic_estimate("rnaseq", signals, machine)
    assert est["recommended_memory_gb"] == 40  # rnaseq profile peak
    assert est["recommended_cpus"] == 12
    assert est["estimated_disk_gb"] == 80.0  # 10 GB * disk_mult 8
    assert est["estimated_walltime_hours"] > 0


def test_heuristic_estimate_unknown_pipeline_uses_default():
    machine = {"cpu_count": 4, "total_memory_gb": 16, "disk_free_gb": 100}
    est = _heuristic_estimate("totallymadeup", {"sample_count": 1}, machine)
    assert est["recommended_memory_gb"] == 32  # default profile
    assert est["estimated_disk_gb"] is None  # no input size known


def test_machine_verdict_insufficient_memory():
    estimate = {"recommended_cpus": 12, "recommended_memory_gb": 40, "estimated_disk_gb": 50}
    machine = {"cpu_count": 8, "total_memory_gb": 16, "disk_free_gb": 500}
    verdict, reasons = _machine_verdict(estimate, machine)
    assert verdict == "insufficient"
    assert any("RAM" in r for r in reasons)


def test_machine_verdict_sufficient():
    estimate = {"recommended_cpus": 8, "recommended_memory_gb": 16, "estimated_disk_gb": 50}
    machine = {"cpu_count": 16, "total_memory_gb": 64, "disk_free_gb": 500}
    verdict, reasons = _machine_verdict(estimate, machine)
    assert verdict == "sufficient"
    assert reasons


def test_machine_verdict_insufficient_disk_overrides_marginal():
    estimate = {"recommended_cpus": 8, "recommended_memory_gb": 16, "estimated_disk_gb": 1000}
    machine = {"cpu_count": 3, "total_memory_gb": 18, "disk_free_gb": 50}
    verdict, reasons = _machine_verdict(estimate, machine)
    assert verdict == "insufficient"  # disk shortfall wins over a marginal cpu/mem


def test_suggest_overrides_caps_to_machine():
    estimate = {"estimated_walltime_hours": 5}
    machine = {"cpu_count": 8, "total_memory_gb": 32}
    ov = _suggest_overrides(estimate, machine)
    assert ov["--max_cpus"] == 8
    assert ov["--max_memory"] == "27.GB"  # int(32 * 0.85)
    assert ov["--max_time"] == "10.h"  # 5h * 2


@pytest.mark.asyncio
async def test_estimate_resources_heuristic_end_to_end(monkeypatch, tmp_path):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(execution, "_probe_machine", lambda: {
        "cpu_count": 4, "total_memory_gb": 8, "available_memory_gb": 6, "disk_free_gb": 40,
    })
    result = await estimate_resources(
        "rnaseq", "3.14.0",
        data_summary={"sample_count": 6, "total_input_gb": 20, "paired_end": True},
    )
    assert result["analysis_method"] == "heuristic"
    assert result["machine_verdict"] == "insufficient"  # 8 GB << 40 GB rnaseq peak
    assert result["estimated_requirements"]["recommended_memory_gb"] == 40
    assert "--max_cpus" in result["suggested_overrides"]
    assert result["review_required"] is True
    assert result["confidence"] == "low"
    # Evidence block is present for host-agent reasoning even without a key.
    assert result["reasoning_inputs"]["detected_machine"]["cpu_count"] == 4
    assert result["reasoning_inputs"]["pipeline_profile"]["peak_mem_gb"] == 40


@pytest.mark.asyncio
async def test_estimate_resources_caveats_when_no_size(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    result = await estimate_resources("rnaseq", "3.14.0")
    # No samplesheet, files, or data_summary → size and sample count unknown
    assert any("input size" in c.lower() for c in result["caveats"])
    assert any("sample count" in c.lower() for c in result["caveats"])


@pytest.mark.asyncio
async def test_estimate_resources_llm_refines(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(execution, "_probe_machine", lambda: {
        "cpu_count": 32, "total_memory_gb": 128, "available_memory_gb": 120, "disk_free_gb": 2000,
    })

    payload = json.dumps({
        "estimated_requirements": {
            "recommended_cpus": 24, "recommended_memory_gb": 64,
            "estimated_peak_memory_gb": 64, "estimated_disk_gb": 300,
            "estimated_walltime_hours": 8, "basis": "LLM reasoning",
        },
        "machine_verdict": "sufficient",
        "verdict_reasons": ["plenty of headroom"],
        "suggested_overrides": {"--max_cpus": 32, "--max_memory": "110.GB", "--max_time": "16.h"},
        "confidence": "high",
        "caveats": ["depth assumed ~30x"],
    })

    class _Msg:
        def __init__(self, text):
            self.content = [type("C", (), {"text": text})()]

    class _Client:
        def __init__(self, *a, **k):
            self.messages = self

        def create(self, *a, **k):
            return _Msg(payload)

    import anthropic
    monkeypatch.setattr(anthropic, "Anthropic", _Client)

    result = await estimate_resources(
        "sarek", "3.4.4", data_summary={"sample_count": 4, "total_input_gb": 30}
    )
    assert result["analysis_method"] == "llm"
    assert result["machine_verdict"] == "sufficient"
    assert result["estimated_requirements"]["basis"] == "LLM reasoning"
    assert result["confidence"] == "high"
    assert "depth assumed ~30x" in result["caveats"]


@pytest.mark.asyncio
async def test_estimate_resources_llm_invalid_falls_back(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    class _Msg:
        def __init__(self, text):
            self.content = [type("C", (), {"text": text})()]

    class _Client:
        def __init__(self, *a, **k):
            self.messages = self

        def create(self, *a, **k):
            return _Msg("not json at all")

    import anthropic
    monkeypatch.setattr(anthropic, "Anthropic", _Client)

    result = await estimate_resources("rnaseq", "3.14.0", data_summary={"sample_count": 1})
    assert result["analysis_method"] == "heuristic"  # degraded gracefully
