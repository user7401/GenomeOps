"""Tests for execution tools: preflight, setup, run, and status.

Subprocess interaction (_run_cmd, _launch_background, _pid_alive) is monkeypatched
so no real Nextflow or container engine is needed. The runs registry is redirected
to a tmp path.
"""

import json
from pathlib import Path

import pytest

from nfcore_mcp.tools import execution
from nfcore_mcp.tools.execution import (
    check_execution_environment,
    setup_environment,
    run_pipeline,
    get_run_status,
    _recommend_profile,
    _profile_available,
    _build_exec_args,
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


def test_build_exec_args_structure():
    args = _build_exec_args(
        "rnaseq", "3.14.0", "docker", "ss.csv", "out",
        {"--genome": "GRCh38", "--save_reference": True, "--skip_qc": False},
        "myrun", resume=True,
    )
    assert args[:3] == ["nextflow", "run", "nf-core/rnaseq"]
    assert "-r" in args and "3.14.0" in args
    assert "-name" in args and "myrun" in args
    assert "-resume" in args
    assert "--genome" in args and "GRCh38" in args
    assert "--save_reference" in args      # True bool → flag only
    assert "--skip_qc" not in args         # False bool → omitted


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
    assert "--genome" in captured["args"]

    # Registry file was written
    meta_file = runs_dir / "testrun.json"
    assert meta_file.exists()
    meta = json.loads(meta_file.read_text())
    assert meta["pid"] == 9999
    assert meta["status"] == "running"


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
