"""Execution tools: preflight environment checks, setup, and actually running pipelines.

This is the bridge from "configured command" to "running analysis". Four tools,
escalating in side effects:

  check_execution_environment — read-only. Detects Nextflow/Java/container engines
                                and recommends a viable -profile.
  setup_environment           — caches the pipeline (nextflow pull) and resolves
                                its config as a dry run. No analysis is executed.
  run_pipeline                — launches the real `nextflow run` in the background.
                                Gated: requires the environment to be ready AND an
                                explicit confirm=True, after a preflight check.
  get_run_status              — polls a launched run (alive?/completed?/failed?)
                                and tails its log.

nf-core does NOT require the user to build a conda env per tool: `nextflow run
nf-core/<name>` fetches the pipeline, and -profile docker|singularity|conda makes
Nextflow create each process's environment automatically. The user only needs
Nextflow + Java + one engine installed once — which check_execution_environment
verifies.
"""

import asyncio
import json
import logging
import os
import re
import shutil
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

from nfcore_mcp.tools.results import generate_launch_command

logger = logging.getLogger(__name__)

_RUNS_DIR = Path.home() / ".cache" / "nfcore-mcp" / "runs"
_LOG_TAIL_LINES = 40

# Container/conda engines, in descending order of nf-core preference.
_ENGINE_PROBES: list[tuple[str, list[str]]] = [
    ("docker", ["docker", "--version"]),
    ("singularity", ["singularity", "--version"]),
    ("apptainer", ["apptainer", "--version"]),
    ("podman", ["podman", "--version"]),
    ("conda", ["conda", "--version"]),
    ("mamba", ["mamba", "--version"]),
]

_INSTALL_HINTS = {
    "nextflow": "Install Nextflow: `curl -s https://get.nextflow.io | bash` "
                "(requires Java 17+). See https://nf-co.re/docs/usage/installation",
    "java": "Install Java 17+ (Temurin/OpenJDK). Nextflow requires it. "
            "`conda install -c conda-forge openjdk=17` or your OS package manager.",
    "engine": "Install one container engine — Docker (recommended), "
              "Singularity/Apptainer (HPC), Podman, or Conda. "
              "Docker: https://docs.docker.com/get-docker/",
}


# ===========================================================================
# Tool 1 — check_execution_environment
# ===========================================================================

async def check_execution_environment(profile: str | None = None) -> dict[str, Any]:
    """Detect whether this machine can run nf-core pipelines, and how.

    Read-only preflight. Probes for Nextflow, Java, and every supported container
    engine (Docker, Singularity, Apptainer, Podman, Conda/Mamba), reports their
    versions, recommends a usable -profile, and lists exactly what to install if
    something is missing. Call this before setup_environment or run_pipeline.

    Args:
        profile: Optional profile the user wants to use. If given, the result
            reports whether it is actually available on this machine.

    Returns:
        {
          "ready": bool,                 # True if a pipeline can be launched now
          "nextflow": {"installed","version"},
          "java": {"installed","version"},
          "engines": {"<name>": {"installed","version",["daemon_running"]}},
          "recommended_profile": str | None,
          "requested_profile_available": bool | None,
          "missing": [str],
          "install_hints": {str: str},
          "disk_free_gb": float | None,
          "notes": [str]
        }
    """
    logger.info("Tool call: check_execution_environment(profile=%r)", profile)

    nextflow = await _probe_tool(["nextflow", "-version"])
    java = await _probe_tool(["java", "-version"])

    engines: dict[str, Any] = {}
    for name, args in _ENGINE_PROBES:
        info = await _probe_tool(args)
        if name == "docker" and info["installed"]:
            daemon = await _run_cmd(["docker", "info"], timeout=10)
            info["daemon_running"] = daemon["returncode"] == 0
        engines[name] = info

    recommended = _recommend_profile(engines)

    missing: list[str] = []
    hints: dict[str, str] = {}
    if not nextflow["installed"]:
        missing.append("nextflow")
        hints["nextflow"] = _INSTALL_HINTS["nextflow"]
    if not java["installed"]:
        missing.append("java")
        hints["java"] = _INSTALL_HINTS["java"]
    if recommended is None:
        missing.append("container engine")
        hints["engine"] = _INSTALL_HINTS["engine"]

    ready = nextflow["installed"] and java["installed"] and recommended is not None

    requested_available: bool | None = None
    notes: list[str] = []
    if profile:
        requested_available = _profile_available(profile, engines)
        if not requested_available:
            notes.append(
                f"Requested profile '{profile}' is not available here. "
                f"Recommended instead: {recommended or 'none available'}."
            )

    disk_free = None
    try:
        disk_free = round(shutil.disk_usage(Path.cwd()).free / 1e9, 1)
        if disk_free < 10:
            notes.append(f"Low disk space: {disk_free} GB free. Pipelines can need tens of GB.")
    except Exception:
        pass

    if ready:
        notes.append(
            f"Environment is ready. Recommended profile: '{recommended}'. "
            f"nf-core fetches the pipeline and builds per-process environments "
            f"automatically — no manual conda setup needed."
        )

    return {
        "ready": ready,
        "nextflow": nextflow,
        "java": java,
        "engines": engines,
        "recommended_profile": recommended,
        "requested_profile_available": requested_available,
        "missing": missing,
        "install_hints": hints,
        "disk_free_gb": disk_free,
        "notes": notes,
    }


# ===========================================================================
# Tool 2 — setup_environment
# ===========================================================================

async def setup_environment(
    pipeline_name: str,
    version: str,
    profile: str,
) -> dict[str, Any]:
    """Prepare to run a pipeline without executing the analysis (a dry run).

    Caches the pipeline code (`nextflow pull`) and resolves its configuration for
    the chosen profile (`nextflow config`), which surfaces profile/config errors
    before any compute is spent. No samples are processed. Safe to run repeatedly.

    Run this after check_execution_environment reports ready=true and before
    run_pipeline.

    Args:
        pipeline_name: nf-core pipeline name (e.g. "rnaseq").
        version: Pinned release tag (e.g. "3.14.0").
        profile: Execution profile (e.g. "docker").

    Returns:
        {
          "pipeline","version","profile",
          "ready": bool,                 # True if every step succeeded and env ready
          "environment": {...},          # check_execution_environment result
          "steps": [{"step","success","output"}],
          "notes": [str]
        }
    """
    logger.info("Tool call: setup_environment(%r, %r, %r)", pipeline_name, version, profile)

    env = await check_execution_environment(profile)
    steps: list[dict[str, Any]] = []
    notes: list[str] = []

    if not env["ready"]:
        return {
            "pipeline": pipeline_name,
            "version": version,
            "profile": profile,
            "ready": False,
            "environment": env,
            "steps": steps,
            "notes": ["Environment is not ready — resolve 'missing' items first."],
        }

    repo = f"nf-core/{pipeline_name}"

    pull = await _run_cmd(["nextflow", "pull", repo, "-r", version], timeout=900)
    steps.append({
        "step": "pull",
        "success": pull["returncode"] == 0,
        "output": _tail_text(pull["stdout"] + pull["stderr"], 15),
    })

    if pull["returncode"] == 0:
        cfg = await _run_cmd(
            ["nextflow", "config", repo, "-r", version, "-profile", profile],
            timeout=180,
        )
        steps.append({
            "step": "config",
            "success": cfg["returncode"] == 0,
            "output": _tail_text(cfg["stdout"] + cfg["stderr"], 15),
        })
    else:
        notes.append("Skipped config resolution because `nextflow pull` failed.")

    ready = all(s["success"] for s in steps)
    if ready:
        notes.append(
            f"Setup complete. Launch with run_pipeline('{pipeline_name}', '{version}', "
            f"'{profile}', ...) and confirm=true."
        )

    return {
        "pipeline": pipeline_name,
        "version": version,
        "profile": profile,
        "ready": ready,
        "environment": env,
        "steps": steps,
        "notes": notes,
    }


# ===========================================================================
# Tool 3 — run_pipeline
# ===========================================================================

async def run_pipeline(
    pipeline_name: str,
    version: str,
    profile: str,
    samplesheet_path: str,
    outdir: str,
    params: dict[str, Any] | None = None,
    confirm: bool = False,
    resume: bool = False,
    run_name: str | None = None,
    work_dir: str | None = None,
) -> dict[str, Any]:
    """Launch an nf-core pipeline run in the background. Gated by confirm=true.

    This actually executes compute and can incur real cost and time, so it has two
    gates: the environment must be ready, AND confirm must be true. With
    confirm=false (the default) it performs the preflight check and returns the
    exact command for human review WITHOUT running anything.

    The run is launched detached; poll it with get_run_status(run_name).

    Args:
        pipeline_name: nf-core pipeline name.
        version: Pinned release tag (e.g. "3.14.0"). 'main'/'master' are rejected.
        profile: Execution profile (e.g. "docker").
        samplesheet_path: Path to the validated samplesheet CSV.
        outdir: Output directory (local or cloud).
        params: Optional {"--param": value} extra parameters.
        confirm: Must be true to actually launch. False = preview only.
        resume: Add -resume to continue a previous run from cache.
        run_name: Optional run name; auto-generated if omitted.
        work_dir: Optional working directory to launch from.

    Returns (confirm=false):
        {"requires_confirmation": true, "command", "environment_ready", "message", "review_required": true}

    Returns (confirm=true, launched):
        {"run_name","status":"launched","pid","log_file","outdir","command","monitor_with","review_required": true}
    """
    logger.info("Tool call: run_pipeline(%r, confirm=%r)", pipeline_name, confirm)

    # Reuse generate_launch_command for version/profile validation + display string.
    built = await generate_launch_command(
        pipeline_name, version, profile, samplesheet_path, outdir, params
    )
    if built.get("error"):
        return built
    command = built["command"]

    env = await check_execution_environment(profile)
    if not env["ready"]:
        return {
            "error": True,
            "code": "ENV_NOT_READY",
            "message": (
                "Execution environment is not ready. Missing: "
                f"{', '.join(env['missing'])}. See install_hints."
            ),
            "environment": env,
        }
    if env["requested_profile_available"] is False:
        return {
            "error": True,
            "code": "PROFILE_UNAVAILABLE",
            "message": (
                f"Profile '{profile}' is not available on this machine. "
                f"Recommended: '{env['recommended_profile']}'."
            ),
            "environment": env,
        }

    if not confirm:
        return {
            "requires_confirmation": True,
            "command": command,
            "environment_ready": True,
            "message": (
                "Environment is ready and the command is valid, but nothing has run. "
                "Review the command, then call run_pipeline again with confirm=true to launch. "
                "This will consume compute resources."
            ),
            "review_required": True,
        }

    # --- Launch ---
    name = run_name or _default_run_name(pipeline_name)
    args = _build_exec_args(
        pipeline_name, version, profile, samplesheet_path, outdir, params, name, resume
    )

    _RUNS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = _RUNS_DIR / f"{name}.log"

    try:
        pid = _launch_background(args, log_path, work_dir)
    except Exception as exc:
        logger.warning("Failed to launch pipeline: %s", exc)
        return {
            "error": True,
            "code": "LAUNCH_FAILED",
            "message": f"Could not launch pipeline: {exc}",
        }

    meta = {
        "run_name": name,
        "pid": pid,
        "command": command,
        "args": args,
        "outdir": outdir,
        "log_file": str(log_path),
        "status": "running",
        "started_at": time.time(),
        "pipeline": pipeline_name,
        "version": version,
        "profile": profile,
    }
    _save_run(name, meta)

    return {
        "run_name": name,
        "status": "launched",
        "pid": pid,
        "log_file": str(log_path),
        "outdir": outdir,
        "command": command,
        "monitor_with": f"get_run_status('{name}')",
        "review_required": True,
    }


# ===========================================================================
# Tool 4 — get_run_status
# ===========================================================================

async def get_run_status(run_name: str) -> dict[str, Any]:
    """Check the status of a pipeline launched with run_pipeline.

    Determines whether the run is still alive, has completed, or has failed, and
    tails its log. When a run reaches a terminal state, point parse_run_summary at
    its outdir for QC.

    Args:
        run_name: The run_name returned by run_pipeline.

    Returns:
        {
          "run_name","status": "running"|"completed"|"failed"|"unknown",
          "pid","alive","outdir","log_file","started_at",
          "elapsed_seconds","log_tail":[str],"next_steps":[str]
        }
    """
    logger.info("Tool call: get_run_status(%r)", run_name)

    meta = _load_run(run_name)
    if meta is None:
        return {
            "error": True,
            "code": "RUN_NOT_FOUND",
            "message": f"No run named '{run_name}' is tracked. It may not have been launched here.",
        }

    alive = _pid_alive(meta["pid"])
    log_path = Path(meta["log_file"])
    log_text = log_path.read_text(errors="replace") if log_path.exists() else ""
    log_tail = _tail_lines(log_text, _LOG_TAIL_LINES)

    if alive:
        status = "running"
    else:
        status = _scan_log_for_status(log_text)

    # Persist a terminal status so later calls are stable
    if status in ("completed", "failed") and meta.get("status") != status:
        meta["status"] = status
        meta["ended_at"] = time.time()
        _save_run(run_name, meta)

    started = meta.get("started_at", 0)
    elapsed = round((meta.get("ended_at") or time.time()) - started) if started else None

    next_steps = []
    if status == "completed":
        next_steps.append(f"Run parse_run_summary('{meta['outdir']}') to assess QC and results.")
    elif status == "failed":
        next_steps.append("Inspect log_tail for the error. Common causes: missing genome, "
                          "bad samplesheet, or insufficient resources. Fix and re-run with resume=true.")
    elif status == "running":
        next_steps.append(f"Still running. Poll get_run_status('{run_name}') again later.")
    else:
        next_steps.append("Status could not be determined from the log; inspect log_file directly.")

    return {
        "run_name": run_name,
        "status": status,
        "pid": meta["pid"],
        "alive": alive,
        "outdir": meta["outdir"],
        "log_file": meta["log_file"],
        "started_at": started,
        "elapsed_seconds": elapsed,
        "log_tail": log_tail,
        "next_steps": next_steps,
    }


# ===========================================================================
# Subprocess helpers (monkeypatched in tests)
# ===========================================================================

async def _run_cmd(args: list[str], timeout: int = 30, cwd: str | None = None) -> dict[str, Any]:
    """Run a command to completion; never raises. Returns found/returncode/stdout/stderr."""
    try:
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
        )
    except FileNotFoundError:
        return {"found": False, "returncode": 127, "stdout": "", "stderr": "command not found"}
    except Exception as exc:  # noqa: BLE001
        return {"found": False, "returncode": 1, "stdout": "", "stderr": str(exc)}

    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        try:
            proc.kill()
        except Exception:
            pass
        return {"found": True, "returncode": -1, "stdout": "", "stderr": f"timed out after {timeout}s"}

    return {
        "found": True,
        "returncode": proc.returncode,
        "stdout": out.decode(errors="replace"),
        "stderr": err.decode(errors="replace"),
    }


def _launch_background(args: list[str], log_path: Path, work_dir: str | None) -> int:
    """Start a detached background process writing to log_path. Returns its pid."""
    log_file = open(log_path, "wb")
    proc = subprocess.Popen(
        args,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        cwd=work_dir,
        start_new_session=True,  # detach so it survives the server process
    )
    return proc.pid


def _pid_alive(pid: int) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists but owned by another user
    return True


# ===========================================================================
# Probing / recommendation
# ===========================================================================

async def _probe_tool(args: list[str]) -> dict[str, Any]:
    res = await _run_cmd(args, timeout=15)
    installed = res["found"] and res["returncode"] == 0
    text = f"{res['stdout']} {res['stderr']}".strip()
    return {
        "installed": installed,
        "version": _extract_version(text) if installed else None,
    }


def _extract_version(text: str) -> str | None:
    m = re.search(r"(\d+\.\d+(?:\.\d+)?)", text)
    return m.group(1) if m else None


def _recommend_profile(engines: dict[str, Any]) -> str | None:
    docker = engines.get("docker", {})
    if docker.get("installed") and docker.get("daemon_running"):
        return "docker"
    for name in ("singularity", "apptainer", "podman"):
        if engines.get(name, {}).get("installed"):
            return name
    if engines.get("conda", {}).get("installed") or engines.get("mamba", {}).get("installed"):
        return "conda"
    return None


def _profile_available(profile: str, engines: dict[str, Any]) -> bool:
    p = profile.lower()
    if p == "docker":
        d = engines.get("docker", {})
        return bool(d.get("installed") and d.get("daemon_running"))
    if p in engines:
        return bool(engines[p].get("installed"))
    if p == "conda":
        return bool(engines.get("conda", {}).get("installed") or engines.get("mamba", {}).get("installed"))
    # Institutional/test profiles (slurm, aws, test…) don't map to a local engine;
    # assume the user knows their cluster config.
    return True


# ===========================================================================
# Command building
# ===========================================================================

def _build_exec_args(
    pipeline_name: str,
    version: str,
    profile: str,
    samplesheet_path: str,
    outdir: str,
    params: dict[str, Any] | None,
    run_name: str,
    resume: bool,
) -> list[str]:
    args = [
        "nextflow", "run", f"nf-core/{pipeline_name}",
        "-r", version,
        "-profile", profile,
        "-name", run_name,
        "-ansi-log", "false",
        "--input", samplesheet_path,
        "--outdir", outdir,
    ]
    if resume:
        args.append("-resume")
    if params:
        for key, value in params.items():
            flag = key if key.startswith("--") else f"--{key}"
            if isinstance(value, bool):
                if value:
                    args.append(flag)
            else:
                args.extend([flag, str(value)])
    return args


def _default_run_name(pipeline_name: str) -> str:
    return f"{pipeline_name}_{time.strftime('%Y%m%d_%H%M%S')}"


# ===========================================================================
# Run registry
# ===========================================================================

def _run_meta_path(run_name: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", run_name)
    return _RUNS_DIR / f"{safe}.json"


def _save_run(run_name: str, meta: dict[str, Any]) -> None:
    _RUNS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        _run_meta_path(run_name).write_text(json.dumps(meta))
    except Exception as exc:
        logger.warning("Could not persist run metadata for %s: %s", run_name, exc)


def _load_run(run_name: str) -> dict[str, Any] | None:
    path = _run_meta_path(run_name)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


# ===========================================================================
# Log parsing
# ===========================================================================

def _scan_log_for_status(text: str) -> str:
    """Best-effort terminal status from a Nextflow log of a process that has exited."""
    if not text:
        return "unknown"
    lowered = text.lower()
    success_markers = (
        "pipeline completed successfully",
        "workflow completed successfully",
        "completed successfully",
    )
    failure_markers = (
        "execution aborted",
        "pipeline completed with errors",
        "execution cancelled",
        "error ~",
        "command error",
        "caused by:",
    )
    if any(m in lowered for m in success_markers):
        return "completed"
    if any(m in lowered for m in failure_markers):
        return "failed"
    # Nextflow's neutral completion line without an explicit success phrase
    if "goodbye" in lowered or "completed at" in lowered:
        return "completed"
    return "unknown"


def _tail_lines(text: str, n: int) -> list[str]:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    return lines[-n:]


def _tail_text(text: str, n: int) -> str:
    return "\n".join(_tail_lines(text, n))
