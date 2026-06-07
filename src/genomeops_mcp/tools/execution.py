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

from genomeops_mcp.tools.results import generate_launch_command

logger = logging.getLogger(__name__)

_RUNS_DIR = Path.home() / ".cache" / "genomeops-mcp" / "runs"
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
        preview_name = run_name or _default_run_name(pipeline_name)
        merged_params = _build_params(samplesheet_path, outdir, params)
        preview_args = _build_exec_args(
            pipeline_name, version, profile, preview_name,
            _RUNS_DIR / f"{preview_name}.params.json", resume,
        )
        return {
            "requires_confirmation": True,
            "command": _display_command(preview_args),
            "readable_command": command,
            "params_preview": merged_params,
            "environment_ready": True,
            "message": (
                "Environment is ready and the command is valid, but nothing has run. "
                "Review the command AND the parameter values in params_preview "
                "(these will be written to a -params-file), then call run_pipeline "
                "again with confirm=true to launch. This will consume compute resources."
            ),
            "review_required": True,
        }

    # --- Launch ---
    name = run_name or _default_run_name(pipeline_name)
    _RUNS_DIR.mkdir(parents=True, exist_ok=True)

    # Write all run parameters to a -params-file (JSON). This is more robust than
    # a long --flag value chain: it avoids shell-escaping bugs, handles complex
    # values (lists, nested objects), and leaves a clean, reviewable artifact.
    merged_params = _build_params(samplesheet_path, outdir, params)
    params_file = _RUNS_DIR / f"{name}.params.json"
    _write_params_file(merged_params, params_file)

    args = _build_exec_args(
        pipeline_name, version, profile, name, params_file, resume
    )
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
        "command": _display_command(args),
        "readable_command": command,
        "args": args,
        "params": merged_params,
        "params_file": str(params_file),
        "samplesheet_path": samplesheet_path,
        "outdir": outdir,
        "log_file": str(log_path),
        "status": "running",
        "resume": resume,
        # Resume baseline: where it launched (cache/work live here) and a
        # fingerprint of every input file, so diagnose_resume can later tell
        # whether -resume will actually reuse cached work.
        "launch_dir": work_dir or os.getcwd(),
        "input_fingerprints": _fingerprint_inputs(samplesheet_path),
        "started_at": time.time(),
        "pipeline": pipeline_name,
        "version": version,
        "profile": profile,
        # Provenance: record that this run was launched by an agent via this MCP
        # server with explicit human confirmation (nf-core asks for AI transparency).
        "launched_with": "genomeops-mcp",
        "confirmed": True,
        "engine_versions": _engine_versions(env),
    }
    _save_run(name, meta)

    return {
        "run_name": name,
        "status": "launched",
        "pid": pid,
        "log_file": str(log_path),
        "params_file": str(params_file),
        "outdir": outdir,
        "command": _display_command(args),
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
# Tool 5 — generate_methods_note
# ===========================================================================

# Stable, canonical citations for the framework and engine every nf-core run uses.
_NFCORE_CITATION = {
    "tool": "nf-core",
    "citation": (
        "Ewels PA, Peltzer A, Fillinger S, et al. The nf-core framework for "
        "community-curated bioinformatics pipelines. Nat Biotechnol. "
        "2020;38(3):276-278."
    ),
    "doi": "10.1038/s41587-020-0439-x",
}
_NEXTFLOW_CITATION = {
    "tool": "Nextflow",
    "citation": (
        "Di Tommaso P, Chatzou M, Floden EW, et al. Nextflow enables reproducible "
        "computational workflows. Nat Biotechnol. 2017;35(4):316-319."
    ),
    "doi": "10.1038/nbt.3820",
}

_DEV_VERSION_RE = re.compile(r"(dev|alpha|beta|rc\d*)", re.IGNORECASE)


async def generate_methods_note(run_name: str) -> dict[str, Any]:
    """Draft a publication-ready Methods paragraph from a run's provenance record.

    Turns the recorded run manifest (pipeline, pinned version, profile, exact
    parameters, tool versions) into a citable Methods note, plus the canonical
    references for nf-core and Nextflow and a pointer to the pipeline's Zenodo DOI.
    Deterministic — built from the recorded facts, not generated freehand — so it
    is auditable and reproducible. If the run's output directory is available, a
    short QC sentence (sample count, flagged samples) is folded in.

    Run this after a run completes to document exactly what was done.

    Args:
        run_name: The run_name returned by run_pipeline.

    Returns:
        {
          "run_name","methods_text": str,
          "citations": [{"tool","citation","doi"}],
          "parameters_used": {param: value},
          "tool_versions": {tool: version},
          "warnings": [str],
          "review_required": true
        }
    """
    logger.info("Tool call: generate_methods_note(%r)", run_name)

    meta = _load_run(run_name)
    if meta is None:
        return {
            "error": True,
            "code": "RUN_NOT_FOUND",
            "message": f"No run named '{run_name}' is tracked. It may not have been launched here.",
        }

    pipeline = meta.get("pipeline", "unknown")
    version = meta.get("version", "unknown")
    profile = meta.get("profile", "unknown")
    params: dict[str, Any] = meta.get("params", {})
    tool_versions: dict[str, Any] = meta.get("engine_versions", {})

    # Parameters worth reporting: everything except the bare I/O paths.
    reported = {k: v for k, v in params.items() if k not in ("input", "outdir")}

    warnings: list[str] = []
    if version in ("master", "main", "unknown"):
        warnings.append(
            f"This run used '{version}' rather than a pinned release tag — results "
            f"may not be reproducible. Re-run against a fixed version for publication."
        )
    elif _DEV_VERSION_RE.search(version):
        warnings.append(
            f"Version '{version}' is a development/pre-release; cite with caution."
        )

    # Best-effort QC sentence from the output directory.
    qc_sentence = ""
    outdir = meta.get("outdir")
    if outdir:
        try:
            from genomeops_mcp.tools.results import parse_run_summary
            summary = await parse_run_summary(outdir)
            if summary.get("summary_available"):
                n = len(summary.get("samples", []))
                flagged = summary.get("flagged_samples", [])
                qc_sentence = f" Quality control was assessed for {n} sample(s) using MultiQC"
                if flagged:
                    qc_sentence += f"; {len(flagged)} sample(s) were flagged for review"
                qc_sentence += "."
        except Exception as exc:  # noqa: BLE001 — QC is optional enrichment
            logger.debug("parse_run_summary unavailable for methods note: %s", exc)

    # --- Compose the paragraph from recorded facts ---
    nf_version = tool_versions.get("nextflow")
    engine_clause = f"Nextflow v{nf_version} [2]" if nf_version else "Nextflow [2]"

    sentences = [
        f"Sequencing data were processed with the nf-core/{pipeline} pipeline "
        f"(version {version}) [1], implemented in {engine_clause}."
    ]
    sentences.append(
        f"The pipeline was executed with the '{profile}' configuration profile, "
        f"which provisions all per-process software environments automatically."
    )
    if reported:
        param_str = ", ".join(f"--{k} {v}" for k, v in reported.items())
        sentences.append(f"Non-default parameters were: {param_str}.")
    else:
        sentences.append("All parameters were left at the pipeline defaults.")
    if qc_sentence:
        sentences.append(qc_sentence.strip())
    sentences.append(
        "This analysis was launched via the GenomeOps agent tooling with explicit "
        "user confirmation; the complete parameter set is recorded in the run's "
        "params file for reproducibility."
    )

    methods_text = " ".join(sentences)

    citations = [
        {**_NFCORE_CITATION},
        {**_NEXTFLOW_CITATION},
        {
            "tool": f"nf-core/{pipeline}",
            "citation": (
                f"nf-core/{pipeline} pipeline, version {version}. "
                f"Each release is archived with a Zenodo DOI — see "
                f"https://nf-co.re/{pipeline} for the exact citation."
            ),
            "doi": None,
        },
    ]

    return {
        "run_name": run_name,
        "methods_text": methods_text,
        "citations": citations,
        "parameters_used": reported,
        "tool_versions": tool_versions,
        "warnings": warnings,
        "review_required": True,
    }


# ===========================================================================
# Tool 6 — list_runs
# ===========================================================================

async def list_runs() -> dict[str, Any]:
    """List every pipeline run launched through this server, newest first.

    Reads the local run registry. For runs still marked "running", liveness is
    re-checked: if the process has exited, its terminal status (completed/failed)
    is resolved from the log and persisted. Use this to find a run_name to pass to
    get_run_status, generate_methods_note, diagnose_run_failure, or stop_pipeline.

    Returns:
        {
          "runs": [{"run_name","pipeline","version","profile","status",
                    "alive","outdir","started_at","log_file"}],
          "total": int
        }
    """
    logger.info("Tool call: list_runs()")
    _RUNS_DIR.mkdir(parents=True, exist_ok=True)

    runs: list[dict[str, Any]] = []
    for path in _RUNS_DIR.glob("*.json"):
        if path.name.endswith(".params.json"):
            continue
        try:
            meta = json.loads(path.read_text())
        except Exception:
            continue

        status = meta.get("status", "unknown")
        alive: bool | None = None
        if status == "running":
            alive = _pid_alive(meta.get("pid", 0))
            if not alive:
                log_path = Path(meta.get("log_file", ""))
                log_text = log_path.read_text(errors="replace") if log_path.exists() else ""
                status = _scan_log_for_status(log_text)
                if status in ("completed", "failed"):
                    meta["status"] = status
                    meta["ended_at"] = time.time()
                    _save_run(meta["run_name"], meta)

        runs.append({
            "run_name": meta.get("run_name"),
            "pipeline": meta.get("pipeline"),
            "version": meta.get("version"),
            "profile": meta.get("profile"),
            "status": status,
            "alive": alive,
            "outdir": meta.get("outdir"),
            "started_at": meta.get("started_at"),
            "log_file": meta.get("log_file"),
        })

    runs.sort(key=lambda r: r.get("started_at") or 0, reverse=True)
    return {"runs": runs, "total": len(runs)}


# ===========================================================================
# Tool 7 — stop_pipeline
# ===========================================================================

async def stop_pipeline(run_name: str, confirm: bool = False) -> dict[str, Any]:
    """Terminate a running pipeline. Gated by confirm=true.

    Sends SIGTERM to the detached Nextflow process group. With confirm=false (the
    default) it only reports what would be stopped, without signalling anything.
    Stopping a run mid-flight may leave partial/incomplete outputs in the work
    directory; the run can later be continued with run_pipeline(..., resume=true).

    Args:
        run_name: The run_name returned by run_pipeline.
        confirm: Must be true to actually send the termination signal.

    Returns (confirm=false): {"requires_confirmation": true, ...}
    Returns (confirm=true):  {"run_name","status":"stopped","signal_sent",...}
    """
    logger.info("Tool call: stop_pipeline(%r, confirm=%r)", run_name, confirm)

    meta = _load_run(run_name)
    if meta is None:
        return {
            "error": True,
            "code": "RUN_NOT_FOUND",
            "message": f"No run named '{run_name}' is tracked. It may not have been launched here.",
        }

    pid = meta.get("pid", 0)
    if not _pid_alive(pid):
        return {
            "run_name": run_name,
            "status": meta.get("status", "unknown"),
            "already_stopped": True,
            "message": "This run is not currently active; nothing to stop.",
        }

    if not confirm:
        return {
            "requires_confirmation": True,
            "run_name": run_name,
            "pid": pid,
            "message": (
                f"Run '{run_name}' (pid {pid}) is active. Call stop_pipeline again "
                f"with confirm=true to send SIGTERM. Partial outputs may remain; you "
                f"can resume later with run_pipeline(..., resume=true)."
            ),
            "review_required": True,
        }

    sent = _terminate_pid(pid)
    meta["status"] = "stopped"
    meta["ended_at"] = time.time()
    _save_run(run_name, meta)

    return {
        "run_name": run_name,
        "status": "stopped",
        "signal_sent": sent,
        "message": (
            "Termination signal sent." if sent else
            "Could not signal the process (it may have just exited)."
        ),
        "next_steps": [
            f"Resume later with run_pipeline(..., resume=true) using run_name '{run_name}'.",
        ],
        "review_required": True,
    }


# ===========================================================================
# Tool 8 — diagnose_run_failure
# ===========================================================================

async def diagnose_run_failure(run_name: str) -> dict[str, Any]:
    """Explain WHY a failed run failed, and how to fix it.

    Goes beyond tailing the log: locates the failed process and its work directory
    from the Nextflow log, reads the task's .command.err / .command.sh / .exitcode,
    classifies the root cause (out of memory, time limit, disk, container/engine,
    missing input, or a tool-level error), and proposes concrete fixes. When
    ANTHROPIC_API_KEY is set the diagnosis is refined by an LLM reading the actual
    error text; otherwise a fast heuristic is used.

    Run this after get_run_status reports status="failed".

    Args:
        run_name: The run_name returned by run_pipeline.

    Returns:
        {
          "run_name","status",
          "failed_process": str | None,
          "exit_code": int | None,
          "work_dir": str | None,
          "error_excerpt": [str],
          "likely_cause": str,
          "suggested_fixes": [str],
          "command_preview": str | None,
          "analysis_method": "llm" | "heuristic",
          "review_required": true
        }
    """
    logger.info("Tool call: diagnose_run_failure(%r)", run_name)

    meta = _load_run(run_name)
    if meta is None:
        return {
            "error": True,
            "code": "RUN_NOT_FOUND",
            "message": f"No run named '{run_name}' is tracked. It may not have been launched here.",
        }

    log_path = Path(meta.get("log_file", ""))
    log_text = log_path.read_text(errors="replace") if log_path.exists() else ""
    status = meta.get("status", "unknown")
    if status not in ("failed",):
        # Re-derive from the log in case the registry is stale.
        if not _pid_alive(meta.get("pid", 0)):
            status = _scan_log_for_status(log_text)

    parsed = _parse_failure(log_text)

    # Pull the failed task's own files for richer evidence.
    work_files = _read_work_dir(parsed.get("work_dir"))
    error_text = work_files.get("command_err") or "\n".join(parsed.get("error_excerpt", []))
    exit_code = work_files.get("exit_code")
    if exit_code is None:
        exit_code = parsed.get("exit_code")

    # Heuristic baseline (always available); LLM refines if a key is present.
    cause, fixes = _classify_failure(exit_code, error_text)
    method = "heuristic"

    if os.environ.get("ANTHROPIC_API_KEY"):
        refined = await _llm_diagnose(meta, parsed, work_files, error_text, exit_code)
        if refined is not None:
            cause = refined.get("likely_cause", cause)
            fixes = refined.get("suggested_fixes", fixes) or fixes
            method = "llm"

    return {
        "run_name": run_name,
        "status": status,
        "failed_process": parsed.get("failed_process"),
        "exit_code": exit_code,
        "work_dir": parsed.get("work_dir"),
        "error_excerpt": _tail_lines(error_text, 25) if error_text else [],
        "likely_cause": cause,
        "suggested_fixes": fixes,
        "command_preview": work_files.get("command_sh"),
        "analysis_method": method,
        "review_required": True,
    }


# ---------------------------------------------------------------------------
# Failure parsing & classification
# ---------------------------------------------------------------------------

_PROCESS_RE = re.compile(r"Error executing process\s*>\s*'?([^'\n]+)'?")
_EXIT_INLINE_RE = re.compile(r"exit status\s*\(?(\d+)\)?", re.IGNORECASE)
_WORKDIR_RE = re.compile(r"Work dir:\s*\n\s*(\S+)", re.IGNORECASE)


def _parse_failure(log_text: str) -> dict[str, Any]:
    """Extract failed process, exit code, work dir, and error block from a NF log."""
    out: dict[str, Any] = {
        "failed_process": None,
        "exit_code": None,
        "work_dir": None,
        "error_excerpt": [],
    }
    if not log_text:
        return out

    m = _PROCESS_RE.search(log_text)
    if m:
        out["failed_process"] = m.group(1).strip()

    m = _EXIT_INLINE_RE.search(log_text)
    if m:
        try:
            out["exit_code"] = int(m.group(1))
        except ValueError:
            pass

    m = _WORKDIR_RE.search(log_text)
    if m:
        out["work_dir"] = m.group(1).strip()

    out["error_excerpt"] = _extract_section(log_text, "Command error:")
    return out


def _extract_section(log_text: str, header: str) -> list[str]:
    """Return the indented lines following a 'Header:' marker in a Nextflow error."""
    lines = log_text.splitlines()
    collected: list[str] = []
    capturing = False
    for line in lines:
        if header.lower() in line.lower():
            capturing = True
            continue
        if capturing:
            # Section ends at the next non-indented, non-empty line.
            if line and not line.startswith((" ", "\t")):
                break
            if line.strip():
                collected.append(line.strip())
    return collected


def _read_work_dir(work_dir: str | None) -> dict[str, Any]:
    """Read the failed task's .command.err/.command.sh/.exitcode if accessible."""
    out: dict[str, Any] = {"command_err": None, "command_sh": None, "exit_code": None}
    if not work_dir:
        return out
    base = Path(work_dir)
    if not base.exists():
        return out

    err = base / ".command.err"
    if err.exists():
        try:
            out["command_err"] = _tail_text(err.read_text(errors="replace"), 40)
        except OSError:
            pass

    sh = base / ".command.sh"
    if sh.exists():
        try:
            out["command_sh"] = _tail_text(sh.read_text(errors="replace"), 30)
        except OSError:
            pass

    code = base / ".exitcode"
    if code.exists():
        try:
            out["exit_code"] = int(code.read_text(errors="replace").strip())
        except (OSError, ValueError):
            pass

    return out


def _classify_failure(exit_code: int | None, error_text: str | None) -> tuple[str, list[str]]:
    """Heuristic root-cause + concrete fixes from exit code and error text."""
    text = (error_text or "").lower()

    if exit_code == 137 or any(s in text for s in ("out of memory", "oom", "killed", "memory limit", "exceeds available memory")):
        return (
            "Out of memory — the process was killed for exceeding its memory allocation.",
            [
                "Increase the memory for the failing process via a custom config "
                "(process.withName:'<PROCESS>' { memory = '32 GB' }) or raise --max_memory.",
                "Run on a machine/queue with more RAM, or reduce the number of parallel tasks.",
                "Re-run with run_pipeline(..., resume=true) so completed tasks are not repeated.",
            ],
        )

    if exit_code in (140, 143) or any(s in text for s in ("time limit", "walltime", "killed by signal 15", "exceeded the time")):
        return (
            "Time limit exceeded — the scheduler or engine stopped the process.",
            [
                "Increase the time budget for the process (process.time) or raise --max_time.",
                "Resume with run_pipeline(..., resume=true) to continue from cached tasks.",
            ],
        )

    if "no space left on device" in text or "disk quota exceeded" in text:
        return (
            "Out of disk space in the work directory.",
            [
                "Free space or point the work directory at a larger volume "
                "(run_pipeline(..., work_dir=...)).",
                "Remove old work directories once runs are summarised.",
            ],
        )

    if any(s in text for s in ("unable to find image", "manifest unknown", "pull access denied",
                               "cannot connect to the docker daemon", "error pulling image",
                               "failed to pull")):
        return (
            "Container image or engine problem — the required image could not be obtained.",
            [
                "Check the container engine is running (check_execution_environment) and has network access.",
                "Verify the chosen -profile matches an available engine.",
                "Retry setup_environment to pre-pull the pipeline and resolve config.",
            ],
        )

    if "command not found" in text:
        return (
            "A tool was not found in the process environment.",
            [
                "Ensure you are using a container/conda profile (-profile docker|singularity|conda), "
                "not running tools from the host.",
                "Re-run setup_environment to resolve the pipeline configuration.",
            ],
        )

    if any(s in text for s in ("no such file", "does not exist", "cannot open", "not found")):
        return (
            "A required input or reference file was missing.",
            [
                "Check the file paths in your samplesheet are correct and accessible "
                "(validate_samplesheet).",
                "Confirm the reference (--genome or --fasta) is set and reachable.",
            ],
        )

    if exit_code is not None and exit_code != 0:
        return (
            f"A tool exited with a non-zero status ({exit_code}) — a tool-level error.",
            [
                "Inspect error_excerpt and command_preview for the specific message.",
                "Common causes: malformed input, an unsupported parameter combination, "
                "or a corrupt input file. Fix and resume with run_pipeline(..., resume=true).",
            ],
        )

    return (
        "Cause could not be determined automatically.",
        [
            "Inspect the work_dir's .command.err and .command.log directly.",
            "Re-run get_run_status for the full log tail, or set ANTHROPIC_API_KEY "
            "for an AI-assisted diagnosis.",
        ],
    )


async def _llm_diagnose(
    meta: dict[str, Any],
    parsed: dict[str, Any],
    work_files: dict[str, Any],
    error_text: str | None,
    exit_code: int | None,
) -> dict[str, Any] | None:
    """Refine the failure diagnosis via Claude reading the actual error. None on failure."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    try:
        import anthropic
    except ImportError:
        return None

    prompt = f"""You are a Nextflow/nf-core troubleshooting expert. A pipeline run failed.
Diagnose the root cause and give concrete, actionable fixes.

Pipeline: nf-core/{meta.get('pipeline')} (version {meta.get('version')})
Profile: {meta.get('profile')}
Failed process: {parsed.get('failed_process')}
Exit code: {exit_code}

--- Command that was run (.command.sh) ---
{work_files.get('command_sh') or '(unavailable)'}

--- Error output (.command.err / log) ---
{error_text or '(no error text captured)'}

Respond with ONLY valid JSON:
{{
  "likely_cause": "<one sentence root cause>",
  "suggested_fixes": ["<concrete fix>", "<concrete fix>"]
}}"""

    try:
        import json as _json
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model="claude-opus-4-8",
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = message.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        parsed_resp = _json.loads(raw.strip())
        if not isinstance(parsed_resp.get("suggested_fixes"), list):
            return None
        return parsed_resp
    except Exception as exc:  # noqa: BLE001 — degrade to heuristic
        logger.warning("LLM diagnosis failed for %s: %s", meta.get("run_name"), exc)
        return None


# ===========================================================================
# Tool 9 — diagnose_resume
# ===========================================================================

async def diagnose_resume(run_name: str) -> dict[str, Any]:
    """Predict whether `-resume` will actually reuse cached work, and why not.

    Nextflow computes each task's cache key from the full path, last-modified time,
    and size of its inputs, and needs the .nextflow cache plus the work directory
    intact. A single changed input or a deleted work dir silently forces a full
    re-run — a notorious source of confusion. This checks all three up front:
    the cache directory, the work directory, and whether any recorded input file
    has changed since the run was launched.

    Run this before re-launching with run_pipeline(..., resume=true).

    Args:
        run_name: The run_name returned by run_pipeline.

    Returns:
        {
          "run_name",
          "resume_will_reuse_cache": bool,
          "cache_present": bool,
          "work_dir_present": bool,
          "changed_inputs": [{"path","reason"}],
          "issues": [str],
          "recommendations": [str],
          "review_required": true
        }
    """
    logger.info("Tool call: diagnose_resume(%r)", run_name)

    meta = _load_run(run_name)
    if meta is None:
        return {
            "error": True,
            "code": "RUN_NOT_FOUND",
            "message": f"No run named '{run_name}' is tracked. It may not have been launched here.",
        }

    launch_dir = Path(meta.get("launch_dir") or ".")
    cache_present = (launch_dir / ".nextflow").exists()
    work_dir_present = (launch_dir / "work").exists()

    changed = _compare_fingerprints(meta.get("input_fingerprints", {}))

    issues: list[str] = []
    if not cache_present:
        issues.append(
            f"Nextflow cache (.nextflow/) not found in the launch directory "
            f"({launch_dir}) — there is no cache to resume from; the run will start over."
        )
    if not work_dir_present:
        issues.append(
            f"Work directory ({launch_dir / 'work'}) not found — cached task outputs "
            f"are gone, so resume will re-execute everything."
        )
    for ch in changed:
        issues.append(f"Input changed: {ch['path']} ({ch['reason']}) — tasks using it will re-run.")

    resume_ok = cache_present and work_dir_present and not changed

    recommendations: list[str] = []
    if resume_ok:
        recommendations.append(
            "Cache, work directory, and inputs are intact — "
            "run_pipeline(..., resume=true) should reuse completed tasks."
        )
    else:
        if not cache_present or not work_dir_present:
            recommendations.append(
                "Preserve both .nextflow/ and work/ between runs to enable resume; "
                "without them a fresh run is required."
            )
        if changed:
            recommendations.append(
                "Restore the original input files (same path, contents, and timestamp) "
                "or accept that the affected steps will re-run. Re-generating a samplesheet "
                "changes its timestamp even if contents are identical."
            )

    return {
        "run_name": run_name,
        "resume_will_reuse_cache": resume_ok,
        "cache_present": cache_present,
        "work_dir_present": work_dir_present,
        "changed_inputs": changed,
        "issues": issues,
        "recommendations": recommendations,
        "review_required": True,
    }


# ---------------------------------------------------------------------------
# Input fingerprinting (for resume diagnosis)
# ---------------------------------------------------------------------------

def _fingerprint_file(path: str) -> dict[str, Any] | None:
    """Return {size, mtime} for a local file, or None if it isn't an accessible file."""
    try:
        st = Path(path).stat()
    except OSError:
        return None
    return {"size": st.st_size, "mtime": round(st.st_mtime, 3)}


def _fingerprint_inputs(samplesheet_path: str) -> dict[str, dict[str, Any]]:
    """Fingerprint the samplesheet and every local file path referenced inside it.

    Nextflow's resume cache is sensitive to the path/mtime/size of every input, so
    we record the samplesheet itself plus the data files it points to. Cloud paths
    and unreadable cells are skipped.
    """
    prints: dict[str, dict[str, Any]] = {}
    fp = _fingerprint_file(samplesheet_path)
    if fp:
        prints[samplesheet_path] = fp

    # Scan the samplesheet for referenced local file paths.
    try:
        text = Path(samplesheet_path).read_text(errors="replace")
    except OSError:
        return prints

    for cell in re.split(r"[,\t\n\r]+", text):
        cell = cell.strip().strip('"').strip("'")
        if not cell or "://" in cell:
            continue
        # Only treat things that look like file paths and actually exist.
        if ("/" in cell or "." in cell) and Path(cell).is_file():
            cfp = _fingerprint_file(cell)
            if cfp:
                prints[cell] = cfp
    return prints


def _compare_fingerprints(recorded: dict[str, dict[str, Any]]) -> list[dict[str, str]]:
    """Compare recorded input fingerprints against the current filesystem state."""
    changed: list[dict[str, str]] = []
    for path, old in (recorded or {}).items():
        now = _fingerprint_file(path)
        if now is None:
            changed.append({"path": path, "reason": "file no longer exists"})
            continue
        if now["size"] != old.get("size"):
            changed.append({"path": path, "reason": "size changed"})
        elif now["mtime"] != old.get("mtime"):
            changed.append({"path": path, "reason": "modification time changed"})
    return changed


# ===========================================================================
# Tool 10 — estimate_resources
# ===========================================================================

# Rough, deliberately conservative resource profiles for common nf-core pipelines.
#   peak_mem_gb     — memory the single heaviest process typically wants (drives --max_memory)
#   cores           — a comfortable CPU count for reasonable throughput
#   disk_mult       — work/ + results size as a multiple of total input size
#   hours_per_sample — very rough wall-clock per sample at the given core count
# These are planning heuristics, not benchmarks; the LLM path (and the host agent)
# can refine them with knowledge of genome size, read depth, and tool choices.
_PIPELINE_RESOURCE_PROFILES: dict[str, dict[str, Any]] = {
    "rnaseq":     {"peak_mem_gb": 40, "cores": 12, "disk_mult": 8,  "hours_per_sample": 1.5, "note": "STAR genome index loading dominates memory (~38 GB for human)."},
    "sarek":      {"peak_mem_gb": 36, "cores": 16, "disk_mult": 12, "hours_per_sample": 6.0, "note": "Variant calling (GATK/BWA) is CPU- and time-heavy; WGS far exceeds WES."},
    "chipseq":    {"peak_mem_gb": 24, "cores": 8,  "disk_mult": 6,  "hours_per_sample": 1.0, "note": "BWA alignment + peak calling; memory scales with genome size."},
    "atacseq":    {"peak_mem_gb": 24, "cores": 8,  "disk_mult": 6,  "hours_per_sample": 1.0, "note": "Similar profile to chipseq."},
    "methylseq":  {"peak_mem_gb": 32, "cores": 12, "disk_mult": 8,  "hours_per_sample": 3.0, "note": "Bisulfite alignment (Bismark) is memory- and time-intensive."},
    "viralrecon": {"peak_mem_gb": 12, "cores": 6,  "disk_mult": 4,  "hours_per_sample": 0.5, "note": "Small viral genomes; light compared to human pipelines."},
    "scrnaseq":   {"peak_mem_gb": 32, "cores": 12, "disk_mult": 6,  "hours_per_sample": 2.0, "note": "Single-cell alignment/quantification; memory scales with the index."},
    "taxprofiler":{"peak_mem_gb": 48, "cores": 16, "disk_mult": 5,  "hours_per_sample": 2.0, "note": "Large reference databases can dominate memory (often >40 GB)."},
    "mag":        {"peak_mem_gb": 64, "cores": 16, "disk_mult": 15, "hours_per_sample": 8.0, "note": "Metagenome assembly is extremely memory- and disk-hungry."},
}

_DEFAULT_RESOURCE_PROFILE = {
    "peak_mem_gb": 32, "cores": 8, "disk_mult": 8, "hours_per_sample": 2.0,
    "note": "No specific profile for this pipeline; using conservative generic defaults.",
}


async def estimate_resources(
    pipeline_name: str,
    version: str,
    samplesheet_path: str | None = None,
    file_paths: list[str] | None = None,
    data_summary: dict[str, Any] | None = None,
    profile: str | None = None,
) -> dict[str, Any]:
    """Estimate the compute a run will need, and judge whether this machine can take it.

    Bridges check_execution_environment (what the machine HAS) and run_pipeline
    (what the job NEEDS). From the workload — sample count and total input size,
    drawn from the samplesheet, raw file_paths, or a data_summary — plus the
    pipeline's known resource profile, it proposes rough CPU / memory / disk /
    walltime figures, compares them against the detected machine, and returns a
    verdict (sufficient / marginal / insufficient) with concrete nf-core
    --max_cpus/--max_memory/--max_time overrides sized to what's actually available.

    Resource needs are highly data-dependent, so this is a planning aid, never a
    guarantee — review_required is always true. When ANTHROPIC_API_KEY is set the
    estimate is refined by an LLM; otherwise a heuristic is used. Either way the
    raw evidence is returned in reasoning_inputs so an agent host can reason itself.

    Run this after check_execution_environment and before setup_environment.

    Args:
        pipeline_name: nf-core pipeline name (e.g. "rnaseq").
        version: Pinned release tag (e.g. "3.14.0").
        samplesheet_path: Optional path to the samplesheet — rows are counted and
            referenced local files are sized.
        file_paths: Optional raw input file paths to size, if no samplesheet yet.
        data_summary: Optional pre-computed summary (e.g. from check_feasibility):
            may include sample_count, paired_end, total_input_gb.
        profile: Optional execution profile (informational).

    Returns:
        {
          "pipeline","version",
          "input_signals": {"sample_count","total_input_gb","paired_end","source"},
          "detected_machine": {"cpu_count","total_memory_gb","available_memory_gb","disk_free_gb"},
          "estimated_requirements": {"recommended_cpus","recommended_memory_gb",
              "estimated_peak_memory_gb","estimated_disk_gb","estimated_walltime_hours","basis"},
          "machine_verdict": "sufficient"|"marginal"|"insufficient",
          "verdict_reasons": [str],
          "suggested_overrides": {"--max_cpus","--max_memory","--max_time"},
          "analysis_method": "llm"|"heuristic",
          "confidence": "low"|"medium"|"high",
          "caveats": [str],
          "reasoning_inputs": {...},
          "review_required": true
        }
    """
    logger.info("Tool call: estimate_resources(%r, %r)", pipeline_name, version)

    signals = _gather_input_signals(samplesheet_path, file_paths, data_summary)
    machine = _probe_machine()

    estimate = _heuristic_estimate(pipeline_name, signals, machine)
    verdict, reasons = _machine_verdict(estimate, machine)
    overrides = _suggest_overrides(estimate, machine)
    method = "heuristic"
    confidence = "low"

    caveats = [
        "Resource needs are highly data-dependent (genome size, read depth, sample "
        "heterogeneity); treat these as planning estimates, not guarantees.",
        "Estimates assume a single pipeline run on this machine with no competing workloads.",
    ]
    if signals.get("total_input_gb") is None:
        caveats.append(
            "Total input size could not be measured (no readable local files); "
            "disk and walltime estimates are especially rough."
        )
    if signals.get("sample_count") is None:
        caveats.append(
            "Sample count is unknown; walltime scales with sample count, so the "
            "runtime figure assumes a single sample."
        )

    if os.environ.get("ANTHROPIC_API_KEY"):
        refined = await _llm_estimate(pipeline_name, version, signals, machine, estimate)
        if refined is not None:
            estimate = refined.get("estimated_requirements", estimate) or estimate
            verdict = refined.get("machine_verdict", verdict)
            reasons = refined.get("verdict_reasons", reasons) or reasons
            overrides = refined.get("suggested_overrides", overrides) or overrides
            confidence = refined.get("confidence", "medium")
            extra = refined.get("caveats")
            if isinstance(extra, list):
                caveats.extend(c for c in extra if c not in caveats)
            method = "llm"

    return {
        "pipeline": pipeline_name,
        "version": version,
        "input_signals": signals,
        "detected_machine": machine,
        "estimated_requirements": estimate,
        "machine_verdict": verdict,
        "verdict_reasons": reasons,
        "suggested_overrides": overrides,
        "analysis_method": method,
        "confidence": confidence,
        "caveats": caveats,
        # Evidence block: lets an agent host do its own scaling reasoning even when
        # no ANTHROPIC_API_KEY is set and only the heuristic ran server-side.
        "reasoning_inputs": {
            "pipeline": pipeline_name,
            "version": version,
            "input_signals": signals,
            "detected_machine": machine,
            "pipeline_profile": _PIPELINE_RESOURCE_PROFILES.get(
                pipeline_name.lower(), _DEFAULT_RESOURCE_PROFILE
            ),
        },
        "review_required": True,
    }


# ---------------------------------------------------------------------------
# Resource estimation helpers
# ---------------------------------------------------------------------------

def _probe_machine() -> dict[str, Any]:
    """Detect this machine's CPU count, total/available memory, and free disk.

    Best-effort and never raises; any field that can't be determined is None.
    """
    cpu = os.cpu_count() or 1

    total_mem_gb: float | None = None
    try:
        page = os.sysconf("SC_PAGE_SIZE")
        phys = os.sysconf("SC_PHYS_PAGES")
        total_mem_gb = round(page * phys / 1e9, 1)
    except (ValueError, OSError, AttributeError):
        pass

    avail_mem_gb: float | None = None
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                avail_mem_gb = round(int(line.split()[1]) / 1e6, 1)  # kB → GB
                break
    except (OSError, ValueError, IndexError):
        pass
    if avail_mem_gb is None:
        try:
            page = os.sysconf("SC_PAGE_SIZE")
            avail = os.sysconf("SC_AVPHYS_PAGES")
            avail_mem_gb = round(page * avail / 1e9, 1)
        except (ValueError, OSError, AttributeError):
            pass

    disk_free_gb: float | None = None
    try:
        disk_free_gb = round(shutil.disk_usage(Path.cwd()).free / 1e9, 1)
    except OSError:
        pass

    return {
        "cpu_count": cpu,
        "total_memory_gb": total_mem_gb,
        "available_memory_gb": avail_mem_gb,
        "disk_free_gb": disk_free_gb,
    }


def _scan_samplesheet_inputs(samplesheet_path: str) -> tuple[int | None, int]:
    """Return (data_row_count, total_bytes_of_referenced_local_files) from a samplesheet."""
    try:
        text = Path(samplesheet_path).read_text(errors="replace")
    except OSError:
        return None, 0

    lines = [ln for ln in text.splitlines() if ln.strip()]
    row_count = max(len(lines) - 1, 0) if lines else 0  # minus the header row

    total_bytes = 0
    for cell in re.split(r"[,\t\n\r]+", text):
        cell = cell.strip().strip('"').strip("'")
        if not cell or "://" in cell:
            continue
        if ("/" in cell or "." in cell) and Path(cell).is_file():
            fp = _fingerprint_file(cell)
            if fp:
                total_bytes += fp["size"]
    return (row_count or None), total_bytes


def _gather_input_signals(
    samplesheet_path: str | None,
    file_paths: list[str] | None,
    data_summary: dict[str, Any] | None,
) -> dict[str, Any]:
    """Derive sample_count / total_input_gb / paired_end from whatever was provided."""
    signals: dict[str, Any] = {
        "sample_count": None,
        "total_input_gb": None,
        "paired_end": None,
        "source": None,
    }

    # 1. An explicit data_summary (e.g. from check_feasibility) is authoritative.
    if data_summary:
        if data_summary.get("sample_count") is not None:
            signals["sample_count"] = data_summary.get("sample_count")
        if data_summary.get("paired_end") is not None:
            signals["paired_end"] = data_summary.get("paired_end")
        if data_summary.get("total_input_gb") is not None:
            signals["total_input_gb"] = data_summary.get("total_input_gb")
        signals["source"] = "data_summary"

    # 2. The samplesheet: count rows and size the local files it references.
    if samplesheet_path and Path(samplesheet_path).is_file():
        rows, total_bytes = _scan_samplesheet_inputs(samplesheet_path)
        if signals["sample_count"] is None and rows is not None:
            signals["sample_count"] = rows
        if signals["total_input_gb"] is None and total_bytes:
            signals["total_input_gb"] = round(total_bytes / 1e9, 2)
        signals["source"] = signals["source"] or "samplesheet"

    # 3. Raw file paths, if neither of the above gave us a size.
    if file_paths:
        total_bytes = 0
        found = False
        for p in file_paths:
            fp = _fingerprint_file(p)
            if fp:
                total_bytes += fp["size"]
                found = True
        if signals["total_input_gb"] is None and found:
            signals["total_input_gb"] = round(total_bytes / 1e9, 2)
        signals["source"] = signals["source"] or "file_paths"

    return signals


def _heuristic_estimate(
    pipeline_name: str,
    signals: dict[str, Any],
    machine: dict[str, Any],
) -> dict[str, Any]:
    """Rough resource requirements from the pipeline profile + input signals."""
    profile = _PIPELINE_RESOURCE_PROFILES.get(
        pipeline_name.lower(), _DEFAULT_RESOURCE_PROFILE
    )
    sample_count = signals.get("sample_count") or 1
    total_input_gb = signals.get("total_input_gb")

    rec_mem = profile["peak_mem_gb"]
    rec_cpus = profile["cores"]

    est_disk: float | None = None
    if total_input_gb:
        est_disk = max(round(total_input_gb * profile["disk_mult"], 1), 10.0)

    # Walltime: samples run with some parallelism, limited by available cores.
    per_sample_cores = 4
    usable_cpus = machine.get("cpu_count") or rec_cpus
    parallel = max(1, min(sample_count, max(usable_cpus // per_sample_cores, 1)))
    est_hours = round(profile["hours_per_sample"] * sample_count / parallel, 1)

    return {
        "recommended_cpus": rec_cpus,
        "recommended_memory_gb": rec_mem,
        "estimated_peak_memory_gb": rec_mem,
        "estimated_disk_gb": est_disk,
        "estimated_walltime_hours": est_hours,
        "basis": profile["note"],
    }


def _machine_verdict(
    estimate: dict[str, Any],
    machine: dict[str, Any],
) -> tuple[str, list[str]]:
    """Compare the estimate against the detected machine. Returns (verdict, reasons)."""
    reasons: list[str] = []
    verdict = "sufficient"

    def _downgrade(to: str) -> None:
        nonlocal verdict
        order = {"sufficient": 0, "marginal": 1, "insufficient": 2}
        if order[to] > order[verdict]:
            verdict = to

    mem = machine.get("total_memory_gb")
    cpus = machine.get("cpu_count")
    disk = machine.get("disk_free_gb")
    need_mem = estimate.get("recommended_memory_gb")
    need_cpu = estimate.get("recommended_cpus")
    need_disk = estimate.get("estimated_disk_gb")

    if mem is not None and need_mem is not None:
        if mem < need_mem:
            _downgrade("insufficient")
            reasons.append(
                f"Detected {mem} GB RAM but the heaviest process typically wants "
                f"~{need_mem} GB — high risk of OOM kills (exit 137)."
            )
        elif mem < need_mem * 1.25:
            _downgrade("marginal")
            reasons.append(
                f"{mem} GB RAM is close to the ~{need_mem} GB peak need; expect "
                f"swapping or occasional OOM under load."
            )

    if cpus is not None and need_cpu is not None:
        if cpus < need_cpu / 2:
            _downgrade("marginal")
            reasons.append(
                f"Only {cpus} cores detected vs a recommended ~{need_cpu}; "
                f"runtime will be substantially longer."
            )

    if disk is not None and need_disk is not None:
        if disk < need_disk:
            _downgrade("insufficient")
            reasons.append(
                f"Only {disk} GB free disk but ~{need_disk} GB may be needed for "
                f"work/ + results — risk of 'no space left on device'."
            )
        elif disk < need_disk * 1.5:
            _downgrade("marginal")
            reasons.append(
                f"{disk} GB free disk is tight against an estimated ~{need_disk} GB need."
            )

    if verdict == "sufficient":
        reasons.append("Detected resources meet the estimated requirements with headroom.")
    return verdict, reasons


def _suggest_overrides(
    estimate: dict[str, Any],
    machine: dict[str, Any],
) -> dict[str, Any]:
    """nf-core --max_* caps sized to what the machine can actually offer."""
    overrides: dict[str, Any] = {}
    cpus = machine.get("cpu_count")
    mem = machine.get("total_memory_gb")

    if cpus:
        overrides["--max_cpus"] = cpus
    if mem:
        # Leave ~15% headroom for the OS and Nextflow itself.
        overrides["--max_memory"] = f"{max(int(mem * 0.85), 1)}.GB"
    walltime = estimate.get("estimated_walltime_hours")
    if walltime:
        # Give generous slack (2x) so a slightly-slow run isn't killed.
        overrides["--max_time"] = f"{max(int(walltime * 2), 1)}.h"
    return overrides


async def _llm_estimate(
    pipeline_name: str,
    version: str,
    signals: dict[str, Any],
    machine: dict[str, Any],
    heuristic: dict[str, Any],
) -> dict[str, Any] | None:
    """Refine the resource estimate via Claude. Returns None on any failure."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    try:
        import anthropic
    except ImportError:
        return None

    prompt = f"""You are an HPC and bioinformatics resource-planning expert. Estimate the
compute resources an nf-core pipeline run will need, and judge whether the detected
machine can handle it.

Pipeline: nf-core/{pipeline_name} (version {version})

Workload signals:
- Sample count: {signals.get('sample_count')}
- Total input size (GB): {signals.get('total_input_gb')}
- Paired-end: {signals.get('paired_end')}

Detected machine:
- CPU cores: {machine.get('cpu_count')}
- Total RAM (GB): {machine.get('total_memory_gb')}
- Available RAM (GB): {machine.get('available_memory_gb')}
- Free disk (GB): {machine.get('disk_free_gb')}

A heuristic baseline (refine or correct it using your knowledge of this pipeline's
heaviest steps, genome size, and read-depth scaling):
{heuristic}

Reason about the pipeline's heaviest process (e.g. genome index loading, assembly,
variant calling), how runtime scales with sample count and read depth, and how
work-directory disk usage relates to input size. Then give a verdict.

Respond with ONLY valid JSON:
{{
  "estimated_requirements": {{
    "recommended_cpus": <int>,
    "recommended_memory_gb": <int>,
    "estimated_peak_memory_gb": <int>,
    "estimated_disk_gb": <number or null>,
    "estimated_walltime_hours": <number>,
    "basis": "<one sentence on what drives the estimate>"
  }},
  "machine_verdict": "sufficient|marginal|insufficient",
  "verdict_reasons": ["<reason>", "..."],
  "suggested_overrides": {{"--max_cpus": <int>, "--max_memory": "<N>.GB", "--max_time": "<N>.h"}},
  "confidence": "low|medium|high",
  "caveats": ["<caveat>", "..."]
}}"""

    try:
        import json as _json
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model="claude-opus-4-8",
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = message.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        parsed = _json.loads(raw.strip())
        if parsed.get("machine_verdict") not in ("sufficient", "marginal", "insufficient"):
            return None
        return parsed
    except Exception as exc:  # noqa: BLE001 — degrade gracefully to heuristic
        logger.warning("LLM resource estimate failed for %s: %s", pipeline_name, exc)
        return None


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


def _terminate_pid(pid: int) -> bool:
    """Send SIGTERM to a detached run's process group. Returns True if signalled."""
    if not pid:
        return False
    try:
        # Runs are launched with start_new_session=True, so the pid leads its own
        # process group — terminate the whole group so child tasks die too.
        os.killpg(os.getpgid(pid), signal.SIGTERM)
        return True
    except ProcessLookupError:
        return False
    except Exception:  # noqa: BLE001 — fall back to a direct signal
        try:
            os.kill(pid, signal.SIGTERM)
            return True
        except Exception:
            return False


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

def _build_params(
    samplesheet_path: str,
    outdir: str,
    params: dict[str, Any] | None,
) -> dict[str, Any]:
    """Assemble the full parameter dict written to the -params-file.

    The samplesheet (input) and outdir are pipeline parameters like any other, so
    they live in the params file too. Leading dashes on keys (e.g. "--genome") are
    stripped, since the params file uses bare parameter names. False booleans are
    kept verbatim — unlike on the command line, an explicit false in a params file
    is meaningful (it overrides a true default).
    """
    merged: dict[str, Any] = {"input": samplesheet_path, "outdir": outdir}
    if params:
        for key, value in params.items():
            merged[key.lstrip("-")] = value
    return merged


def _write_params_file(params: dict[str, Any], path: Path) -> None:
    """Write the parameter dict as JSON for Nextflow's -params-file (accepts JSON or YAML)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(params, indent=2, default=str))


def _build_exec_args(
    pipeline_name: str,
    version: str,
    profile: str,
    run_name: str,
    params_file: Path,
    resume: bool,
) -> list[str]:
    args = [
        "nextflow", "run", f"nf-core/{pipeline_name}",
        "-r", version,
        "-profile", profile,
        "-name", run_name,
        "-ansi-log", "false",
        "-params-file", str(params_file),
    ]
    if resume:
        args.append("-resume")
    return args


def _display_command(args: list[str]) -> str:
    """Render exec args as a readable, line-wrapped command string.

    Groups each flag with its value on one line (e.g. "-r 3.14.0") and keeps the
    leading "nextflow run nf-core/<pipeline>" together on the first line.
    """
    if not args:
        return ""
    lines: list[str] = []
    current: list[str] = []
    for tok in args:
        if tok.startswith("-") and current:
            lines.append(" ".join(current))
            current = [tok]
        else:
            current.append(tok)
    if current:
        lines.append(" ".join(current))
    return " \\\n  ".join(lines)


def _engine_versions(env: dict[str, Any]) -> dict[str, Any]:
    """Pull installed tool versions out of a check_execution_environment result."""
    versions: dict[str, Any] = {}
    nf = env.get("nextflow") or {}
    if nf.get("version"):
        versions["nextflow"] = nf["version"]
    java = env.get("java") or {}
    if java.get("version"):
        versions["java"] = java["version"]
    for name, info in (env.get("engines") or {}).items():
        if info.get("installed") and info.get("version"):
            versions[name] = info["version"]
    return versions


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
