"""Functions registered with Globus Compute and executed on the HPC login node.

Every function here must be safe to run on a login node:
  - filesystem operations
  - scheduler operations (qsub, qstat, qdel) with throttling
  - code builds

Functions are intentionally narrow. There is no generic `run_shell` exposed
to the agent. If a new capability is needed, add a named function with the
same return-shape conventions below.

Return shape: every function returns a dict containing at least:
    ok:        bool, True if the function ran to completion without errors
    error:     str | None, error class + message when ok=False
    stdout:    str, captured stdout (may be truncated for large outputs)
    stderr:    str, captured stderr
    duration_s: float, wall time for the operation
    plus per-function result fields described in each docstring.

Functions are designed to be self-contained where reasonable so they
serialise cleanly through Globus Compute, but they may import from
agentic.* because the package is installed in the endpoint venv.
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from typing import Iterable

_STDOUT_TRUNCATE_BYTES = 64 * 1024  # 64 KiB cap on returned streams


def _truncate(s: str, limit: int = _STDOUT_TRUNCATE_BYTES) -> str:
    if len(s) <= limit:
        return s
    head = limit // 2
    tail = limit - head - 64
    return f"{s[:head]}\n... [truncated {len(s) - limit} chars] ...\n{s[-tail:]}"


def _run(
    cmd: list[str],
    cwd: str | None = None,
    env: dict | None = None,
    timeout_s: float = 1800.0,
) -> dict:
    """subprocess wrapper that enforces login-node policy and shapes the return."""
    from agentic.safety import assert_login_node_safe

    assert_login_node_safe(cmd)
    start = time.monotonic()
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            env=full_env,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        duration = time.monotonic() - start
        return {
            "ok": proc.returncode == 0,
            "error": None if proc.returncode == 0 else f"exit_code={proc.returncode}",
            "stdout": _truncate(proc.stdout or ""),
            "stderr": _truncate(proc.stderr or ""),
            "exit_code": proc.returncode,
            "duration_s": duration,
            "cmd": cmd,
            "cwd": cwd,
        }
    except subprocess.TimeoutExpired as e:
        return {
            "ok": False,
            "error": f"TimeoutExpired after {timeout_s}s",
            "stdout": _truncate(e.stdout or "" if isinstance(e.stdout, str) else ""),
            "stderr": _truncate(e.stderr or "" if isinstance(e.stderr, str) else ""),
            "exit_code": None,
            "duration_s": time.monotonic() - start,
            "cmd": cmd,
            "cwd": cwd,
        }


# ---------------------------------------------------------------------------
# Diagnostic / smoke
# ---------------------------------------------------------------------------

def ping(message: str = "hello") -> dict:
    """No-op smoke test. Returns host info and a round-tripped message.

    Use this first after registering functions to confirm the endpoint
    can be reached and the agentic package is importable.
    """
    import getpass
    import platform
    import socket
    import sys

    return {
        "ok": True,
        "error": None,
        "stdout": "",
        "stderr": "",
        "duration_s": 0.0,
        "message": message,
        "hostname": socket.gethostname(),
        "fqdn": socket.getfqdn(),
        "user": getpass.getuser(),
        "python": sys.version,
        "platform": platform.platform(),
        "cwd": os.getcwd(),
        "env_modules": os.environ.get("LOADEDMODULES", ""),
    }


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build_nekrs(
    repo_root: str,
    nekrs_home: str,
    system: str = "aurora",
    extra_env: dict | None = None,
) -> dict:
    """Build nekRS by invoking the appropriate BuildMeOn<System> script.

    The script runs cmake + make in the repo's build/ directory and installs
    to nekrs_home. Build can take 5-15 minutes; the timeout below allows up
    to 60 minutes.
    """
    repo = Path(repo_root).expanduser().resolve()
    script_map = {
        "aurora": "BuildMeOnAurora",
        "polaris": "BuildMeOnPolaris",
        "crux": "BuildMeOnCrux",
        "local": "BuildMeOnLocal",
    }
    if system.lower() not in script_map:
        return {
            "ok": False,
            "error": f"unsupported system {system!r}",
            "stdout": "",
            "stderr": "",
            "duration_s": 0.0,
        }
    script = repo / script_map[system.lower()]
    if not script.exists():
        return {
            "ok": False,
            "error": f"build script not found: {script}",
            "stdout": "",
            "stderr": "",
            "duration_s": 0.0,
        }

    # BuildMeOnAurora prompts for confirmation when no NEKRS_HOME arg is given.
    # We always pass nekrs_home explicitly to suppress the interactive prompt.
    return _run(
        ["bash", str(script), str(Path(nekrs_home).expanduser())],
        cwd=str(repo),
        env=extra_env,
        timeout_s=3600.0,
    )


# ---------------------------------------------------------------------------
# Case setup
# ---------------------------------------------------------------------------

def setup_case(
    repo_root: str,
    case_dir: str,
    system: str,
    nekrs_home: str,
    options: dict | None = None,
    extra_env: dict | None = None,
) -> dict:
    """Run scripts/ml/setup_case in case_dir to generate the submission script.

    Submission is suppressed via NEKRS_AGENTIC_NO_SUBMIT=1 (see the patched
    setup_case / nrsqsub_<system> scripts). The agent then calls submit_job
    explicitly on the returned script path. This gives the agent a chance to
    show the user the generated script before queuing it.

    options is a dict of flag -> value mirroring the setup_case CLI:
        nodes, time, proj_id, model, deployment, ml_task, client,
        db_nodes, sim_nodes, train_nodes, ensemble, venv_path.

    Returns the standard shape plus:
        generated_scripts: list[str] of newly-created .sh files in case_dir
    """
    repo = Path(repo_root).expanduser().resolve()
    case = Path(case_dir).expanduser().resolve()
    setup_script = repo / "scripts" / "ml" / "setup_case"
    if not setup_script.exists():
        return {
            "ok": False,
            "error": f"setup_case not found at {setup_script}",
            "stdout": "",
            "stderr": "",
            "duration_s": 0.0,
        }

    args = [
        "bash",
        str(setup_script),
        system,
        str(Path(nekrs_home).expanduser()),
        "--no-submit",
    ]
    flag_map = {
        "venv_path": "--venv_path",
        "nodes": "--nodes",
        "time": "--time",
        "proj_id": "--proj_id",
        "model": "--model",
        "deployment": "--deployment",
        "ml_task": "--ml_task",
        "client": "--client",
        "db_nodes": "--db_nodes",
        "sim_nodes": "--sim_nodes",
        "train_nodes": "--train_nodes",
        "ensemble": "--ensemble",
    }
    for k, v in (options or {}).items():
        flag = flag_map.get(k)
        if flag is None:
            continue
        args.extend([flag, str(v)])

    # Capture which .sh files exist before so we can diff after
    before = {p.name: p.stat().st_mtime for p in case.glob("*.sh")}

    env = {"NEKRS_AGENTIC_NO_SUBMIT": "1"}
    if extra_env:
        env.update(extra_env)
    result = _run(args, cwd=str(case), env=env, timeout_s=1800.0)

    after = {p.name: p.stat().st_mtime for p in case.glob("*.sh")}
    new_or_modified = [
        str(case / n)
        for n, mt in after.items()
        if n not in before or before[n] < mt
    ]
    result["generated_scripts"] = sorted(new_or_modified)
    return result


# ---------------------------------------------------------------------------
# Scheduler operations
# ---------------------------------------------------------------------------

def submit_job(
    system: str,
    script_path: str,
    queue: str | None = None,
    cwd: str | None = None,
) -> dict:
    """Submit a generated PBS script via qsub. Returns the job_id on success.

    Does NOT poll for completion. Use get_job_status (with care) for that.
    """
    from agentic.schedulers import get_scheduler

    sched = get_scheduler(system)
    script = Path(script_path).expanduser().resolve()
    if not script.exists():
        return {
            "ok": False,
            "error": f"submission script not found: {script}",
            "stdout": "",
            "stderr": "",
            "duration_s": 0.0,
        }
    cmd = list(sched.submit_cmd)
    if queue:
        cmd.extend(["-q", queue])
    cmd.append(str(script))
    work_dir = cwd or str(script.parent)
    result = _run(cmd, cwd=work_dir, timeout_s=120.0)
    if result["ok"]:
        try:
            result["job_id"] = sched.parse_submit_output(result["stdout"])
        except Exception as e:
            result["job_id"] = None
            result["error"] = f"submit ok but could not parse job id: {e}"
    return result


def get_job_status(
    system: str,
    job_id: str,
    min_interval_s: float = 15.0,
) -> dict:
    """Query the scheduler for a single job's status.

    Per-job-id throttle: if called more often than min_interval_s for the
    same job, the call sleeps until the interval has passed before issuing
    qstat. This caps the load on the login-node scheduler regardless of how
    aggressively the agent polls.

    The skill documents a 30-60s poll cadence as the norm; this throttle
    exists as a safety net, not as the recommended cadence.
    """
    from agentic.schedulers import get_scheduler
    from agentic.safety import throttle_query

    waited = throttle_query(f"status:{system}:{job_id}", min_interval_s=min_interval_s)
    sched = get_scheduler(system)
    cmd = list(sched.status_cmd_template) + [job_id]
    result = _run(cmd, timeout_s=60.0)
    result["throttle_wait_s"] = waited
    if result["ok"]:
        try:
            result["status"] = sched.parse_status_output(result["stdout"])
        except Exception as e:
            result["status"] = None
            result["error"] = f"status ok but parse failed: {e}"
    elif "Unknown Job" in (result.get("stderr") or ""):
        # PBS returns nonzero when the job has aged out of qstat -- treat as completed-unknown
        result["status"] = {"state": "unknown_or_completed", "fields": {}}
        result["ok"] = True
        result["error"] = None
    return result


def cancel_job(system: str, job_id: str) -> dict:
    from agentic.schedulers import get_scheduler

    sched = get_scheduler(system)
    cmd = list(sched.cancel_cmd) + [job_id]
    return _run(cmd, timeout_s=60.0)


# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------

def tail_log(path: str, n_lines: int = 200) -> dict:
    """Return the last n_lines of a file. Safe for live-growing PBS .o/.e files."""
    p = Path(path).expanduser().resolve()
    if not p.exists():
        return {
            "ok": False,
            "error": f"file not found: {p}",
            "stdout": "",
            "stderr": "",
            "duration_s": 0.0,
        }
    start = time.monotonic()
    try:
        with p.open("rb") as f:
            # Read tail efficiently for large files
            f.seek(0, os.SEEK_END)
            size = f.tell()
            block = 8192
            data = b""
            while size > 0 and data.count(b"\n") <= n_lines:
                read = min(block, size)
                size -= read
                f.seek(size)
                data = f.read(read) + data
        text = data.decode("utf-8", errors="replace")
        lines = text.splitlines()[-n_lines:]
        body = "\n".join(lines)
        return {
            "ok": True,
            "error": None,
            "stdout": _truncate(body),
            "stderr": "",
            "duration_s": time.monotonic() - start,
            "n_lines_returned": len(lines),
            "file_size_bytes": p.stat().st_size,
        }
    except Exception as e:
        return {
            "ok": False,
            "error": f"{type(e).__name__}: {e}",
            "stdout": "",
            "stderr": "",
            "duration_s": time.monotonic() - start,
        }


def list_results(
    case_dir: str,
    patterns: Iterable[str] = ("*.log", "*.o*", "*.e*", "logfile*", "*.fld*", "out*"),
) -> dict:
    """List recent output artefacts in a case directory matching common patterns.

    Returns file names with sizes and mtimes. The agent uses this to find
    the right .o<jobid> / .e<jobid> file to tail without guessing.
    """
    p = Path(case_dir).expanduser().resolve()
    if not p.exists():
        return {
            "ok": False,
            "error": f"directory not found: {p}",
            "stdout": "",
            "stderr": "",
            "duration_s": 0.0,
        }
    start = time.monotonic()
    hits: list[dict] = []
    seen: set[str] = set()
    for pattern in patterns:
        for f in p.glob(pattern):
            if not f.is_file() or str(f) in seen:
                continue
            seen.add(str(f))
            st = f.stat()
            hits.append({"path": str(f), "size": st.st_size, "mtime": st.st_mtime})
    hits.sort(key=lambda d: d["mtime"], reverse=True)
    return {
        "ok": True,
        "error": None,
        "stdout": "",
        "stderr": "",
        "duration_s": time.monotonic() - start,
        "files": hits,
    }


# ---------------------------------------------------------------------------
# Registry helper
# ---------------------------------------------------------------------------

# Names a client.register.py walks over to register all functions in one pass.
REGISTERED_FUNCTIONS = (
    "ping",
    "build_nekrs",
    "setup_case",
    "submit_job",
    "get_job_status",
    "cancel_job",
    "tail_log",
    "list_results",
)
