---
name: run-on-aurora
description: Drive nekRS-ML builds, case setup, job submission, and monitoring on ALCF Aurora from a laptop agent via a Globus Compute login-node endpoint. Use when the user asks to "build on Aurora", "run this case on Aurora", "submit my GNN training to Aurora", "check my Aurora job", "tail the Aurora log", or similar.
---

# Driving nekRS-ML on Aurora

You are calling a Globus Compute endpoint that runs as the user on an **Aurora login node**. The endpoint exists so you can do filesystem operations, run the existing build/setup shell scripts, and talk to the PBS scheduler — without the user having to SSH in.

## Hard rules (do not violate)

1. **No compute on the login node.** Everything that uses MPI / multi-rank GPU work must go through PBS via `submit_job`. The endpoint refuses `mpiexec`, `mpirun`, `srun`, `aprun`, etc., by policy. Don't try to bypass that guard.
2. **Any script returned by `setup_case` is a compute script — submit it, do NOT bash-execute it.** Regardless of filename (`run.sh`, `submit_nekrs.sh`, ...) or whether it has `#PBS` directives, every script produced by `setup_case` is meant to run *inside* a PBS allocation. It contains `mpiexec`. Use `submit_job(script_path=...)` even if the file looks like a plain bash script — never `bash run.sh`, `./run.sh`, or `ssh aurora 'bash run.sh'`. The endpoint guard will refuse a direct bash-execution of any script containing `mpiexec` (it inspects script contents), but you should not depend on it catching you — the right call is always `submit_job`.
3. **Don't hammer the scheduler.** Poll `get_job_status` no faster than **once per 60 seconds** per job. The function has a 15s throttle as a safety net, but the throttle exists to prevent accidents — your default cadence should be slower. If a job is expected to run for hours, poll every few minutes.
4. **Always show the user the generated submission script before submitting.** The `setup_case` flow generates the script but does not submit. After `setup_case`, read the generated script with `tail_log` (or report its path and have the user inspect), then call `submit_job`. Don't chain them silently.
5. **No destructive ops without confirmation.** `cancel_job` cancels a running job; deleting case output, build directories, etc. is out of scope for v1.

## Prerequisites you should check first

Before you start calling functions, verify the user has these set up. If anything is missing, walk them through it — don't try to work around missing setup.

- `~/.config/nekrs-ml-agentic/endpoints.json` and `functions.json` exist. Both are created by `python -m agentic.client.setup --uuid <UUID> --repo-root <PATH-ON-AURORA>` (the laptop-side one-shot). If they're missing, point the user at that command — don't try to construct the JSON by hand.
- The Aurora-side endpoint is running. If `System("aurora").ping(...)` fails to return within ~60s, the endpoint is probably down and needs `globus-compute-endpoint start nekrs-ml-aurora` on a login node.

The one-time bootstrap on Aurora is [agentic/globus_endpoints/setup_aurora.sh](../../../agentic/globus_endpoints/setup_aurora.sh); the laptop-side one-shot is `python -m agentic.client.setup`.

## How to call functions

Always use the `System` class — never construct the Globus Compute Executor yourself. Every name in `agentic.functions.REGISTERED_FUNCTIONS` is callable as a method; `System` auto-injects `system`, `repo_root`, and `nekrs_home` from the endpoint config when the underlying function accepts them, so you usually don't pass them explicitly.

```python
from agentic.client import System
hpc = System("aurora")

# Sanity check
result = hpc.ping(message="from-claude")
assert result["ok"], result["error"]
print(result["hostname"], result["user"])
```

All calls return a dict with `ok`, `error`, `stdout`, `stderr`, `duration_s`, plus per-function fields. If `ok` is False, surface `error` and the last lines of `stderr` to the user — don't paper over failures.

## The standard workflows

### Build nekRS

```python
hpc = System("aurora")
out = hpc.build_nekrs(nekrs_home="/home/<user>/.local/nekrs")
```

Takes 5–15 minutes. The `extra_env` arg lets you pass `{"ENABLE_SMARTREDIS": "ON"}` for SmartRedis builds. On failure, the `stderr` field has the cmake/make tail.

**Long-running calls and client timeouts.** Each `System` call uses a per-function client-side timeout from `agentic.functions.DEFAULT_TIMEOUTS_S` (e.g., `build_nekrs` = 3900s, `setup_case` = 1900s, `ping` = 60s) — long enough that the build *will* complete normally. If you need to extend further for a specific call, pass `_timeout=<seconds>` as a kwarg:

```python
out = hpc.build_nekrs(nekrs_home="...", _timeout=7200)  # bump to 2h
```

If you ever hit a client-side timeout while the function is still running server-side, **don't re-submit** — the original task is almost certainly still progressing on the endpoint. Instead poll for the expected side-effect (e.g., the installed binary appearing under `nekrs_home/bin/`) with `list_results` or `tail_log` on the build log.

### Set up and submit a case

This is two steps on purpose: setup generates, submit queues.

```python
hpc = System("aurora")
case = "/lus/flare/projects/<proj>/<user>/nekRS-ML/examples/tgv_gnn_offline"

setup = hpc.setup_case(
    case_dir=case,
    nekrs_home="/home/<user>/.local/nekrs",
    options={
        "nodes": 2,
        "time": "01:00",
        "proj_id": "<your_project>",
        "model": "dist-gnn",
        "deployment": "offline",
    },
)
assert setup["ok"], setup["error"]
# Show the user what got generated. Usually one or two .sh files.
print(setup["generated_scripts"])
# Optional: show the script content so the user can sanity-check before queuing
preview = hpc.tail_log(setup["generated_scripts"][-1], n_lines=200)
print(preview["stdout"])
```

**Valid `options=` keys** (any others are silently ignored — don't invent ones):

| Key | Meaning | Example |
|---|---|---|
| `nodes` | PBS node count | `2` |
| `time` | walltime `HH:MM` | `"01:00"` |
| `proj_id` | PBS `-A` project | `"datascience"` |
| `model` | ML model | `"dist-gnn"` / `"sr-gnn"` |
| `deployment` | how nekRS + ML talk | `"offline"` / `"colocated"` / `"clustered"` |
| `ml_task` | for examples that need it | `"train"` / `"inference"` |
| `client` | data transport | `"posix"` / `"smartredis"` / `"adios"` |
| `db_nodes`, `sim_nodes`, `train_nodes` | per-component node splits (clustered/colocated only) | int |
| `ensemble` | ensemble launcher | `"el"` |
| `venv_path` | custom venv on the HPC | absolute path |

**Queue is NOT an `options=` key.** It's specified per-submission on `submit_job`. The other PBS attributes (account, walltime, nodes, filesystems) are bundled in the `pbs_hints` dict that `setup_case` returns — splat them into `submit_job`:

```python
sub = hpc.submit_job(
    script_path=setup["generated_scripts"][-1],
    queue="debug-scaling",      # or "prod", "debug", etc. — depends on your allocation
    **setup["pbs_hints"],       # account, walltime, nodes, filesystems
)
assert sub["ok"], sub["error"]
job_id = sub["job_id"]
```

**Why both `pbs_hints` and the queue?** Some examples (those using the system-wide `scripts/nrsqsub_<system>`) embed `#PBS` directives in the generated script and don't need `pbs_hints` at all — qsub reads the values from the script. Other examples (the per-example `nrsrun_<system>` like `tgv_gnn_offline/run.sh`) produce a plain bash script with no `#PBS` directives, and qsub has to learn the account / walltime / nodes from kwargs. Splatting `**setup["pbs_hints"]` always works: redundant for the first kind, essential for the second.

`queue` is intentionally left off `pbs_hints` because the right queue depends on the user's intent for this particular submission (debug vs prod vs scaling), not on the case configuration. Ask the user if you don't know.

### Monitor a job (correctly)

```python
import time

job_id = "12345.aurora-pbs-0001..."
while True:
    s = hpc.get_job_status(job_id=job_id)
    state = (s.get("status") or {}).get("state")
    print(state, s.get("throttle_wait_s"))
    if state in ("F", "unknown_or_completed", None):
        break
    time.sleep(60)  # 60s minimum; longer for long jobs
```

PBS state codes: `Q` queued, `R` running, `E` exiting, `F` finished, `H` held. After `F`, fetch the `.o<jobid>` / `.e<jobid>` file via `list_results` + `tail_log` to summarize the run for the user.

### Fetch results

```python
listing = hpc.list_results(case_dir=case)
for f in listing["files"][:10]:
    print(f["path"], f["size"])

# Tail the PBS stdout
out_file = next(f["path"] for f in listing["files"] if ".o" in f["path"])
log = hpc.tail_log(path=out_file, n_lines=500)
print(log["stdout"])
```

### Check that a path exists on the HPC (no dedicated tool — use `list_results`)

There's no `stat` / `exists` function. Use `list_results` as the probe: it returns `ok: False` with `error: "directory not found: ..."` when the path is missing. Costs one Globus Compute roundtrip — don't shell out to anything else.

```python
probe = hpc.list_results(case_dir="/lus/flare/projects/<proj>/<user>/nekRS-ML/examples/tgv_gnn_offline")
if not probe["ok"] and "not found" in (probe.get("error") or ""):
    # the path doesn't exist -- ask the user for the right one rather than guessing
    ...
```

### Cancel a job

```python
hpc.cancel_job(job_id=job_id)
```

Confirm with the user before calling this unless they explicitly asked to cancel.

## When things go wrong

- **`System(...)` raises "No endpoint configured"**: the user hasn't created `endpoints.json`. Point them at `python -m agentic.client.setup --uuid <UUID> --repo-root <PATH>` rather than writing the file by hand.
- **`ping` hangs or times out**: the endpoint process on Aurora is probably down. Tell the user to SSH in once and `globus-compute-endpoint start nekrs-ml-aurora`. Suggest they wrap it in tmux so it survives logout.
- **`build_nekrs` fails with cmake errors**: report the last 50 lines of `stderr` to the user; this is usually a missing module or wrong oneAPI version, not something the agent can fix automatically.
- **`setup_case` runs but `generated_scripts` is empty**: the per-example `nrsrun_aurora` may use a non-standard naming scheme. Fall back to listing files in `case_dir` with a wider pattern and ask the user which one to submit.
- **`get_job_status` returns `state=unknown_or_completed`**: the job has aged out of PBS's active table. It either finished or was cancelled long ago. Use `list_results` + `tail_log` on the `.o<jobid>` file to confirm.
- **`AttributeError: 'System' has no remote function 'foo'`**: `foo` isn't in `agentic.functions.REGISTERED_FUNCTIONS`. Either you're calling the wrong name or it hasn't been added yet — do NOT try to invent a workaround.

## What this skill does NOT cover

- Running the actual simulation/training on login nodes (forbidden — always goes through `submit_job`).
- Polaris, Crux, or Frontier (Aurora-only for v1; per-system skills will mirror this one).
- Generating new `examples/` workflows from prompts (deferred to a later phase).
- Inference endpoint calls (deferred — when added, will be its own function).

If the user asks for any of these, explain the limitation rather than improvising.
