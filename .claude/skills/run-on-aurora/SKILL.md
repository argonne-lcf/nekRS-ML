---
name: run-on-aurora
description: Drive nekRS-ML builds, case setup, job submission, and monitoring on ALCF Aurora from a laptop agent via a Globus Compute login-node endpoint. Use when the user asks to "build on Aurora", "run this case on Aurora", "submit my GNN training to Aurora", "check my Aurora job", "tail the Aurora log", or similar.
---

# Driving nekRS-ML on Aurora

You are calling a Globus Compute endpoint that runs as the user on an **Aurora login node**. The endpoint exists so you can do filesystem operations, run the existing build/setup shell scripts, and talk to the PBS scheduler — without the user having to SSH in.

## Hard rules (do not violate)

1. **No compute on the login node.** Everything that uses MPI / multi-rank GPU work must go through PBS via `submit_job`. The endpoint refuses `mpiexec`, `mpirun`, `srun`, `aprun`, etc., by policy. Don't try to bypass that guard.
2. **Don't hammer the scheduler.** Poll `get_job_status` no faster than **once per 60 seconds** per job. The function has a 15s throttle as a safety net, but the throttle exists to prevent accidents — your default cadence should be slower. If a job is expected to run for hours, poll every few minutes.
3. **Always show the user the generated submission script before submitting.** The `setup_case` flow generates the script but does not submit. After `setup_case`, read the generated script with `tail_log` (or report its path and have the user inspect), then call `submit_job`. Don't chain them silently.
4. **No destructive ops without confirmation.** `cancel_job` cancels a running job; deleting case output, build directories, etc. is out of scope for v1.

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

Once the user is happy:

```python
sub = hpc.submit_job(script_path=setup["generated_scripts"][-1])
assert sub["ok"], sub["error"]
job_id = sub["job_id"]
```

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
