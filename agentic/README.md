# nekRS-ML agentic layer

Drive nekRS-ML builds, case setup, job submission, and monitoring on ALCF
HPC systems from a laptop agent (Claude Code) — without SSH'ing in.

The agent talks to a long-running Globus Compute endpoint that you start
once on an HPC login node. The endpoint runs the existing build/setup
shell scripts in this repo and shells out to PBS. Compute work itself
never runs on the login node: it always goes through `qsub`.

v1 ships support for **Aurora** only. Polaris and Frontier are planned;
the scheduler and skill layers are structured so adding them is small.

```
┌────────────┐                    ┌──────────────────────────┐
│  Laptop    │   Globus Compute   │  Aurora login node       │
│ Claude     │ ───────────────────►   nekrs-ml-aurora        │
│ Code       │   (function UUID)  │   endpoint (LocalProvider│
│            │ ◄───────────────── │   + agentic functions)   │
└────────────┘    JSON result     └────────────┬─────────────┘
                                               │ qsub / qstat / qdel
                                               ▼
                                       ┌───────────────┐
                                       │  PBS jobs on  │
                                       │ compute nodes │
                                       └───────────────┘
```

## What you can do from the agent

Any function listed in `agentic.functions.REGISTERED_FUNCTIONS`:

| Function          | What it does                                                           |
|-------------------|------------------------------------------------------------------------|
| `ping`            | Round-trip smoke check; returns hostname / user / Python version       |
| `build_nekrs`     | Runs `BuildMeOnAurora` against your chosen `NEKRS_HOME`                |
| `setup_case`      | Runs `scripts/ml/setup_case` in a case dir; generates submission script (no qsub) |
| `submit_job`      | `qsub` a generated submission script; returns the PBS job ID           |
| `get_job_status`  | `qstat -f -F json` on one job, with per-job rate limiting              |
| `cancel_job`      | `qdel` a job (the agent will confirm with you first)                   |
| `tail_log`        | Read the last N lines of any file (PBS `.o<jobid>`, run logs, etc.)    |
| `list_results`    | Glob common output patterns in a case directory                         |

The login-node endpoint **refuses to run** `mpiexec`, `mpirun`, `srun`,
`aprun`, `jsrun`, or `ibrun`. Compute always goes through `submit_job`.

---

## First-time setup walkthrough

Total time end-to-end: ~30 minutes, most of it waiting for the Aurora
build to finish.

### Prerequisites

- An ALCF account with Aurora access and an active project allocation.
- A clone of nekRS-ML in your project directory on Aurora (e.g.
  `/lus/flare/projects/<proj>/<user>/nekRS-ML`). This is what the endpoint
  will operate on.
- A clone of nekRS-ML on your laptop (this repo). It's where Claude Code
  reads the skill from.
- Python 3.10+ on your laptop.
- Globus credentials linked to your ALCF identity (Globus auth is what
  Globus Compute uses; no SSH keys involved for the agent path).

### Part 1 — bootstrap the endpoint on Aurora (one time)

SSH to an Aurora login node. From the repo on Aurora:

```bash
cd /lus/flare/projects/<proj>/<user>/nekRS-ML
bash agentic/globus_endpoints/setup_aurora.sh
```

This script:
1. Loads the `frameworks` module
2. Creates a venv at `~/.local/nekrs-ml-agentic`
3. Installs `globus-compute-endpoint` and the local `agentic` package
4. Initialises an endpoint named `nekrs-ml-aurora` and writes the
   `LocalProvider` config from
   [globus_endpoints/aurora_config.yaml](globus_endpoints/aurora_config.yaml)
5. Prints next-step instructions

Then start the endpoint **inside a long-running session** (so it survives
logout):

```bash
tmux new -s gc-endpoint
source ~/.local/nekrs-ml-agentic/bin/activate
module load frameworks
globus-compute-endpoint start nekrs-ml-aurora
# Follow the printed URL, authenticate as your ALCF identity
# detach: Ctrl-b d
```

Grab the endpoint UUID:

```bash
globus-compute-endpoint list
# look for: nekrs-ml-aurora   Running   <UUID>
```

Copy that UUID. You'll need it in Part 2.

### Part 2 — set up your laptop (one time)

```bash
cd </path/to>/nekRS-ML        # your laptop clone
python3 -m venv .agentic-venv
source .agentic-venv/bin/activate
pip install -e ./agentic

python -m agentic.client.setup \
    --uuid <PASTE_UUID_FROM_PART_1> \
    --repo-root /lus/flare/projects/<proj>/<user>/nekRS-ML
```

That one command:
1. Writes the endpoint entry to `~/.config/nekrs-ml-agentic/endpoints.json`
2. Registers all functions in `agentic.functions` with Globus Compute and
   saves their UUIDs to `~/.config/nekrs-ml-agentic/functions.json`
3. Calls `ping()` on the Aurora endpoint as a round-trip smoke test

Expected last few lines on success:

```
[setup] Pinging endpoint for 'aurora' ...
  hostname : aurora-uan-xxxx.hostmgmt.cm.aurora.alcf.anl.gov
  user     : <your username>
  python   : 3.12.x (main, ...)
```

If `ping` fails, the endpoint on Aurora is probably not running — re-attach
your tmux session and check.

Optional: pass `--nekrs-home /home/<user>/.local/nekrs` if you want
`System` to auto-fill that into `build_nekrs` / `setup_case` calls
without you specifying it every time.

### Part 3 — test the plumbing directly from Python (no agent yet)

Before involving Claude, confirm the API works from a plain Python shell.
This separates "endpoint is healthy" from "agent picked the right tool."

```python
from agentic.client import System

hpc = System("aurora")
print(hpc.ping(message="hello-from-laptop"))

# List output files in any existing case directory on Aurora
out = hpc.list_results(case_dir="/lus/flare/projects/<proj>/<user>/nekRS-ML/examples/tgv_gnn_offline")
for f in out["files"][:5]:
    print(f["mtime"], f["size"], f["path"])
```

If both calls return dicts with `"ok": True`, the agentic layer is wired
up end-to-end.

### Part 4 — drive it from Claude Code

From the laptop clone of nekRS-ML, open the repo with Claude Code:

```bash
cd </path/to>/nekRS-ML
claude
```

Claude will load the `run-on-aurora` skill automatically when your prompt
matches its triggers. Try, in order of growing scope:

1. **"Ping the Aurora endpoint and report the login node it landed on."**
   Quickest validation that the skill activates and the agent uses
   `System("aurora").ping(...)` correctly.

2. **"List the most recently modified output files in
   `<absolute-path-to-case-on-aurora>` and show me the top five."**
   Validates `list_results`.

3. **"Build nekRS on Aurora into `/home/<user>/.local/nekrs`. Tell me
   when it finishes and surface any errors."**
   The first real test. Takes 5–15 minutes. The agent should call
   `build_nekrs` and wait for it; on failure, surface the tail of stderr
   from cmake/make.

4. **"Set up the `tgv_gnn_offline` example on Aurora for a 2-node,
   1-hour run using my project `<your_project>` and the dist-gnn model
   in offline deployment. Show me the generated submit script before
   queuing it. Then submit and monitor every couple of minutes until
   the job finishes; tail the last 200 lines of the output when done."**
   End-to-end happy-path workflow: `setup_case` → preview → `submit_job`
   → `get_job_status` loop → `tail_log`.

If any step misbehaves (the agent tries to invent commands, polls too
fast, runs compute on the login node), the skill itself probably needs
tightening — open an issue or update
[../.claude/skills/run-on-aurora/SKILL.md](../.claude/skills/run-on-aurora/SKILL.md).

---

## Day-to-day after first-time setup

You shouldn't need to re-run any setup unless something changes:

| If you ...                                              | Run                                                                |
|---------------------------------------------------------|--------------------------------------------------------------------|
| Restart your laptop                                     | nothing — config is on disk                                        |
| The Aurora login node reboots / endpoint dies           | re-attach the tmux session and `globus-compute-endpoint start nekrs-ml-aurora`         |
| Edit the body of a function in [functions.py](functions.py) | `python -m agentic.client.register --force`                       |
| Add a new function to [functions.py](functions.py)      | append to `REGISTERED_FUNCTIONS`, then `python -m agentic.client.register --only <name>` |
| Change the endpoint UUID (rare)                         | `python -m agentic.client.setup --uuid <NEW> --repo-root <PATH>`   |
| Add a new HPC system                                    | add an entry to `agentic.schedulers._SCHEDULERS`, write a per-system skill, re-run `setup` with `--system <name>` |

## Layout of this package

| Path                                | Purpose                                                                 |
|-------------------------------------|-------------------------------------------------------------------------|
| [functions.py](functions.py)        | The login-node-safe functions registered with Globus Compute            |
| [safety.py](safety.py)              | Login-node policy guard, qstat per-job throttle, path validation        |
| [schedulers.py](schedulers.py)      | PBS scheduler wrapper for Aurora (Slurm adapter pending for Frontier)   |
| [globus_endpoints/](globus_endpoints/) | Endpoint config YAML and per-system bootstrap shell scripts          |
| [client/system.py](client/system.py)| `System("aurora")` — laptop-side handle that dispatches dynamically to remote functions |
| [client/setup.py](client/setup.py)  | The one-shot `python -m agentic.client.setup` orchestrator              |
| [client/register.py](client/register.py) | Registers functions with Globus Compute; persists UUIDs            |
| [tests/test_local.py](tests/test_local.py) | Local unit tests for safety + scheduler parsing (no HPC needed)   |
| [tests/smoke_remote.py](tests/smoke_remote.py) | Manual end-to-end `ping` against a real endpoint              |

## Limitations in v1

- **Aurora only.** Polaris is next; Frontier needs a Slurm scheduler adapter first.
- **No automatic endpoint respawn.** If the login node reboots, you re-attach tmux and restart the endpoint manually. A systemd-user unit is the obvious next step.
- **No workflow generation.** The agent can run *existing* examples; building new ones from prompts is a later phase.
- **No ALCF inference endpoint integration.** Planned as a separate function (`summarize_with_alcf_llm` or similar).

## Troubleshooting

| Symptom                                                              | Likely cause                                                         | Fix                                                                                            |
|----------------------------------------------------------------------|----------------------------------------------------------------------|------------------------------------------------------------------------------------------------|
| `python -m agentic.client.setup` hangs on registration               | First-time Globus auth needs your browser                            | Watch the terminal for an auth URL and complete it                                              |
| `ping` returns immediately with `ok=False, error="EndpointDownError"` | Endpoint process on Aurora is not running                            | Re-attach tmux on Aurora and start it                                                          |
| `ping` times out after 5 minutes                                     | Endpoint process is wedged                                            | `globus-compute-endpoint stop nekrs-ml-aurora && globus-compute-endpoint start nekrs-ml-aurora`|
| `build_nekrs` fails with cmake errors                                | Usually a missing module on the login node                            | Read the stderr tail; fix the environment, re-run                                              |
| `LoginNodePolicyError: Refusing to run 'mpiexec'`                    | Something in the call path tried to launch MPI on the login node      | This is by design — route through `submit_job` instead                                         |
| `AttributeError: 'System' has no remote function 'foo'`              | Calling something not in `REGISTERED_FUNCTIONS`                       | Check the list in [functions.py](functions.py); add + re-register if needed                    |
