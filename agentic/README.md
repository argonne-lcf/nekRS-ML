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
- A clone of nekRS-ML on your laptop. It's where Claude Code
  reads the skills from.
- Python 3.10+ on your laptop.
- Globus credentials linked to your ALCF identity (Globus auth is what
  Globus Compute uses; no SSH keys involved for the agent path).

### Part 1 — bootstrap the endpoint on Aurora (one time)

SSH to an Aurora login node. From the repo on Aurora:

```bash
cd /lus/flare/projects/<proj>/<user>/nekRS-ML
./agentic/globus_endpoints/setup_aurora.sh
```

The script:
1. Loads the `frameworks` module
2. Creates a venv inside the repo at `<repo>/_env-agentic`. Override with
   `VENV_PATH=...` if you want it elsewhere.
3. Installs `globus-compute-endpoint` and the local `agentic` package
4. Initialises an endpoint named `nekrs-ml-aurora` via
   `globus-compute-endpoint configure --template-config ...`, pointing at
   the `LocalProvider` engine template
   [globus_endpoints/aurora_user_config.yaml.j2](globus_endpoints/aurora_user_config.yaml.j2)
   (matches the ALCF reference flow). On re-runs it just refreshes that
   one file so any edits to the source propagate.
5. Prints the next-step instructions, including **the Python version
   under `frameworks`** — write it down, you'll need it in Part 2.

Now start the endpoint. The very first start needs to be **foreground or
inside tmux** so you can see and complete the Globus auth URL (browser
flow). After that the token is cached and you can use `--detach`.

**Option A — foreground (one-off testing or first-time auth)**. Runs in
your SSH session and dies when you Ctrl-C.

```bash
module load frameworks
source <repo>/_env-agentic/bin/activate
globus-compute-endpoint start nekrs-ml-aurora
# follow the printed URL, authenticate as your ALCF identity
# Ctrl-C when you're done
```

**Option B — tmux (good for first-time auth + live log access)**. Same
as Option A but inside a tmux session so you can detach the terminal
and reattach later for log scrollback.

```bash
tmux new -s gc-endpoint
module load frameworks
source <repo>/_env-agentic/bin/activate
globus-compute-endpoint start nekrs-ml-aurora
# detach: Ctrl-b d    reattach later: tmux a -t gc-endpoint
```

**Option C — detached daemon (`--detach`, simplest persistent option)**.
The endpoint daemonises itself, no terminal session required. **Only use
this after you've completed first-time auth via Option A or B**, because
`--detach` returns immediately and won't show the auth URL.

```bash
module load frameworks
source <repo>/_env-agentic/bin/activate
globus-compute-endpoint start nekrs-ml-aurora --detach
# Endpoint logs end up in ~/.globus_compute/nekrs-ml-aurora/EndpointLogs/
# Stop with:   globus-compute-endpoint stop nekrs-ml-aurora
# Restart:     globus-compute-endpoint restart nekrs-ml-aurora --detach
```

After the endpoint is running (any option), grab the UUID:

```bash
globus-compute-endpoint list
# look for: nekrs-ml-aurora   Running   <UUID>
```

Copy the UUID. You'll need it (and the Python version) in Part 2.

### Part 2 — set up your laptop (one time)

**Important: match the Python version between laptop and endpoint.**
Use the same `MAJOR.MINOR.PATCH` your Part 1 setup script reported
(e.g., `3.12.12` — *not* just `3.12`). Two reasons:

1. **Correctness.** Globus Compute serialises functions and arguments with
   pickle. A `MAJOR.MINOR` mismatch can succeed at registration and then
   fail unpredictably at call time with `ModuleNotFoundError` or worse.
2. **Cleanliness.** While a `PATCH`-only mismatch (e.g., `3.12.13` vs
   `3.12.12`) is harmless for pickle, but the SDK prints a
   `UserWarning: Environment differences detected` on every single call.
   Matching exactly silences that warning and keeps the output legible.

**Install Python version**

Option 1 — conda / miniconda / micromamba (recommended — easiest to
pin an exact patch):

```bash
# Substitute the EXACT version your Part 1 setup printed (e.g., 3.12.12).
conda create -n nekrs-ml-agentic python=3.12.12 -y
conda activate nekrs-ml-agentic
# skip the venv step below and jump straight to `pip install -e ./agentic`.
```

Option 2 — pyenv:

```bash
pyenv install 3.12.12     # exact version
pyenv shell 3.12.12       # use it in this shell
# Then create the venv with that interpreter (see block below).
```

**Create the venv and run setup** — the venv step is needed for
Option 2; with Option 1 the conda env replaces it, so jump
straight to `pip install -e ./agentic`.

```bash
cd </path/to>/nekRS-ML            # your laptop clone

# (Options 2/3 only) create the venv with the matching interpreter.
# Use the exact version you installed above (e.g., `python3.12` from Homebrew
# or `$(pyenv which python)` for pyenv).
python3.12 -m venv _env-agentic   # name is gitignored
source _env-agentic/bin/activate

pip install -e ./agentic

python -m agentic.client.setup \
    --uuid       <PASTE_UUID_FROM_PART_1> \
    --repo-root  /lus/flare/projects/<proj>/<user>/nekRS-ML \
    --nekrs-home /home/<user>/.local/nekrs
```

**About `--nekrs-home`** (strongly recommended): the path on the HPC where
nekRS is or will be installed. Once stored on the endpoint entry, `System`
auto-injects it into every call that takes a `nekrs_home` argument
(`build_nekrs`, `setup_case`), so neither you nor the agent has to repeat
it. You can change it later with another `setup` invocation, or just
override it per call. Skip it if you genuinely have multiple nekRS
installs and want to pass the path explicitly each time.

The setup command:
1. Writes the endpoint entry to `~/.config/nekrs-ml-agentic/endpoints.json`
2. **Authenticates with Globus Compute.** First run opens a browser tab
   to log in via your ALCF identity; the token is cached under
   `~/.globus_compute/` for ~30 days. If the browser doesn't open
   automatically, the terminal prints a URL — visit it manually and paste
   the returned auth code back. Pass `--reauth` to clear the cached token
   and force a fresh login (use this when sessions expire or get into a
   bad state — same recovery as `rm -r ~/.globus_compute/storage.db*`).
3. Registers all functions in `agentic.functions` with Globus Compute and
   saves their UUIDs to `~/.config/nekrs-ml-agentic/functions.json`
4. Calls `ping()` on the Aurora endpoint as a round-trip smoke test
5. Compares your laptop's Python MAJOR.MINOR with the endpoint's and
   prints a loud warning if they differ — re-create the laptop venv with
   the matching interpreter and re-run if you see one

Expected last few lines on success:

```
[setup] Pinging endpoint for 'aurora' ...
  hostname : aurora-uan-xxxx.hostmgmt.cm.aurora.alcf.anl.gov
  user     : <your username>
  python   : 3.12.x (main, ...)
```

If `ping` fails, the endpoint on Aurora is probably not running — restart
it (Option A/B/C above).

### Part 3 — test the setup directly from Python (no agent yet)

Before involving Claude, confirm the API works without the LLM in the
loop. This separates "endpoint is healthy" from "agent picked the right
tool" when things misbehave later.

Use the bundled
[`agentic.tests.smoke_remote`](tests/smoke_remote.py) script:

```bash
# minimum: just confirm the laptop -> endpoint pipeline is alive
python -m agentic.tests.smoke_remote

# fuller check: also exercises list_results and tail_log on a real case
python -m agentic.tests.smoke_remote \
    --case-dir /lus/flare/projects/<proj>/<user>/nekRS-ML/examples/tgv_gnn_offline
```

A successful run prints the Aurora login node it landed on, the user
identity, the Python version, and (if `--case-dir` is set) the five most
recently modified output files plus a tail of the most recent one.
Exit codes: `0` success, `1` a remote call returned `ok=False`, `2`
laptop-side setup error (e.g., you forgot to run `agentic.client.setup`).

If you'd rather poke at the API in a REPL, the same checks in Python:

```python
from agentic.client import System
hpc = System("aurora")
print(hpc.ping(message="hello-from-laptop")["hostname"])
print(hpc.list_results(case_dir="/lus/flare/projects/<proj>/<user>/nekRS-ML/examples/tgv_gnn_offline"))
```

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

2. **"Build nekRS on Aurora into `/home/<user>/.local/nekrs`. Tell me
   when it finishes and surface any errors."**
   The first real test. Takes 5–15 minutes. The agent should call
   `build_nekrs` and wait for it; on failure, surface the tail of stderr
   from cmake/make.

4. **"Set up the `tgv_gnn_offline` example on Aurora for a 1-node,
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

## Use cases after first-time setup

You shouldn't need to re-run any setup unless something changes:

| If you ...                                              | Run                                                                |
|---------------------------------------------------------|--------------------------------------------------------------------|
| Restart your laptop                                     | nothing — config is on disk                                        |
| The Aurora login node reboots / endpoint dies           | SSH back in and run `globus-compute-endpoint start nekrs-ml-aurora --detach` (auth is already cached) |
| Edit the body of a function in [functions.py](functions.py) | `python -m agentic.client.register --force`                       |
| Add a new function to [functions.py](functions.py)      | append to `REGISTERED_FUNCTIONS`, then `python -m agentic.client.register --only <name>` |
| Change the endpoint UUID (rare)                         | `python -m agentic.client.setup --uuid <NEW> --repo-root <PATH>`   |
| Globus Compute token expires / says "auth required"     | `python -m agentic.client.setup --uuid <UUID> --repo-root <PATH> --reauth` (re-runs the browser auth flow) |
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
- **No automatic endpoint respawn.** If the login node reboots, you restart the endpoint manually. A systemd-user unit is the obvious next step.
- **No workflow generation.** The agent can run *existing* examples; building new ones from prompts is a later phase.
- **No ALCF inference endpoint integration.** Planned as a separate function (`summarize_with_alcf_llm` or similar).

## Troubleshooting

| Symptom                                                              | Likely cause                                                         | Fix                                                                                            |
|----------------------------------------------------------------------|----------------------------------------------------------------------|------------------------------------------------------------------------------------------------|
| `python -m agentic.client.setup` hangs on registration               | First-time Globus auth needs your browser                            | Watch the terminal for an auth URL and complete it                                              |
| `setup` ends with `WARNING: Python version mismatch`                 | Laptop venv uses a different MAJOR.MINOR than the Aurora endpoint     | Recreate the laptop venv with `python<MAJOR.MINOR>` matching the endpoint, then re-run setup    |
| `ping` returns immediately with `ok=False, error="EndpointDownError"` | Endpoint process on Aurora is not running                            | Restart it on Aurora (Part 1 Option A/B/C)                                                     |
| `ping` times out after 5 minutes                                     | Endpoint process is wedged                                            | `globus-compute-endpoint stop nekrs-ml-aurora && globus-compute-endpoint start nekrs-ml-aurora --detach` |
| `ComputeAPIError ... 409 RESOURCE_CONFLICT ... Endpoint ... already in use` (transient — one-off) | Globus Compute briefly held the endpoint after the previous call | `System._call` retries automatically with backoff (1s, 2s, 4s) and drops its cached Executor between attempts |
| `409 RESOURCE_CONFLICT` keeps firing across all 4 retries | The UUID in `endpoints.json` may not match the endpoint currently running on the HPC (common after a teardown/reconfigure cycle) | On the HPC, run `globus-compute-endpoint list` — note the *Running* UUID. If it differs from what's in `~/.config/nekrs-ml-agentic/endpoints.json`, re-run `python -m agentic.client.setup --uuid <correct UUID> ... --force-register`. If a stale UUID is also listed (Disconnected), purge it: `globus-compute-endpoint delete <stale_name> && rm -rf ~/.globus_compute/<stale_name>`. |
| `build_nekrs` fails with cmake errors                                | Usually a missing module on the login node                            | Read the stderr tail; fix the environment, re-run                                              |
| `LoginNodePolicyError: Refusing to run 'mpiexec'`                    | Something in the call path tried to launch MPI on the login node      | This is by design — route through `submit_job` instead                                         |
| `AttributeError: 'System' has no remote function 'foo'`              | Calling something not in `REGISTERED_FUNCTIONS`                       | Check the list in [functions.py](functions.py); add + re-register if needed                    |
