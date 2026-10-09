---
name: hpc-compute-node
description: Determine whether this session can actually run Python, torch, MPI or nekRS on an HPC system, find a running job to ssh into, and load the right environment once there. Use before running or verifying anything that needs torch, adios2, MPI or a GPU, and whenever something fails with a missing shared library, a GLIBC version error, or an MPI_Init abort.
---

# Running things on an HPC system

Sessions usually start on a login node, where most of this project cannot run.
Establish where you are before planning any verification, so you do not promise
a check you cannot perform.

The procedure is the same everywhere; only the hostname patterns, the scheduler
and the module names change. Per-system specifics are in the tables below --
**if you work out the details for a system that is not listed, add it.**

## 1. Where am I?

```bash
hostname
```

| System | Login node | Compute node |
|---|---|---|
| Aurora | `aurora-uan-*` | `x####c#s#b#n#` |
| Polaris | `polaris-login-*` | `x####c#s#b#n#` |

A pattern that matches nothing here means an unlisted system: find out, act on
what you learn, and add a row.

## 2. What you can and cannot do on a login node

Reading, searching and analyzing files is fine, and so is light Python. Heavy
compute, GPUs and MPI are not. 

**If you are stuck on a login node**, say so explicitly, state that the change
is unverified, and give the user the exact command to run. Never present a
static review as if it were a test.

## 3. Find a compute node

Do not submit jobs on your own initiative. Check whether one is already
running -- that costs nothing -- and then **ask before using it**, since the
allocation may be reserved for other work.

PBS systems (Aurora, Polaris):

```bash
qstat -u $USER -n
```

`R` in the `S` column is running, `Q` is queued. For a running job the node
list is on the following line, e.g. `x4104c2s0b0n0/0*208`; take the hostname
before the `/`:

```bash
ssh x4104c2s0b0n0
```

If everything is `Q`, there is nothing to attach to -- say so rather than
waiting on it.

On a Slurm system use `squeue -u $USER` and `srun --jobid=<id> --pty bash`
instead; add a row here once you have confirmed the details.

## 4. Load the environment

Without this there is often no `python` on the node at all.

| System | Setup |
|---|---|
| Aurora | `module load frameworks` (python3.12, torch XPU, mpi4py) |
| Polaris | `module use /soft/modulefiles/ && module load conda && conda activate` |

`tests/sites.py` holds the authoritative per-system `prepare_cmds` the test
suite uses; check there when a module name looks stale.

Examples that use a project venv activate it *after* the module:

```bash
source <example>/../_env_dist-gnn_posix/bin/activate     # or _env_dist-gnn_adios
```

Environment variables a hand-run example typically needs on Aurora:

```bash
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export FI_CXI_RX_MATCH_MODE=hybrid
export UR_L0_USE_COPY_ENGINE=0
```

The example's own `nrsrun_<system>` script is the source of truth for these;
read it rather than guessing.

## 5. Gotchas that cost a cycle each

These are not Aurora-specific; expect them on any batch system.

- **`/tmp` is node-local**, not shared with the login node. Write scratch to the
  project filesystem, and clean it up when you are done.
- **`ssh` buffers output.** A long `ssh node "..."` can return nothing for
  minutes. Have the job write results to a file under the project tree and poll
  the file instead of waiting on the pipe.
- **Always bound a run** with `timeout -s KILL <seconds>`, and `pkill` the ranks
  afterwards. A hung MPI job holds the allocation until it expires.
- **Allocations end mid-session.** Re-check `hostname` or re-run the queue
  command before assuming the node you used earlier is still there.

## Extending this skill

Keep the five steps above system-independent and put specifics in the tables.
When you learn something new -- another machine, a changed module, a scheduler
that is not PBS -- add it here with the command you actually ran, not one you
believe should work.
