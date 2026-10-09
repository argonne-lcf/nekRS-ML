# Working in this repository

Instructions for coding agents. Scoped to what is **not** discoverable in a
command or two -- build and test instructions live in `README.md` and
`tests/README.md`.

## Where you are running matters

Sessions usually start on login nodes, where most of this project
cannot run. Check before assuming:

```bash
hostname   # *-uan-*  -> login node on ALCF systems
           # x####c#s#b#n# -> compute node on ALCF systems
```

On a login node, you cannot run heavy compute operations, those are done on compute nodes, but some Python for reading and analyzing files is fine. If you think you need to perform compute operations, stop and ask the user to submit a job. Never report a Python, MPI or GPU change as verified from a login node.
Say plainly that it is unverified.
Claiming otherwise is worse than saying nothing, because it is believable.

See the skills at @.claude/skills/hpc-compute-node/SKILL.md for information on how to get access to Python and the ML libraries depending on the system you are on.

### Getting onto a compute node

Ask the user -- do not submit jobs on your own initiative. If there is a job running, ask if you can use it, the job may be reserved for other work. 

To see wich jobs are running, see the skills at @.claude/skills/hpc-compute-node/SKILL.md.

Inside the compute node, you likely have to load the environment first; without it there is no `python` at all. See the skills for what to load depending on the system.

Two things that will waste your time otherwise: `/tmp` is **not** shared with
the login node, so write scratch to the project filesystem; and output piped
back over `ssh` buffers badly, so have long jobs write results to a file under
the project tree and poll that instead.

## Verifying a distributed change

A clean exit is not evidence. For anything touching communication, the bar is a
**bit-identical comparison against the implementation being replaced**: run the
same example both ways and diff the per-step losses.

```bash
# in examples/tgv_gnn_offline_traj, 4 ranks, once per mode
mpiexec -n 4 -ppn 4 python ../../3rd_party/gnn/dist-gnn/main.py \
    master_addr=$(hostname|cut -d. -f1) halo_swap_mode=<MODE> ...
```

Identical losses to the last digit across every step is the pass. Anything less
specific -- "it ran", "loss looks reasonable", "the test is green" -- does not
distinguish a correct change from one that is quietly wrong.

Before changing anything in `3rd_party/gnn/dist-gnn` that sends or receives,
read `3rd_party/gnn/dist-gnn/DEVELOPING.md` first. It records failures that
cost multi-node debugging sessions and that the test suite cannot catch.

## Style

Python is linted with ruff (`3rd_party/gnn/pyproject.toml`): `line-length = 80`,
double quotes, isort. The tree has pre-existing violations, so the bar is **add
no new ones** -- do not reformat code you were not asked to touch, and do not
"fix" unrelated findings while you are in a file.

Match the density and voice of the surrounding comments. This codebase explains
*why* in prose, especially where a line is load-bearing for a non-obvious
reason; a comment that only restates the code is noise.

## Commits

Short imperative subject with the PR number, e.g.
`Resolve possible race in get_edge_weights (#91)`. PRs target `main`. Important, commit
only when asked, otherwise leave changes uncommitted. 

## Vendored trees

`3rd_party/nek5000` and `3rd_party/gslib` are upstream code. Read them freely --
they are often where the answer is -- but changing them is a separate
conversation, not something to fold into an unrelated fix.
