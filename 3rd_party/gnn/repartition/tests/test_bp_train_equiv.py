"""Multi-step BP5 training data equivalence test.

Where test_adios_equiv.py gates a single-step solution stream, this gates
the multi-step trainingData.bp written by adios_client_t (trajGenWriteBP /
writeToFileBP): every ADIOS step is one snapshot, and each snapshot must
route to exactly what the .bin path would produce for the same data.

    python repartition/tests/gen_synthetic.py --out /tmp/synth --src-size 2
    python repartition/tests/bin_to_bp.py --src /tmp/synth --out /tmp/synth_bp
    mpirun -n M python repartition/tests/test_bp_train_equiv.py \
        --bin /tmp/synth --bp /tmp/synth_bp --method rcb

Run at M == W and M != W: the routed field must be identical to the .bin
reference in both cases, since the routing plan comes from the graph and is
reused across every step.

The fixture scales snapshot k by (k+1), so an off-by-one in the step walk
shows up as a clean integer factor rather than a subtle numerical drift.
"""

# ruff: file-ignore[module-import-not-at-top-of-file]  # imports follow sys.path setup

import argparse
import os
import sys

import numpy as np

# torch is imported here, sometimes torch before mpi import matters
try:
    import torch  # noqa: F401
except ImportError:
    pass

from mpi4py import MPI

HERE = os.path.dirname(os.path.abspath(__file__))
PKG_PARENT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, PKG_PARENT)

from repartition import (  # noqa: E402
    AdiosSource,
    BinSource,
    Repartitioner,
    open_bp_read,
)

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
SIZE = COMM.Get_size()


def check(cond, msg):
    all_ok = COMM.allreduce(bool(cond), op=MPI.LAND)
    if not all_ok:
        if not cond:
            print(f"[RANK {RANK}] FAIL: {msg}", flush=True)
        COMM.Barrier()
        if RANK == 0:
            print(f"FAILED: {msg}", flush=True)
        sys.exit(1)
    if RANK == 0:
        print(f"PASS: {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True, help="gnn_outputs_* directory")
    ap.add_argument("--bp", required=True, help="dir holding graph.bp")
    ap.add_argument(
        "--train-bp",
        default="trainingData.bp",
        help="training data file name inside --bp",
    )
    ap.add_argument("--method", default="rcb")
    args = ap.parse_args()

    bsrc = BinSource(args.bin)
    asrc = AdiosSource(os.path.join(args.bp, "graph.bp"), comm=COMM)
    brp = Repartitioner(bsrc, COMM, method=args.method)
    arp = Repartitioner(asrc, COMM, method=args.method)

    check(
        np.array_equal(brp.elems.ordinals, arp.elems.ordinals),
        "the two sources place the same elements on this rank",
    )

    # reference: the unscaled field routed through the .bin path
    src_size = bsrc.src_size
    ref = brp.read_field(
        lambda s: os.path.join(
            args.bin, f"u_field_rank_{s}_size_{src_size}.bin"
        ),
        ncols=3,
    )

    nsteps = 0
    seen_tsteps = []
    path = os.path.join(args.bp, args.train_bp)
    with open_bp_read(path, COMM) as st:
        for _ in st.steps():
            asrc.attach_field_stream(st)
            u = arp.read_field("u", ncols=3)
            tstep = int(np.asarray(st.read("tstep")).reshape(-1)[0])
            seen_tsteps.append(tstep)

            scale = nsteps + 1
            same_shape = u.shape == ref.shape
            check(
                same_shape and np.allclose(u, scale * ref, rtol=0, atol=0),
                f"step {nsteps} (tstep={tstep}) routes to {scale}x the .bin "
                f"reference (shape {u.shape})",
            )
            nsteps += 1

    check(nsteps > 0, "training file has at least one step")
    check(
        seen_tsteps == sorted(seen_tsteps),
        f"tstep scalars are monotonic ({seen_tsteps})",
    )
    # a rank-0-only scalar written at start {0} must be visible to every rank
    check(
        all(t != 0 for t in seen_tsteps),
        "tstep scalars are non-zero on every rank (start-{0})",
    )

    if RANK == 0:
        print(
            f"ALL CHECKS PASSED (M={SIZE}, W={bsrc.src_size}, "
            f"steps={nsteps}, method={args.method})"
        )


if __name__ == "__main__":
    main()
