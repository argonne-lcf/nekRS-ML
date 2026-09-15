"""AdiosSource == BinSource equivalence test (the online-path acceptance gate).

The ADIOS reader and the binary reader describe the same graph in different
layouts: the .bin files are row-major per source rank, graph.bp is a global
concatenation of component-major per-writer blocks. Repartitioning either at
the same rank count must produce bit-identical arrays, because the element
ordinals, the partitioner input and the rebuild are all layout-agnostic.

    python repartition/tests/bin_to_bp.py \
        --src /tmp/synth_pad --out /tmp/synth_bp
    mpirun -n M python repartition/tests/test_adios_equiv.py \
        --bin /tmp/synth_pad --bp /tmp/synth_bp --method rcb

Any difference here is a layout misread in AdiosSource -- with one caveat
worth stating plainly: this test only proves the two readers AGREE. That the
BP layout itself matches what nekRS writes is established separately, by
deriving bin_to_bp.py from the C++ writer.
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
    ap.add_argument("--method", default="rcb")
    ap.add_argument(
        "--field",
        default="in_u",
        help="node field to compare (bp variable / bin u_field_*)",
    )
    args = ap.parse_args()

    bsrc = BinSource(args.bin)
    asrc = AdiosSource(os.path.join(args.bp, "graph.bp"), comm=COMM)

    # -- source metadata agrees before anything is read
    check(asrc.Np == bsrc.Np, f"Np agrees ({asrc.Np} == {bsrc.Np})")
    check(
        asrc.src_size == bsrc.src_size,
        f"writer size agrees ({asrc.src_size} == {bsrc.src_size})",
    )
    check(
        np.array_equal(asrc.ne_per_src, bsrc.ne_per_src),
        f"elements per writer agree ({list(asrc.ne_per_src)})",
    )

    brp = Repartitioner(bsrc, COMM, method=args.method)
    arp = Repartitioner(asrc, COMM, method=args.method)
    ba, aa = brp.graph_arrays(), arp.graph_arrays()

    check(
        np.array_equal(brp.elems.ordinals, arp.elems.ordinals),
        "the two sources place the same elements on this rank",
    )

    for key in (
        "pos",
        "global_ids",
        "edge_index",
        "local_unique_mask",
        "halo_unique_mask",
    ):
        b, a = ba[key], aa[key]
        same = b.shape == a.shape and b.dtype == a.dtype
        if same:
            same = np.array_equal(b, a)
        check(same, f"{key} identical (shape {b.shape}, dtype {b.dtype})")

    # -- node field: padded row-major .bin vs component-major padded BP block
    src_size = bsrc.src_size
    ub = brp.read_field(
        lambda s: os.path.join(
            args.bin, f"u_field_rank_{s}_size_{src_size}.bin"
        ),
        ncols=3,
    )
    from adios2 import Stream  # noqa: PLC0415

    with Stream(os.path.join(args.bp, "solution.bp"), "r") as sol:
        sol.begin_step()
        asrc.attach_field_stream(sol)
        ua = arp.read_field(args.field, ncols=3)
        sol.end_step()
    same_shape = ub.shape == ua.shape
    check(
        same_shape and np.array_equal(ub, ua),
        f"routed node field '{args.field}' identical (shape {ub.shape}, "
        f"max|diff| "
        f"{float(np.abs(ub - ua).max()) if same_shape else float('nan'):.3e})",
    )

    if RANK == 0:
        print(
            f"ALL CHECKS PASSED (M={SIZE}, W={bsrc.src_size}, "
            f"method={args.method})"
        )


if __name__ == "__main__":
    main()
