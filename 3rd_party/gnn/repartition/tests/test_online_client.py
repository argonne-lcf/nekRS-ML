"""OnlineClient ADIOS layout test -- the W == M path and the return path.

test_adios_equiv.py covers AdiosSource against BinSource. This covers the
thin layer in dist-gnn/client.py around it: the direct per-block read taken
when the ML rank count equals the nekRS rank count (the path production runs
today), and the ADIOS put_array return path.

Ground truth here is not another reader but the .bin files themselves, so a
shared misreading cannot pass. Run at M == W (4 for the standard fixture):

    mpirun -n 4 python repartition/tests/test_online_client.py \
        --bin /tmp/synth_pad --bp /tmp/synth_bp
"""

# ruff: file-ignore[module-import-not-at-top-of-file]

import argparse
import os
import sys

import numpy as np
from mpi4py import MPI

HERE = os.path.dirname(os.path.abspath(__file__))
GNN = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, GNN)
sys.path.insert(0, os.path.join(GNN, "dist-gnn"))

import adios2  # noqa: E402

from client import OnlineClient  # noqa: E402
from repartition import BinSource  # noqa: E402

COMM = MPI.COMM_WORLD
RANK, SIZE = COMM.Get_rank(), COMM.Get_size()
HAS_MPI_ADIOS = getattr(adios2, "is_built_with_mpi", False)


def check(cond, msg):
    if not COMM.allreduce(bool(cond), op=MPI.LAND):
        if not cond:
            print(f"[RANK {RANK}] FAIL: {msg}", flush=True)
        COMM.Barrier()
        if RANK == 0:
            print(f"FAILED: {msg}", flush=True)
        sys.exit(1)
    if RANK == 0:
        print(f"PASS: {msg}", flush=True)


def make_stub(bp_dir):
    """An OnlineClient with its ADIOS state filled in, no live simulation."""
    c = OnlineClient.__new__(OnlineClient)
    c.backend = "adios"
    c.comm, c.rank, c.size = COMM, RANK, SIZE
    c.timers = {"init": [], "data": [], "meta_data": []}
    c.solutionStream = None
    c.graph_source = None
    c.repart = None
    sys.path.insert(0, GNN)
    from repartition import AdiosSource

    src = AdiosSource(os.path.join(bp_dir, "graph.bp"), comm=COMM)
    c.graph_source = src
    c.N_list = [int(n) for n in src.n_per_src]
    c.num_edges_list = [int(e) for e in src.num_edges_per_src]
    c.fieldOffset_list = [int(f) for f in src.fo_per_src]
    return c, src


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--bp", required=True)
    args = ap.parse_args()

    bsrc = BinSource(args.bin)
    if SIZE != bsrc.src_size:
        if RANK == 0:
            print(
                f"SKIP: this test covers the W == M path; run with "
                f"-n {bsrc.src_size}"
            )
        return

    cwd = os.getcwd()
    os.chdir(args.bp)  # _read_own_graph_block opens "graph.bp" by name
    try:
        c, src = make_stub(args.bp)
        check(
            src.Np == bsrc.Np,
            f"Np resolved past the start-{{1}} defect ({src.Np}, not 0)",
        )
        blk = c._read_own_graph_block(src)

        n = c.N_list[RANK]
        suffix = f"_rank_{RANK}_size_{SIZE}.bin"
        ref = {
            "pos": np.fromfile(
                os.path.join(args.bin, "pos_node" + suffix), dtype=np.float64
            ).reshape(-1, 3),
            "global_ids": np.fromfile(
                os.path.join(args.bin, "global_ids" + suffix), dtype=np.int64
            ),
            "local_unique_mask": np.fromfile(
                os.path.join(args.bin, "local_unique_mask" + suffix),
                dtype=np.int32,
            ),
            "halo_unique_mask": np.fromfile(
                os.path.join(args.bin, "halo_unique_mask" + suffix),
                dtype=np.int32,
            ),
        }
        for k, v in ref.items():
            got = np.asarray(blk[k]).reshape(v.shape)
            check(
                np.array_equal(got, v),
                f"_read_own_graph_block {k} matches "
                f"{'pos_node' if k == 'pos' else k}{suffix}",
            )

        ei_ref = (
            np
            .fromfile(
                os.path.join(args.bin, "edge_index" + suffix), dtype=np.int32
            )
            .reshape(-1, 2)
            .T
        )
        check(
            blk["edge_index"].shape == ei_ref.shape
            and np.array_equal(blk["edge_index"], ei_ref),
            f"_read_own_graph_block edge_index is (2,E)={ei_ref.shape} and "
            f"matches the .bin",
        )

        # -- the fieldOffset fix: read the padded solution block ----------
        u_ref = np.fromfile(
            os.path.join(args.bin, f"u_field_rank_{RANK}_size_{SIZE}.bin"),
            dtype=np.float64,
        ).reshape(-1, 3)[:n]
        with adios2.Stream(os.path.join(args.bp, "solution.bp"), "r") as sol:
            sol.begin_step()
            c.solutionStream = sol
            u_got = c._read_own_field_block("in_u", 3)
            # what the pre-fix code did: one contiguous N*3 read, F-reshaped
            base = sum(c.fieldOffset_list[:RANK]) * 3
            old = sol.read("in_u", [base], [n * 3]).reshape((-1, 3), order="F")
            sol.end_step()
        c.solutionStream = None

        pad = c.fieldOffset_list[RANK] - n
        check(
            u_got.shape == u_ref.shape and np.allclose(u_got, u_ref),
            f"_read_own_field_block in_u matches the .bin "
            f"(N={n}, fieldOffset={c.fieldOffset_list[RANK]}, pad={pad})",
        )
        # the fixture must actually exercise padding, or the test is vacuous
        check(
            COMM.allreduce(int(pad > 0), op=MPI.SUM) > 0,
            "fixture has non-zero alignStride padding on some rank",
        )
        stale = not np.allclose(old, u_ref)
        check(
            stale,
            "the pre-fix contiguous N*3 read is demonstrably wrong here "
            f"(max|diff|={np.abs(old - u_ref).max():.3e})",
        )

        # -- return path --------------------------------------------------
        if HAS_MPI_ADIOS:
            gid = ref["global_ids"][:n].astype(np.int64)
            payload = u_ref.astype(np.float32)
            c.put_array(
                f"checkpt_u_rank_{RANK}_size_{SIZE}", payload, global_ids=gid
            )
            COMM.Barrier()
            counts = COMM.allgather(n)
            start, total = sum(counts[:RANK]), sum(counts)
            with adios2.Stream("checkpt_u.bp", "r", COMM) as st:
                st.begin_step()
                back = (
                    st
                    .read("checkpt_u", [start * 3], [n * 3])
                    .reshape((3, -1))
                    .T
                )
                gback = st.read("checkpt_u_global_ids", [start], [n])
                st.end_step()
            check(
                np.allclose(back, payload) and np.array_equal(gback, gid),
                f"put_array round-trips component-major with global_ids "
                f"(global rows {total})",
            )
        elif RANK == 0:
            print("SKIP: put_array round-trip needs an MPI-enabled adios2")

        if RANK == 0:
            print(f"ALL CHECKS PASSED (M=W={SIZE})")
    finally:
        os.chdir(cwd)


if __name__ == "__main__":
    main()
