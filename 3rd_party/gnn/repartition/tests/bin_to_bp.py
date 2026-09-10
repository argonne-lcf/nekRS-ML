"""Turn a gnn_outputs_* binary directory into the ADIOS2 files nekRS writes.

Serial: one process emits W blocks per variable, reproducing what W nekRS
writer ranks would have produced. That gives a local fixture for the online
reader without needing an MPI-enabled ADIOS2 or a running simulation.

    python repartition/tests/bin_to_bp.py \
        --src /tmp/synth_pad --out /tmp/synth_bp

Everything the reader must honour is encoded in LAYOUT below, each entry
citing the writer line it mirrors. This file is the fixture's contract with
nekRS: if it drifts from the C++, the equivalence test still passes while the
real stream breaks, so it is verified against the writer rather than against
the reader.
"""

import argparse
import os
import shutil

import numpy as np
from adios2 import Stream

# --- writer layout (src/plugins/gnn.cpp, trajGen.cpp, adiosStreamer.cpp) ----
# graph.bp, per writer rank w with N_w nodes and E_w edges:
#   N            int32   shape {W}            start {w}          count {1}
#   num_edges    int32   shape {W}            start {w}          count {1}
#   Np           int32   shape {1}            start {1}          count {1}
#                        ^ off-by-one in the writer (gnn.cpp:303); readers
#                          must fetch it without a selection.
#   pos_node     f8      shape {3*sum(N)}     start {3*off_w}    count {3*N_w}
#                        component-major: [x(0..N_w-1), y(...), z(...)]
#   global_ids   i8      shape {sum(N)}       start {off_w}      count {N_w}
#   local_unique_mask,
#   halo_unique_mask
#                int32   shape {sum(N)}       start {off_w}      count {N_w}
#   edge_index   int32   shape {2*sum(E)}     start {2*eoff_w}   count {2*E_w}
#                        component-major: [nei(0..E_w-1), own(...)],
#                        node ids are block-LOCAL
#   field_offset int64   shape {W}            start {w}          count {1}
#                        alignStride(N_w) as the writer computed it
#                        (gnn.cpp:300); the reader prefers this to
#                        recomputing align_stride(N_w) itself.
#   No padding anywhere in graph.bp: N_w == Ne_w * Np exactly.
#
# solution stream (in_u/out_u), per writer rank w:
#   fo_w = alignStride(N_w) = ceil(N_w/32)*32   -- up to 31 pad doubles
#                                                  PER COMPONENT
#   in_u/out_u   f8      shape {3*sum(fo)}    start {3*foff_w}   count {3*fo_w}
#                        component-major: [u(0..fo_w-1), v(...), w(...)]
ALIGN = 32


def align_stride(n, align=ALIGN):
    return -(-int(n) // align) * align


def load_bin_dir(src, src_size=None):
    if src_size is None:
        import glob
        import re

        hits = glob.glob(os.path.join(src, "pos_node_rank_0_size_*.bin"))
        src_size = int(re.search(r"_size_(\d+)\.bin$", hits[0]).group(1))
    np_pts = int(
        float(
            open(os.path.join(src, f"Np_rank_0_size_{src_size}"))
            .read()
            .split()[0]
        )
    )

    def path(name, w, ext=".bin"):
        return os.path.join(src, f"{name}_rank_{w}_size_{src_size}{ext}")

    blocks = []
    for w in range(src_size):
        pos = np.fromfile(path("pos_node", w), dtype=np.float64).reshape(-1, 3)
        gid = np.fromfile(path("global_ids", w), dtype=np.int64).reshape(-1)
        ei = np.fromfile(path("edge_index", w), dtype=np.int32).reshape(-1, 2)
        lum = np.fromfile(path("local_unique_mask", w), dtype=np.int32)
        hum = np.fromfile(path("halo_unique_mask", w), dtype=np.int32)
        assert pos.shape[0] == gid.shape[0] == lum.shape[0] == hum.shape[0]
        assert pos.shape[0] % np_pts == 0, "graph.bp blocks are never padded"
        blocks.append({
            "pos": pos,
            "gid": gid,
            "ei": ei,
            "lum": lum,
            "hum": hum,
        })
    return src_size, np_pts, blocks


def write_graph_bp(out, np_pts, blocks):
    W = len(blocks)
    N = np.array([b["pos"].shape[0] for b in blocks], dtype=np.int32)
    E = np.array([b["ei"].shape[0] for b in blocks], dtype=np.int32)
    FO = np.array([align_stride(int(n)) for n in N], dtype=np.int64)
    noff = np.concatenate([[0], np.cumsum(N.astype(np.int64))])
    eoff = np.concatenate([[0], np.cumsum(E.astype(np.int64))])
    ntot, etot = int(noff[-1]), int(eoff[-1])

    shutil.rmtree(out, ignore_errors=True)
    with Stream(out, "w") as s:
        s.begin_step()
        # Np exactly as the writer defines it -- shape {1} start {1} count
        # {1}, a block one past the end of its own global shape, written by
        # rank 0 alone (gnn.cpp:303,320-322). Reproduced rather than
        # corrected: a no-selection read of this returns 0, not Np, so a
        # fixture that wrote it correctly would hide the bug from the reader.
        s.write("Np", np.array([np_pts], dtype=np.int32), [1], [1], [1])
        for w, b in enumerate(blocks):
            s.write("N", N[w : w + 1], [W], [w], [1])
            s.write("num_edges", E[w : w + 1], [W], [w], [1])
            s.write("field_offset", FO[w : w + 1], [W], [w], [1])

            n = int(N[w])
            s.write(
                "pos_node",
                np.ascontiguousarray(b["pos"].T).reshape(-1),  # component-major
                [3 * ntot],
                [3 * int(noff[w])],
                [3 * n],
            )
            s.write("global_ids", b["gid"], [ntot], [int(noff[w])], [n])
            s.write("local_unique_mask", b["lum"], [ntot], [int(noff[w])], [n])
            s.write("halo_unique_mask", b["hum"], [ntot], [int(noff[w])], [n])

            e = int(E[w])
            s.write(
                "edge_index",
                np.ascontiguousarray(b["ei"].T).reshape(-1),  # [nei..., own...]
                [2 * etot],
                [2 * int(eoff[w])],
                [2 * e],
            )
        s.end_step()
    return N, E


def write_solution_bp(out, name, blocks, fields):
    """in_u/out_u layout: fieldOffset-strided, component-major, padded."""
    N = [b["pos"].shape[0] for b in blocks]
    fo = [align_stride(n) for n in N]
    foff = np.concatenate([[0], np.cumsum(np.array(fo, dtype=np.int64))])
    tot = int(foff[-1])

    shutil.rmtree(out, ignore_errors=True)
    with Stream(out, "w") as s:
        s.begin_step()
        for var, per_block in fields.items():
            for w, arr in enumerate(per_block):
                n, f = N[w], fo[w]
                assert arr.shape == (n, 3), f"{var} block {w}: {arr.shape}"
                blk = np.zeros(3 * f, dtype=np.float64)
                for c in range(3):
                    blk[c * f : c * f + n] = arr[:, c]
                s.write(var, blk, [3 * tot], [3 * int(foff[w])], [3 * f])
        s.end_step()
    return fo


def write_training_bp(out, blocks, snapshots, mode="traj"):
    """trainingData.bp: the multi-step file written by adios_client_t.

    One ADIOS step per snapshot (trajGen.cpp trajGenWriteBP /
    gnn.cpp writeToFileBP). Field variables use exactly the padded
    component-major layout of in_u/out_u, so the same reader path applies.

    Every step carries BOTH rank-0 scalars -- int 'tstep' and f8 'time' --
    written at start {0} (NOT the start-{1} defect that Np has in graph.bp),
    because writeToFileBP puts both unconditionally: the reader takes
    whichever suits its dist-gnn time-dependency mode.

    snapshots is a list of (scalar, {varname: [per-writer (n, ncols) array]}).
    `mode` says which scalar `scalar` is; the other is derived so the file
    still carries the pair the real writer emits.
    """
    N = [b["pos"].shape[0] for b in blocks]
    fo = [align_stride(n) for n in N]
    foff = np.concatenate([[0], np.cumsum(np.array(fo, dtype=np.int64))])
    tot = int(foff[-1])

    shutil.rmtree(out, ignore_errors=True)
    with Stream(out, "w") as s:
        for scalar, fields in snapshots:
            s.begin_step()
            for var, per_block in fields.items():
                ncols = per_block[0].shape[1]
                for w, arr in enumerate(per_block):
                    n, f = N[w], fo[w]
                    assert arr.shape == (n, ncols), (
                        f"{var} block {w}: {arr.shape}"
                    )
                    blk = np.zeros(ncols * f, dtype=np.float64)
                    for c in range(ncols):
                        blk[c * f : c * f + n] = arr[:, c]
                    s.write(
                        var,
                        blk,
                        [ncols * tot],
                        [ncols * int(foff[w])],
                        [ncols * f],
                    )
            if mode == "traj":
                tstep, t = int(scalar), 0.01 * float(scalar)
            else:
                tstep, t = round(float(scalar) * 100), float(scalar)
            s.write("tstep", np.array([tstep], dtype=np.int32), [1], [0], [1])
            s.write("time", np.array([t], dtype=np.float64), [1], [0], [1])
            s.end_step()
    return fo


def write_checkpoint_bp(out, blocks, per_block):
    """checkpoint.bp: identical layout to in_u/out_u, one 'checkpoint' var.

    adiosStreamer.cpp:154-176 defines it {global_fo*dim},{offset_fo*dim},
    {fo*dim}, and the offsets there are the true per-writer scan of
    fieldOffset built in gnn.cpp:256-274 -- so writer blocks may differ in
    size and the layout still holds.
    """
    return write_solution_bp(
        out, "checkpoint", blocks, {"checkpoint": per_block}
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="gnn_outputs_* directory")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--src-size", type=int, default=None)
    ap.add_argument(
        "--field",
        default="u_field",
        help="bin node-field prefix to emit as in_u/out_u (blank to skip)",
    )
    args = ap.parse_args()

    src_size, np_pts, blocks = load_bin_dir(args.src, args.src_size)
    os.makedirs(args.out, exist_ok=True)
    N, E = write_graph_bp(os.path.join(args.out, "graph.bp"), np_pts, blocks)
    print(f"graph.bp: W={src_size} Np={np_pts} N={list(N)} num_edges={list(E)}")

    if args.field:
        per_block = []
        for w in range(src_size):
            p = os.path.join(
                args.src, f"{args.field}_rank_{w}_size_{src_size}.bin"
            )
            rows = np.fromfile(p, dtype=np.float64).reshape(-1, 3)
            per_block.append(rows[: blocks[w]["pos"].shape[0]])  # drop padding
        fo = write_solution_bp(
            os.path.join(args.out, "solution.bp"),
            "solution",
            blocks,
            {"in_u": per_block, "out_u": per_block},
        )
        print(f"solution.bp: in_u/out_u fieldOffset={fo} (N={list(N)})")

        write_checkpoint_bp(
            os.path.join(args.out, "checkpoint.bp"), blocks, per_block
        )
        print(f"checkpoint.bp: checkpoint fieldOffset={fo}")

        # multi-step trajectory file: 3 snapshots, each a distinct scaling of
        # the field so a mis-stepped read is detectable
        snaps = [
            (
                10 * (k + 1),
                {"u": [(k + 1) * rows for rows in per_block]},
            )
            for k in range(3)
        ]
        write_training_bp(
            os.path.join(args.out, "trainingData.bp"), blocks, snaps
        )
        print(f"trainingData.bp: 3 steps of u, fieldOffset={fo}")


if __name__ == "__main__":
    main()
