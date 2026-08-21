"""Materialize a gnn_outputs directory (and optional field/trajectory data)
repartitioned to the current MPI communicator size.

    mpirun -n M python -m repartition.cli --src-dir gnn_outputs_poly_7 \
        [--out-dir OUT] [--method rcb|block] [--fld] \
        [--traj-dir traj_poly_7/tinit_0.000000_dtfactor_10 [--traj-out OUT2]]

Writes the five graph arrays and the Np file named *_rank_r_size_M
(model-agnostic), plus -- as a dist-gnn integration, skippable with
--no-halo -- the three halo .npy files (halo_info / node_degree /
edge_weights, via dist-gnn's create_halo_info_par functions), so existing
offline dist-gnn training/inference runs unchanged at the new size. --fld
routes every fld_* snapshot found in the source dir; --traj-dir routes
every file of every data_rank_*_size_S trajectory subdirectory.

Run from 3rd_party/gnn (or with that directory on PYTHONPATH).
"""

# ruff: file-ignore[module-import-not-at-top-of-file]  # imports follow sys.path setup

import argparse
import glob
import os
import re
import sys

import numpy as np
from mpi4py import MPI

HERE = os.path.dirname(os.path.abspath(__file__))
PKG_PARENT = os.path.abspath(os.path.join(HERE, ".."))
DISTGNN = os.path.join(PKG_PARENT, "dist-gnn")
sys.path.insert(0, PKG_PARENT)

from repartition import (
    BinSource,
    Repartitioner,
)

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
SIZE = COMM.Get_size()

# nekRS pads node arrays to alignStride<dfloat> boundaries: at most
# ALIGN_SIZE_BYTES / sizeof(double) = 32 extra rows (src/core/nekrsSys.hpp)
MAX_PAD_ROWS = 32


def infer_ncols(path, n_nodes_src_rank):
    items = os.path.getsize(path) // 8
    hits = [
        c
        for c in (1, 2, 3, 4, 6, 9)
        if items % c == 0
        and n_nodes_src_rank
        <= items // c
        < n_nodes_src_rank + MAX_PAD_ROWS
    ]
    if len(hits) != 1:
        raise RuntimeError(
            f"cannot infer column count of {path} "
            f"(items={items}, N={n_nodes_src_rank}, candidates={hits})"
        )
    return hits[0]


def write_graph(rp, out_dir):
    arrs = rp.graph_arrays()
    sfx = f"_rank_{RANK}_size_{SIZE}.bin"
    arrs["pos"].tofile(os.path.join(out_dir, "pos_node" + sfx))
    arrs["global_ids"].tofile(os.path.join(out_dir, "global_ids" + sfx))
    arrs["edge_index"].tofile(os.path.join(out_dir, "edge_index" + sfx))
    arrs["local_unique_mask"].tofile(
        os.path.join(out_dir, "local_unique_mask" + sfx)
    )
    arrs["halo_unique_mask"].tofile(
        os.path.join(out_dir, "halo_unique_mask" + sfx)
    )
    if RANK == 0:
        with open(
            os.path.join(out_dir, f"Np_rank_0_size_{SIZE}"), "w"
        ) as f:
            f.write(f"{rp.Np}\n")


def write_halo_files(rp, out_dir):
    """dist-gnn integration: derive halo_info / node_degree / edge_weights
    at the new size with dist-gnn's own machinery. Other models can skip
    this (--no-halo) and derive their own metadata from the five arrays."""
    if DISTGNN not in sys.path:
        sys.path.insert(0, DISTGNN)
    import create_halo_info_par as chip
    import graph_connectivity as gcon
    import torch
    import torch_geometric.utils as pyg_utils
    from torch_geometric.data import Data

    arrs = rp.graph_arrays()
    pos = torch.tensor(arrs["pos"])
    data_full = Data(
        x=None,
        edge_index=torch.tensor(arrs["edge_index"].astype(np.int64).T),
        pos_orig=pos,
        pos=pos,
        global_ids=torch.tensor(arrs["global_ids"].reshape(-1)),
        local_unique_mask=torch.tensor(
            arrs["local_unique_mask"].astype(np.int64)
        ),
        halo_unique_mask=torch.tensor(
            arrs["halo_unique_mask"].astype(np.int64)
        ),
    )
    data_full.edge_index = pyg_utils.remove_self_loops(
        data_full.edge_index
    )[0]
    data_full.edge_index = pyg_utils.coalesce(data_full.edge_index)
    data_full.edge_index = pyg_utils.to_undirected(data_full.edge_index)
    data_full.local_ids = torch.arange(pos.shape[0])

    data_reduced, idx_f2r = gcon.get_reduced_graph(data_full)
    # gid==0 (element-interior) nodes must get unique negative ids before
    # the edge-weight cantor pairing, exactly as create_halo_info_par does
    # (otherwise all interior nodes alias to gid 0 and duplicate counting
    # is wrong)
    gcon.update_global_ids(data_full, data_reduced, idx_f2r)

    halo_ids = chip.get_reduced_halo_ids(data_reduced)
    halo_info_glob = chip.get_halo_info_fast(data_reduced, halo_ids)
    halo_info = halo_info_glob[RANK]
    node_degree = chip.get_node_degree(data_reduced, halo_info)
    edge_freq = chip.get_edge_weights(data_reduced, halo_info_glob)

    sfx = f"_rank_{RANK}_size_{SIZE}.npy"
    np.save(
        os.path.join(out_dir, "halo_info" + sfx),
        halo_info.numpy() if hasattr(halo_info, "numpy") else halo_info,
    )
    np.save(
        os.path.join(out_dir, "node_degree" + sfx),
        np.asarray(node_degree),
    )
    np.save(
        os.path.join(out_dir, "edge_weights" + sfx),
        np.asarray(edge_freq),
    )


def route_fld_files(rp, src, out_dir):
    n0 = int(src.ne_per_src[0]) * src.Np
    pat = os.path.join(
        src.src_dir, f"fld_*_rank_0_size_{src.src_size}.bin"
    )
    for f0 in sorted(glob.glob(pat)):
        base = os.path.basename(f0)
        prefix = re.sub(rf"_rank_0_size_{src.src_size}\.bin$", "", base)
        ncols = infer_ncols(f0, n0)
        routed = rp.read_field(
            lambda s, p=prefix: os.path.join(
                src.src_dir, f"{p}_rank_{s}_size_{src.src_size}.bin"
            ),
            ncols=ncols,
        )
        out = os.path.join(
            out_dir, f"{prefix}_rank_{RANK}_size_{SIZE}.bin"
        )
        routed.tofile(out)
        if RANK == 0:
            print(f"routed {prefix} ({ncols} cols)", flush=True)


def route_traj_dir(rp, src, traj_dir, traj_out):
    n0 = int(src.ne_per_src[0]) * src.Np
    src_sub = os.path.join(
        traj_dir, f"data_rank_0_size_{src.src_size}"
    )
    out_sub = os.path.join(traj_out, f"data_rank_{RANK}_size_{SIZE}")
    if RANK == 0:
        os.makedirs(out_sub, exist_ok=True)
    COMM.Barrier()
    os.makedirs(out_sub, exist_ok=True)
    for fname in sorted(os.listdir(src_sub)):
        ncols = infer_ncols(os.path.join(src_sub, fname), n0)
        routed = rp.read_field(
            lambda s, fn=fname: os.path.join(
                traj_dir, f"data_rank_{s}_size_{src.src_size}", fn
            ),
            ncols=ncols,
        )
        routed.tofile(os.path.join(out_sub, fname))
    if RANK == 0:
        print(
            f"routed trajectory ({len(os.listdir(src_sub))} files) "
            f"-> {out_sub}",
            flush=True,
        )


def route_fld_snapshots(rp, snapshot_files, field_map, out_dir):
    """Write .f field records as fld_<name>_time_<t>_rank_R_size_M.bin."""
    from repartition.fld import FldHeader

    for path in snapshot_files:
        hdr = FldHeader(path)
        t = round(hdr.time * 10.0) / 10.0
        for record, name, ncols in field_map:
            routed = rp.read_field((path, record), ncols=ncols)
            out = os.path.join(
                out_dir,
                f"fld_{name}_time_{t:.1f}_rank_{RANK}_size_{SIZE}.bin",
            )
            routed.tofile(out)
            if RANK == 0:
                print(
                    f"routed {os.path.basename(path)}:{record} -> "
                    f"fld_{name}_time_{t:.1f} ({ncols} cols)",
                    flush=True,
                )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-dir", help="gnn_outputs_poly_* source directory")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument(
        "--method", default="rcb", choices=["rcb", "block", "parrsb"]
    )
    ap.add_argument("--fld", action="store_true")
    ap.add_argument("--traj-dir", default=None)
    ap.add_argument("--traj-out", default=None)
    ap.add_argument(
        "--fld-mesh",
        help=".f file with the mesh record (X); reconstructs the graph "
        "from coordinates instead of reading gnn_outputs",
    )
    ap.add_argument(
        "--fld-snapshots",
        nargs="*",
        default=None,
        help=".f files whose U/P records become fld_u/fld_p training "
        "snapshots (default: the --fld-mesh file)",
    )
    ap.add_argument(
        "--periodic",
        default="",
        help="periodic axes for .f coincidence matching, e.g. xyz",
    )
    ap.add_argument(
        "--fld-traj",
        nargs="*",
        default=None,
        help=".f files (time-ordered) whose U records become a "
        "data_rank_R_size_M/u_step_<istep>.bin trajectory",
    )
    ap.add_argument(
        "--fld-traj-out",
        default=None,
        help="output trajectory directory (contains data_rank_* subdirs)",
    )
    ap.add_argument(
        "--no-halo",
        action="store_true",
        help="skip writing the dist-gnn halo .npy files (for models that "
        "derive their own halo metadata from the five graph arrays)",
    )
    args = ap.parse_args()

    if args.fld_mesh:
        from repartition.fld import FldSource

        periodic = tuple(ax in args.periodic.lower() for ax in "xyz")
        src = FldSource(args.fld_mesh, periodic=periodic)
        out_dir = args.out_dir
        if out_dir is None:
            raise SystemExit("--out-dir is required with --fld-mesh")
        desc = f"{args.fld_mesh} (.f mesh, periodic={args.periodic!r}"
    else:
        if not args.src_dir:
            raise SystemExit("pass --src-dir or --fld-mesh")
        src = BinSource(args.src_dir)
        out_dir = args.out_dir or args.src_dir
        if src.src_size == SIZE and os.path.realpath(
            out_dir
        ) == os.path.realpath(args.src_dir):
            raise SystemExit(
                "output size equals source size and out-dir equals "
                "src-dir; this would overwrite the source files. "
                "Pass --out-dir."
            )
        desc = f"{args.src_dir} (size {src.src_size}"
    if RANK == 0:
        os.makedirs(out_dir, exist_ok=True)
        print(
            f"repartitioning {desc}, Np={src.Np}, "
            f"{src.n_elements_total} elements) -> size {SIZE} "
            f"with method={args.method}",
            flush=True,
        )
    COMM.Barrier()

    rp = Repartitioner(src, COMM, method=args.method)
    write_graph(rp, out_dir)
    if SIZE > 1 and not args.no_halo:
        write_halo_files(rp, out_dir)
    if args.fld_mesh:
        if args.fld_traj:
            traj_out = args.fld_traj_out or out_dir
            sub = os.path.join(
                traj_out, f"data_rank_{RANK}_size_{SIZE}"
            )
            os.makedirs(sub, exist_ok=True)
            for i, path in enumerate(args.fld_traj):
                routed = rp.read_field((path, "U"), ncols=3)
                routed.tofile(os.path.join(sub, f"u_step_{i}.bin"))
            if RANK == 0:
                print(
                    f"routed .f trajectory ({len(args.fld_traj)} files) "
                    f"-> {sub}",
                    flush=True,
                )
        else:
            snaps = args.fld_snapshots or [args.fld_mesh]
            route_fld_snapshots(
                rp, snaps, [("U", "u", 3), ("P", "p", 1)], out_dir
            )
    if args.fld:
        route_fld_files(rp, src, out_dir)
    if args.traj_dir:
        route_traj_dir(rp, src, args.traj_dir, args.traj_out or args.traj_dir)
    COMM.Barrier()
    if RANK == 0:
        print("done", flush=True)


if __name__ == "__main__":
    main()
