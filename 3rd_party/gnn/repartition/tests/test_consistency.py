"""MPI consistency test for the repartition package.

Run after gen_synthetic.py, with an arbitrary rank count M (independent of
the synthetic source size):

    mpirun -n M python repartition/tests/test_consistency.py \
        --src /tmp/synth --method rcb

Checks, in order:
1. mask invariants (one representative per on-rank gid, masks disjoint,
   coincident copies have identical coordinates)
2. the gid-canonicalized global reduced edge set is exactly the reference
3. sum over ranks of sum(1/node_degree) equals the global unique node count
   (halo_info / node_degree / edge_weights computed by the real
   create_halo_info_par functions at the current size)
4. one round of the model's aggregation semantics -- edge features scaled by
   1/edge_freq, add-aggregated at receivers, halo-swapped, and index-added
   into owned copies (mirrors gnn.py DistributedMessagePassingLayer) --
   matches the serial global-graph reference per gid to fp64 tolerance
5. node-level field files (fieldOffset-padded) are routed consistently
6. 1/node_degree-weighted global mean equals the serial unique-gid mean
"""

# ruff: file-ignore[module-import-not-at-top-of-file]  # imports follow sys.path setup

import argparse
import os
import sys

import numpy as np
from mpi4py import MPI

HERE = os.path.dirname(os.path.abspath(__file__))
PKG_PARENT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, PKG_PARENT)  # for `from repartition import ...`
sys.path.insert(0, os.path.join(PKG_PARENT, "dist-gnn"))  # halo machinery
sys.path.insert(0, HERE)

import create_halo_info_par as chip
import graph_connectivity as gcon
import torch
import torch_geometric.utils as pyg_utils
from gen_synthetic import (
    field_fn,
    node_feature,
)
from torch_geometric.data import (
    Data,
)

from repartition import (
    BinSource,
    Repartitioner,
)

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
SIZE = COMM.Get_size()


def fail(msg):
    print(f"[RANK {RANK}] FAIL: {msg}", flush=True)
    COMM.Abort(1)


def check(cond, msg):
    ok = bool(cond)
    all_ok = COMM.allreduce(ok, op=MPI.LAND)
    if not all_ok:
        if not ok:
            print(f"[RANK {RANK}] FAIL: {msg}", flush=True)
        COMM.Barrier()
        if RANK == 0:
            print(f"FAILED: {msg}", flush=True)
        sys.exit(1)
    if RANK == 0:
        print(f"PASS: {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--method", default="rcb")
    args = ap.parse_args()

    ref = np.load(os.path.join(args.src, "reference.npz"))
    ref_pairs = ref["ref_pairs"]
    uniq_gids_ref = ref["uniq_gids"]
    y_ref = ref["y_ref"]
    n_unique_ref = int(ref["n_unique"])

    src = BinSource(args.src)
    rp = Repartitioner(src, COMM, method=args.method)
    arrs = rp.graph_arrays()
    pos = arrs["pos"]
    gli = arrs["global_ids"].reshape(-1)
    ei = arrs["edge_index"].astype(np.int64)
    lum = arrs["local_unique_mask"]
    hum = arrs["halo_unique_mask"]
    n = pos.shape[0]

    # -- balance sanity
    counts = COMM.allgather(rp.elems.n_elements)
    if RANK == 0:
        print(
            f"method={args.method} src_size={src.src_size} -> {SIZE} ranks, "
            f"elements per rank: {counts}",
            flush=True,
        )
    check(min(counts) > 0, "every rank owns at least one element")

    # -- check 1: mask invariants
    n_uniq_local = np.unique(gli).shape[0]
    check(
        int((lum + hum).sum()) == n_uniq_local,
        "exactly one representative per on-rank gid",
    )
    check(int((lum * hum).sum()) == 0, "masks are disjoint")
    order = np.argsort(gli, kind="stable")
    gs, ps = gli[order], pos[order]
    same = gs[1:] == gs[:-1]
    check(
        np.allclose(ps[1:][same], ps[:-1][same]),
        "coincident copies have identical coordinates",
    )

    # -- check 2: global reduced edge set equals reference
    gmax = COMM.allreduce(int(gli.max()), op=MPI.MAX) + 1
    a, b = gli[ei[:, 0]], gli[ei[:, 1]]
    lo = np.minimum(a, b).astype(np.int64)
    hi = np.maximum(a, b).astype(np.int64)
    keys = np.unique(lo * gmax + hi)
    all_keys = COMM.gather(keys, root=0)
    if RANK == 0:
        union = np.unique(np.concatenate(all_keys))
        ref_keys = np.unique(
            ref_pairs[:, 0].astype(np.int64) * gmax + ref_pairs[:, 1]
        )
        ok = union.shape == ref_keys.shape and np.array_equal(union, ref_keys)
    else:
        ok = True
    check(ok, "global reduced edge set matches serial reference")

    # -- build reduced graph + halo machinery with the real pipeline
    data_full = Data(
        x=None,
        edge_index=torch.tensor(ei.T),
        pos_orig=torch.tensor(pos),
        pos=torch.tensor(pos),
        global_ids=torch.tensor(gli.astype(np.int64)),
        local_unique_mask=torch.tensor(lum.astype(np.int64)),
        halo_unique_mask=torch.tensor(hum.astype(np.int64)),
    )
    data_full.edge_index = pyg_utils.remove_self_loops(data_full.edge_index)[0]
    data_full.edge_index = pyg_utils.coalesce(data_full.edge_index)
    data_full.edge_index = pyg_utils.to_undirected(data_full.edge_index)
    data_full.local_ids = torch.arange(n)

    data_reduced, idx_f2r = gcon.get_reduced_graph(data_full)
    idx_r2f = gcon.get_upsample_indices(data_full, data_reduced, idx_f2r)
    check(
        torch.allclose(data_reduced.pos[idx_r2f], data_full.pos),
        "reduced<->full round trip",
    )

    halo_ids = chip.get_reduced_halo_ids(data_reduced)
    halo_info_glob = chip.get_halo_info_fast(data_reduced, halo_ids)
    halo_info = halo_info_glob[RANK].numpy().astype(np.int64)
    node_degree = chip.get_node_degree(data_reduced, torch.tensor(halo_info))
    node_degree = np.asarray(node_degree, dtype=np.float64)
    edge_freq = chip.get_edge_weights(data_reduced, halo_info_glob)
    edge_freq = np.asarray(edge_freq, dtype=np.float64)

    # -- check 3: effective node count
    eff = COMM.allreduce(float((1.0 / node_degree).sum()), op=MPI.SUM)
    check(
        abs(eff - n_unique_ref) < 1e-8 * n_unique_ref + 1e-6,
        f"sum(1/node_degree) == global unique nodes ({eff} vs {n_unique_ref})",
    )

    # -- check 4: one aggregation round with halo consistency
    gid_red = data_reduced.global_ids.numpy().astype(np.int64)
    ei_red = data_reduced.edge_index.numpy().astype(np.int64)
    n_local = gid_red.shape[0]
    n_halo = halo_info.shape[0]

    x = node_feature(gid_red)
    w = 1.0 / edge_freq
    agg = np.zeros((n_local + n_halo, 3), dtype=np.float64)
    np.add.at(agg, ei_red[1], x[ei_red[0]] * w[:, None])

    if n_halo or SIZE > 1:
        nbrs = np.unique(halo_info[:, 3]).astype(int)
        recv_bufs = {}
        reqs = []
        for j in nbrs:
            rows = halo_info[halo_info[:, 3] == j]
            sendbuf = np.ascontiguousarray(agg[rows[:, 0]])
            reqs.append(COMM.Isend(sendbuf, dest=j, tag=7))
            recv_bufs[j] = (rows[:, 1], np.empty_like(sendbuf))
        for j in nbrs:
            COMM.Recv(recv_bufs[j][1], source=j, tag=7)
        MPI.Request.Waitall(reqs)
        for j in nbrs:
            slots, buf = recv_bufs[j]
            agg[slots] = buf
        np.add.at(agg, halo_info[:, 0], agg[halo_info[:, 1]])

    row = np.searchsorted(uniq_gids_ref, gid_red)
    check(
        np.array_equal(uniq_gids_ref[row], gid_red),
        "all local gids exist in reference",
    )
    err = np.abs(agg[:n_local] - y_ref[row]).max()
    check(
        err < 1e-11,
        f"halo-consistent aggregation matches serial reference "
        f"(max err {err:.3e})",
    )

    # -- check 5: field routing
    u = rp.read_field(
        lambda s: os.path.join(
            args.src, f"u_field_rank_{s}_size_{src.src_size}.bin"
        ),
        ncols=3,
    )
    check(
        np.allclose(u, field_fn(pos), atol=1e-14),
        "field files route consistently with the graph",
    )

    # -- check 6: degree-weighted global statistics
    xw = x / node_degree[:, None]
    local_sum = xw.sum(axis=0)
    glob_sum = np.zeros(3)
    COMM.Allreduce(local_sum, glob_sum, op=MPI.SUM)
    mean = glob_sum / n_unique_ref
    mean_ref = node_feature(uniq_gids_ref).mean(axis=0)
    check(
        np.allclose(mean, mean_ref, atol=1e-12),
        "degree-weighted global mean matches unique-gid mean",
    )

    if RANK == 0:
        print(f"ALL CHECKS PASSED (M={SIZE}, method={args.method})")


if __name__ == "__main__":
    main()
