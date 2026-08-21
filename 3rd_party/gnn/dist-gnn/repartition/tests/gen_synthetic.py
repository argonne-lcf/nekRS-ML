"""Generate a synthetic gnn_outputs directory (serial, no MPI, no nekRS).

Builds a conforming nex x ney x nez hex box mesh on [0,1]^3 with a GLL
lattice of order p per element, assigns exact integer global ids from the
global conforming lattice (so coincident nodes share a gid with no floating
point matching), block-partitions elements to a chosen source size S, and
writes the per-rank binary files in the exact formats of the nekRS gnn
plugin (gnn.cpp:174-213). Masks and per-rank edge sets are computed with an
independent implementation (global knowledge, serial) so they cross-check
the MPI rendezvous logic in repartition.rebuild.

Also writes:
- a synthetic node field u(pos) per source rank, fieldOffset-padded like
  nekRS trajectory files
- reference.npz: canonical global reduced edge set (gid pairs), unique gid
  count, and reference one-round aggregation y_ref per gid used by
  test_consistency.py
"""

import argparse
import os

import numpy as np
from numpy.polynomial import legendre


def gll_points(p):
    c = np.zeros(p + 1)
    c[p] = 1.0
    interior = legendre.legroots(legendre.legder(c))
    return np.concatenate([[-1.0], interior, [1.0]])


def field_fn(pos):
    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
    return np.stack(
        [
            np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y),
            y * y + 0.5 * z,
            np.cos(np.pi * z) + x,
        ],
        axis=1,
    )


def node_feature(gids):
    """Deterministic per-gid feature used in the aggregation reference."""
    g = gids.astype(np.float64)
    return np.stack(
        [np.sin(0.01 * g), np.cos(0.02 * g) + 0.1, 0.001 * (g % 97)],
        axis=1,
    )


def stencil_template(p):
    """6-point GLL lattice stencil inside one element, both directions."""
    nq = p + 1
    ii, jj, kk = np.meshgrid(
        np.arange(nq), np.arange(nq), np.arange(nq), indexing="ij"
    )
    lin = (ii + jj * nq + kk * nq * nq).reshape(-1)
    edges = []
    for di, dj, dk in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:
        src_ok = (
            (ii + di < nq) & (jj + dj < nq) & (kk + dk < nq)
        ).reshape(-1)
        nbr = (
            (ii + di) + (jj + dj) * nq + (kk + dk) * nq * nq
        ).reshape(-1)
        a = lin[src_ok]
        b = nbr[src_ok]
        edges.append(np.stack([a, b], axis=1))
        edges.append(np.stack([b, a], axis=1))
    return np.unique(np.concatenate(edges, axis=0), axis=0).astype(np.int64)


def build_mesh(nex, ney, nez, p):
    nq = p + 1
    np_pts = nq**3
    t = (gll_points(p) + 1.0) / 2.0  # [0, 1] within element

    # local linear index convention n = i + j*nq + k*nq*nq (matches template)
    lin = np.arange(np_pts)
    la = lin % nq
    lb = (lin // nq) % nq
    lc = lin // (nq * nq)

    sx, sy = nex * p + 1, ney * p + 1
    ne = nex * ney * nez
    pos = np.empty((ne * np_pts, 3), dtype=np.float64)
    gids = np.empty(ne * np_pts, dtype=np.int64)
    for e in range(ne):
        ix = e % nex
        iy = (e // nex) % ney
        iz = e // (nex * ney)
        gx = ix * p + la
        gy = iy * p + lb
        gz = iz * p + lc
        sl = slice(e * np_pts, (e + 1) * np_pts)
        gids[sl] = 1 + gx + gy * sx + gz * (sx * sy)
        pos[sl, 0] = (ix + t[la]) / nex
        pos[sl, 1] = (iy + t[lb]) / ney
        pos[sl, 2] = (iz + t[lc]) / nez
    return pos, gids, np_pts


def rank_arrays(pos, gids, template, np_pts, el_range, rank_of_gid):
    """Per-source-rank arrays with independent (serial) mask/edge logic."""
    e0, e1 = el_range
    n = (e1 - e0) * np_pts
    p_r = pos[e0 * np_pts : e1 * np_pts]
    g_r = gids[e0 * np_pts : e1 * np_pts]

    uniq, first_idx, inverse = np.unique(
        g_r, return_index=True, return_inverse=True
    )
    rep = first_idx[inverse]
    # shared with another rank?
    shared_uniq = np.array(
        [len(rank_of_gid[g]) > 1 for g in uniq], dtype=bool
    )
    shared = shared_uniq[inverse]
    is_rep = np.arange(n) == rep
    local_mask = (is_rep & ~shared).astype(np.int32)
    halo_mask = (is_rep & shared).astype(np.int32)

    offs = np.arange(e1 - e0, dtype=np.int64) * np_pts
    ei = (template[None, :, :] + offs[:, None, None]).reshape(-1, 2)
    a, b = rep[ei[:, 0]], rep[ei[:, 1]]
    keep = a != b
    key = np.unique(a[keep] * n + b[keep])
    ei_out = np.stack([key // n, key % n], axis=1).astype(np.int32)
    return p_r, g_r, ei_out, local_mask, halo_mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--nex", type=int, default=4)
    ap.add_argument("--ney", type=int, default=3)
    ap.add_argument("--nez", type=int, default=2)
    ap.add_argument("--poly", type=int, default=3)
    ap.add_argument("--src-size", type=int, default=4)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    pos, gids, np_pts = build_mesh(args.nex, args.ney, args.nez, args.poly)
    template = stencil_template(args.poly)
    ne = args.nex * args.ney * args.nez
    s = args.src_size
    bounds = [(r * ne // s, (r + 1) * ne // s) for r in range(s)]

    # global gid -> set of source ranks holding it
    rank_of_gid = {}
    for r, (e0, e1) in enumerate(bounds):
        for g in np.unique(gids[e0 * np_pts : e1 * np_pts]):
            rank_of_gid.setdefault(int(g), set()).add(r)

    for r, (e0, e1) in enumerate(bounds):
        p_r, g_r, ei_r, lm, hm = rank_arrays(
            pos, gids, template, np_pts, (e0, e1), rank_of_gid
        )
        sfx = f"_rank_{r}_size_{s}.bin"
        p_r.tofile(os.path.join(args.out, "pos_node" + sfx))
        g_r.reshape(-1, 1).tofile(os.path.join(args.out, "global_ids" + sfx))
        ei_r.tofile(os.path.join(args.out, "edge_index" + sfx))
        lm.tofile(os.path.join(args.out, "local_unique_mask" + sfx))
        hm.tofile(os.path.join(args.out, "halo_unique_mask" + sfx))

        # padded field file (emulates fieldOffset alignment)
        n_r = (e1 - e0) * np_pts
        pad = ((n_r + 31) // 32) * 32 - n_r
        u = field_fn(p_r)
        u_padded = np.concatenate(
            [u, np.zeros((pad, 3), dtype=np.float64)], axis=0
        )
        u_padded.tofile(
            os.path.join(args.out, f"u_field_rank_{r}_size_{s}.bin")
        )

    with open(os.path.join(args.out, f"Np_rank_0_size_{s}"), "w") as f:
        f.write(f"{np_pts}\n")

    # ---- global references -------------------------------------------
    # canonical reduced edge set as sorted gid pairs
    offs = np.arange(ne, dtype=np.int64) * np_pts
    ei_all = (template[None, :, :] + offs[:, None, None]).reshape(-1, 2)
    ga, gb = gids[ei_all[:, 0]], gids[ei_all[:, 1]]
    keep = ga != gb
    lo = np.minimum(ga[keep], gb[keep])
    hi = np.maximum(ga[keep], gb[keep])
    gmax = int(gids.max()) + 1
    ref_edges = np.unique(lo * gmax + hi)
    ref_pairs = np.stack([ref_edges // gmax, ref_edges % gmax], axis=1)

    # one-round aggregation reference: y[g] = sum over nbr gids of x[nbr]
    uniq_gids = np.unique(gids)
    x = node_feature(uniq_gids)
    gid_to_row = {int(g): i for i, g in enumerate(uniq_gids)}
    y = np.zeros_like(x)
    src_rows = np.array([gid_to_row[int(g)] for g in ref_pairs[:, 0]])
    dst_rows = np.array([gid_to_row[int(g)] for g in ref_pairs[:, 1]])
    np.add.at(y, dst_rows, x[src_rows])
    np.add.at(y, src_rows, x[dst_rows])

    np.savez(
        os.path.join(args.out, "reference.npz"),
        ref_pairs=ref_pairs,
        uniq_gids=uniq_gids,
        y_ref=y,
        n_unique=len(uniq_gids),
        Np=np_pts,
        n_elements=ne,
    )
    print(
        f"synthetic mesh: {ne} elements, Np={np_pts}, "
        f"{len(uniq_gids)} unique nodes, {len(ref_pairs)} reduced edges, "
        f"source size {s} -> {args.out}"
    )


if __name__ == "__main__":
    main()
