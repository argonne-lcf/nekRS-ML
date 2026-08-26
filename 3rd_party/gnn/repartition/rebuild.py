"""Regenerate the per-rank graph arrays for the current partition.

Produces exactly what the nekRS gnn plugin would have written had the
simulation run on this communicator (up to the choice of edge duplicates,
which downstream reduction makes irrelevant):

- edge_index: the intra-element stencil template tiled over local elements,
  with every endpoint mapped to its on-rank representative copy (min local
  index among same-gid copies). This is connectivity-equivalent to the
  plugin's coincident-copy neighbor-union augmentation after graph
  reduction, because reduction keeps exactly the representatives.
- local_unique_mask / halo_unique_mask: per gnn.cpp:358-773 semantics.
  local: first on-rank copy of a gid entirely on this rank (gid==0 nodes are
  always kept). halo: first on-rank copy of a gid also present on >=1 other
  rank (every sharing rank marks one copy).

Cross-rank sharing is detected with a rendezvous hash over gids (the same
scheme gslib's gs_setup uses: home rank = gid % size), replacing the
ogsHostGatherScatter min/max-rank + ogsGsUnique calls.
"""

import numpy as np

from .mpiutil import alltoallv


def _shared_flags(uniq_gids, comm):
    """For each locally-unique positive gid, is it present on another rank?

    Collective. Every rank must pass its full set of locally-unique gids so
    the rendezvous count equals the number of ranks holding each gid.
    """
    size = comm.Get_size()
    home = uniq_gids % size
    order = np.argsort(home, kind="stable")
    send = uniq_gids[order]
    scounts = np.bincount(home, minlength=size).astype(np.int64)
    rcounts = np.empty(size, dtype=np.int64)
    comm.Alltoall(scounts, rcounts)

    recv = alltoallv(send, scounts, rcounts, comm)
    _, inv, cnt = np.unique(recv, return_inverse=True, return_counts=True)
    flags = (cnt[inv] > 1).astype(np.int8)

    back = alltoallv(flags, rcounts, scounts, comm)
    shared = np.empty(uniq_gids.shape[0], dtype=bool)
    shared[order] = back.astype(bool)
    return shared


def rebuild_graph_arrays(elems, template_edges, comm):
    """Return the five trainer arrays for the current partition.

    template_edges: (Et, 2) int64, intra-element edges in [0, Np).
    Returns dict with pos (N,3) f8, global_ids (N,1) i8, edge_index (E,2) i4
    records (matching the on-disk .bin layout), local_unique_mask (N,) i4,
    halo_unique_mask (N,) i4.
    """
    np_pts = elems.Np
    ne = elems.n_elements
    n = elems.n_nodes
    gids = elems.gids.reshape(-1).astype(np.int64)

    # --- representatives: first on-rank copy (min local index) per gid.
    # gid==0 marks never-coincident nodes in the plugin's convention; give
    # them unique synthetic negative ids so each is its own representative.
    work = gids.copy()
    zeros = np.nonzero(work == 0)[0]
    if zeros.size:
        work[zeros] = -1 - np.arange(zeros.size, dtype=np.int64)
    uniq, first_idx, inverse = np.unique(
        work, return_index=True, return_inverse=True
    )
    rep = first_idx[inverse]

    # --- cross-rank sharing per unique gid (positive gids only)
    pos_sel = uniq > 0
    shared_uniq = np.zeros(uniq.shape[0], dtype=bool)
    shared_uniq[pos_sel] = _shared_flags(uniq[pos_sel].astype(np.int64), comm)
    shared_node = shared_uniq[inverse]

    is_rep = np.arange(n, dtype=np.int64) == rep
    local_unique_mask = (is_rep & ~shared_node).astype(np.int32)
    halo_unique_mask = (is_rep & shared_node).astype(np.int32)

    # --- edges: tile template over elements, map through representatives
    template_edges = np.asarray(template_edges, dtype=np.int64)
    offsets = np.arange(ne, dtype=np.int64) * np_pts
    ei = (template_edges[None, :, :] + offsets[:, None, None]).reshape(-1, 2)
    a = rep[ei[:, 0]]
    b = rep[ei[:, 1]]
    keep = a != b
    key = np.unique(a[keep] * n + b[keep])
    edge_index = np.stack([key // n, key % n], axis=1)

    return {
        "pos": np.ascontiguousarray(elems.pos, dtype=np.float64),
        "global_ids": gids.reshape(-1, 1),
        "edge_index": edge_index.astype(np.int32),
        "local_unique_mask": local_unique_mask,
        "halo_unique_mask": halo_unique_mask,
    }
