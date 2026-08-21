"""Element partitioning strategies.

Each strategy maps every locally-held element to a destination rank in
[0, comm.size). Strategies must be deterministic for a given input so that
repeated runs (graph vs. field data passes) produce identical layouts.

- "block": contiguous equal split of the global element ordinal space. Cheap
  and, when the source ordering came from a good partitioner (parRSB), it
  preserves locality well for integer refactorings of the rank count.
- "rcb": recursive coordinate bisection on element centroids. Centroids are
  gathered to rank 0 (fine for preprocessing-scale meshes; use "parrsb" for
  very large meshes).
- "parrsb": nek5000's parRSB (recursive spectral bisection on the element
  connectivity graph induced by shared corner-vertex gids). Fully
  distributed — no rank-0 gather — and connectivity-aware, so it keeps
  periodic neighbors together where RCB always cuts them. Needs the ctypes
  shim built by build_parrsb_shim.sh (see parrsb.py).
"""

import numpy as np

from .mpiutil import alltoallv, displs


def partition_block(n_total, ordinals, size):
    return (ordinals.astype(np.int64) * size // n_total).astype(np.int64)


def _rcb_serial(centroids, ordinals, size):
    """Serial RCB: returns dest rank per row of centroids."""
    dest = np.empty(centroids.shape[0], dtype=np.int64)

    # stack of (index array, rank range [r0, r1))
    stack = [(np.arange(centroids.shape[0]), 0, size)]
    while stack:
        idx, r0, r1 = stack.pop()
        nranks = r1 - r0
        if nranks == 1:
            dest[idx] = r0
            continue
        c = centroids[idx]
        axis = int(np.argmax(c.max(axis=0) - c.min(axis=0)))
        rmid = r0 + nranks // 2
        nleft = idx.shape[0] * (rmid - r0) // nranks
        # deterministic total order: coordinate, then ordinal tie-break
        order = np.lexsort((ordinals[idx], c[:, axis]))
        stack.append((idx[order[:nleft]], r0, rmid))
        stack.append((idx[order[nleft:]], rmid, r1))
    return dest


def partition_rcb(centroids, ordinals, comm):
    """Gather-based parallel RCB (rank 0 computes, results scattered)."""
    size = comm.Get_size()
    rank = comm.Get_rank()

    n_local = np.array([centroids.shape[0]], dtype=np.int64)
    counts = np.empty(size, dtype=np.int64)
    comm.Allgather(n_local, counts)

    scounts = np.zeros(size, dtype=np.int64)
    scounts[0] = centroids.shape[0]
    rcounts = counts if rank == 0 else np.zeros(size, dtype=np.int64)

    cent_all = alltoallv(
        np.ascontiguousarray(centroids).reshape(-1),
        scounts * 3,
        rcounts * 3,
        comm,
    ).reshape(-1, 3)
    ords_all = alltoallv(ordinals, scounts, rcounts, comm)

    if rank == 0:
        dest_all = _rcb_serial(cent_all, ords_all, size)
    else:
        dest_all = np.empty(0, dtype=np.int64)

    # scatter back in the same per-rank segments
    return alltoallv(dest_all, rcounts, scounts, comm)


def corner_lattice_indices(np_pts):
    """Within-element node indices of the 8 hex corners (np_pts = Np).

    Nodes are lattice-ordered i + j*nq + k*nq^2 (i/x fastest, the gnn plugin
    convention); corners follow nekRS's hex vertex order (meshLoadReference-
    NodesHex3D): (-,-,-), (+,-,-), (+,+,-), (-,+,-), then the k=nq-1 plane.
    """
    nq = round(np_pts ** (1.0 / 3.0))
    if nq * nq * nq != np_pts:
        raise ValueError(f"Np={np_pts} is not a cube")
    lo, hi = 0, nq - 1
    return np.array(
        [
            lo + nq * lo + nq * nq * lo,
            hi + nq * lo + nq * nq * lo,
            hi + nq * hi + nq * nq * lo,
            lo + nq * hi + nq * nq * lo,
            lo + nq * lo + nq * nq * hi,
            hi + nq * lo + nq * nq * hi,
            hi + nq * hi + nq * nq * hi,
            lo + nq * hi + nq * nq * hi,
        ],
        dtype=np.int64,
    )


def element_corners(elems):
    """(Ne, 8) corner gids and (Ne, 8, 3) corner coords for parRSB.

    gid==0 marks never-shared (single-gather) nodes in nekRS outputs; parRSB
    treats vertex ids purely as equality labels, so aliasing all of them to
    one vertex would glue unrelated elements together. Give each instance a
    unique negative label derived from the partition-independent element
    ordinal (deterministic across runs and rank counts).
    """
    corners = corner_lattice_indices(elems.Np)
    gids = elems.gids.reshape(elems.n_elements, elems.Np)[:, corners].copy()
    zero = gids == 0
    if zero.any():
        ords = np.broadcast_to(elems.ordinals[:, None], gids.shape)
        cidx = np.broadcast_to(
            np.arange(8, dtype=np.int64)[None, :], gids.shape
        )
        gids[zero] = -(ords[zero] * 8 + cidx[zero] + 1)
    xyz = elems.pos.reshape(elems.n_elements, elems.Np, 3)[:, corners, :]
    return gids, xyz


def partition_elements(elems, comm, method="rcb"):
    size = comm.Get_size()
    if size == 1:
        return np.zeros(elems.n_elements, dtype=np.int64)
    if method == "block":
        n_total = comm.allreduce(elems.n_elements)
        return partition_block(n_total, elems.ordinals, size)
    if method == "rcb":
        return partition_rcb(elems.centroids(), elems.ordinals, comm)
    if method == "parrsb":
        from .parrsb import partition_parrsb

        gids, xyz = element_corners(elems)
        return partition_parrsb(gids, xyz, comm)
    raise ValueError(f"unknown partition method: {method}")


__all__ = [
    "corner_lattice_indices",
    "displs",
    "element_corners",
    "partition_block",
    "partition_elements",
    "partition_rcb",
]
