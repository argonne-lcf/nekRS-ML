"""Intra-element GLL stencil edge template (matches gnn_connectivity.cpp)."""

import numpy as np


def stencil_template(nq):
    """6-point lattice stencil inside one element of nq^3 GLL points, both
    directions, local linear index convention n = i + j*nq + k*nq*nq."""
    lin = np.arange(nq**3)
    i = lin % nq
    j = (lin // nq) % nq
    k = lin // (nq * nq)
    edges = []
    for di, dj, dk in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:
        ok = (i + di < nq) & (j + dj < nq) & (k + dk < nq)
        nbr = (i + di) + (j + dj) * nq + (k + dk) * nq * nq
        a, b = lin[ok], nbr[ok]
        edges.append(np.stack([a, b], axis=1))
        edges.append(np.stack([b, a], axis=1))
    return np.unique(np.concatenate(edges, axis=0), axis=0).astype(np.int64)
