"""Small mpi4py helpers used across the repartition package."""

import numpy as np
from mpi4py import MPI

_MPI_TYPE = {
    np.dtype(np.float64): MPI.DOUBLE,
    np.dtype(np.int64): MPI.INT64_T,
    np.dtype(np.int32): MPI.INT32_T,
    np.dtype(np.int8): MPI.INT8_T,
}


def displs(counts):
    d = np.zeros_like(counts)
    np.cumsum(counts[:-1], out=d[1:])
    return d


def alltoallv(send, scounts, rcounts, comm):
    """Alltoallv of a flat 1D numpy array with per-rank item counts."""
    send = np.ascontiguousarray(send)
    scounts = np.asarray(scounts, dtype=np.int64)
    rcounts = np.asarray(rcounts, dtype=np.int64)
    recv = np.empty(int(rcounts.sum()), dtype=send.dtype)
    t = _MPI_TYPE[send.dtype]
    comm.Alltoallv(
        [send, scounts, displs(scounts), t],
        [recv, rcounts, displs(rcounts), t],
    )
    return recv


def alltoallv_2d(send, scounts_rows, rcounts_rows, comm):
    """Alltoallv of a C-contiguous (n, k) array with per-rank row counts."""
    k = send.shape[1]
    flat = alltoallv(
        np.ascontiguousarray(send).reshape(-1),
        np.asarray(scounts_rows, dtype=np.int64) * k,
        np.asarray(rcounts_rows, dtype=np.int64) * k,
        comm,
    )
    return flat.reshape(-1, k)
