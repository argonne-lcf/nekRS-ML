"""Move whole elements (and any node-level data) to their destination ranks."""

from dataclasses import dataclass

import numpy as np

from .element_data import LocalElements
from .mpiutil import alltoallv, alltoallv_2d


@dataclass
class Routing:
    """Reusable element routing between the source-order local layout and
    the post-redistribution layout. Apply to any (Ne_local*Np, k) array laid
    out in the same source element order to route it consistently."""

    Np: int
    send_order: np.ndarray  # (Ne_local,) permutation grouping by dest rank
    scounts: np.ndarray  # (size,) elements sent per rank
    rcounts: np.ndarray  # (size,) elements received per rank
    recv_order: np.ndarray  # (Ne_new,) permutation sorting by ordinal

    def _node_perm(self, order):
        np_ = self.Np
        return (
            order[:, None] * np_ + np.arange(np_, dtype=np.int64)[None, :]
        ).reshape(-1)

    def route_node_array(self, arr, comm):
        """Route a (Ne_local*Np, k) float64/int64 array to the new layout."""
        arr = np.ascontiguousarray(arr)
        squeeze = arr.ndim == 1
        if squeeze:
            arr = arr.reshape(-1, 1)
        send = arr[self._node_perm(self.send_order)]
        recv = alltoallv_2d(
            send, self.scounts * self.Np, self.rcounts * self.Np, comm
        )
        recv = recv[self._node_perm(self.recv_order)]
        return recv.reshape(-1) if squeeze else recv


def redistribute_elements(elems, dest, comm):
    """Ship elements to dest ranks; receiver orders them by global ordinal.

    Returns (new LocalElements, Routing).
    """
    size = comm.Get_size()
    dest = np.asarray(dest, dtype=np.int64)

    send_order = np.argsort(dest, kind="stable")
    scounts = np.bincount(dest, minlength=size).astype(np.int64)
    rcounts = np.empty(size, dtype=np.int64)
    comm.Alltoall(scounts, rcounts)

    ords_recv = alltoallv(elems.ordinals[send_order], scounts, rcounts, comm)
    recv_order = np.argsort(ords_recv, kind="stable")

    routing = Routing(
        Np=elems.Np,
        send_order=send_order,
        scounts=scounts,
        rcounts=rcounts,
        recv_order=recv_order,
    )

    new = LocalElements(
        Np=elems.Np,
        ordinals=ords_recv[recv_order],
        pos=routing.route_node_array(elems.pos, comm),
        gids=routing.route_node_array(elems.gids, comm),
        fields={
            k: routing.route_node_array(v, comm)
            for k, v in elems.fields.items()
        },
    )
    return new, routing
