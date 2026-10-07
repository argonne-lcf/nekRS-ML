"""
Differentiable neighbor-only halo exchange implemented with dist.batch_isend_irecv.
"""

from typing import List, Sequence

import torch
import torch.distributed as dist


def neighbor_exchange_(
    recv_list: List[torch.Tensor],
    send_list: List[torch.Tensor],
    neighbors: Sequence[int],
    group=None,
) -> None:
    """Swap one tensor with each neighbor, in place into ``recv_list``.

    ``send_list[k]`` goes to ``neighbors[k]`` and ``recv_list[k]`` receives what
    ``neighbors[k]`` sent. Both lists are indexed by position in ``neighbors``,
    not by rank, and must already be the right size.

    Not differentiable; see :func:`neighbor_exchange` for the autograd version.

    Receives are posted before sends so that the batch is already drainable when
    the sends land. Empty buffers are skipped.
    """
    me = dist.get_rank(group=group)

    ops = []
    for k, (buf, peer) in enumerate(zip(recv_list, neighbors)):
        peer = int(peer)
        if not buf.numel():
            continue
        if peer == me:
            buf.copy_(send_list[k])
            continue
        ops.append(dist.P2POp(dist.irecv, buf, peer, group))
    for k, (buf, peer) in enumerate(zip(send_list, neighbors)):
        peer = int(peer)
        if buf.numel() and peer != me:
            ops.append(dist.P2POp(dist.isend, buf, peer, group))

    if not ops:
        return

    for req in dist.batch_isend_irecv(ops):
        req.wait()


class _NeighborExchange(torch.autograd.Function):
    """Autograd wrapper around :func:`neighbor_exchange_`.

    The exchange just copies rows from one rank to another, so as a linear
    operator it is a permutation and its adjoint is the same permutation run
    backwards: the gradient of a send is a receive from that same peer, and vice
    versa. That is the same duality torch.distributed.nn._AlltoAll uses,
    and it is why the backward below is another call to the same helper with the
    roles swapped.

    Send and receive counts per neighbor are equal for a halo (the trainer's
    build_masks enforces it), so the backward can size its buffers from the
    forward's send shapes without any extra negotiation.
    """

    @staticmethod
    def forward(ctx, group, neighbors, *send_tensors):
        ctx.group = group
        ctx.neighbors = neighbors

        send_tensors = tuple(t.contiguous() for t in send_tensors)
        ctx.send_shapes = [t.shape for t in send_tensors]

        # Receive buffers are allocated fresh on every call rather than reused
        # from a pool: autograd hands these straight out as graph outputs, and a
        # pooled buffer would be overwritten by the next layer's exchange before
        # backward ever got to read it.
        #
        # empty_like is only right because a halo sends a peer exactly as many
        # rows as it receives back -- build_masks enforces that, and
        # assert_symmetric_neighbors checks the pairing. If that ever stops
        # holding, the receives would silently truncate or overrun rather than
        # fail, so the sizes are worth stating as a contract here.
        recv_tensors = [torch.empty_like(t) for t in send_tensors]

        neighbor_exchange_(recv_tensors, list(send_tensors), neighbors, group)

        return tuple(recv_tensors)

    @staticmethod
    def backward(ctx, *grad_outputs):
        grads = [g.contiguous() for g in grad_outputs]
        back = [
            torch.empty(shape, dtype=g.dtype, device=g.device)
            for shape, g in zip(ctx.send_shapes, grads)
        ]

        neighbor_exchange_(back, grads, ctx.neighbors, ctx.group)

        return (None, None) + tuple(back)


def neighbor_exchange(
    send_tensors: Sequence[torch.Tensor],
    neighbors: Sequence[int],
    group=None,
) -> tuple:
    """Exchange one tensor with each neighbor and return what they sent back.

    ``send_tensors[k]`` goes to ``neighbors[k]`` and the returned tuple's k-th
    entry is what ``neighbors[k]`` sent here. Differentiable, so this can sit in
    the middle of a message passing layer.
    """
    return _NeighborExchange.apply(group, tuple(neighbors), *send_tensors)


def assert_symmetric_neighbors(neighbors: Sequence[int], comm) -> None:
    """Check that every neighbor claims us back, before any exchange runs.

    A one-sided neighbor list deadlocks the exchange: our receive is posted but
    the matching send never comes. 

    ``comm`` is an mpi4py communicator -- the trainer already has one, and this
    runs during setup where the torch process group may not be the right place to
    put a diagnostic.
    """
    size = comm.Get_size()
    rank = comm.Get_rank()

    row = bytearray(size)
    for peer in neighbors:
        row[int(peer)] = 1

    # Column j of the gathered matrix is who claims rank j as a neighbor.
    claims = comm.alltoall([bytes(row[j : j + 1]) for j in range(size)])

    mine = {int(p) for p in neighbors}
    missing = [j for j, c in enumerate(claims) if c[0] and j not in mine]
    if missing:
        raise RuntimeError(
            f"[rank {rank}] asymmetric halo neighbors: rank(s) {missing} send to "
            f"this rank but are absent from its own neighbor list {list(neighbors)}. "
            "The halo exchange would deadlock; check the graph partitioning."
        )
