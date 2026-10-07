"""Differentiable neighbor-only halo exchange.

``torch.distributed.nn.all_to_all`` is the obvious way to swap halo nodes, but it
is a dense collective: the tensor lists it takes are indexed by rank, so every
rank posts a transfer to all ``world_size - 1`` peers even though a halo only ever
touches a handful of them. The empty transfers are not free -- each one still
costs a descriptor, a match-list entry and a completion -- so the exchange grows
linearly in the number of ranks while the data being moved stays constant.

Measured on Aurora with the production mesh and neighbor topology, one exchange
costs 4.07 ms at 192 GPUs and 74.09 ms at 24576 GPUs, against a steady 8 real
neighbors at every scale. With 8 message passing layers that is most of a second
per step spent on messages of length zero.

PyTorch's NCCL backend already sidesteps this: it lowers the collective to an
isend/irecv loop that skips the zero-length peers, so on CUDA the exchange is
effectively neighbor-only. oneCCL implements the same collective as a true
alltoallv and gets no such treatment. Rather than depend on which vendor happens
to optimize which collective, this module does the neighbor exchange explicitly,
on top of point-to-point primitives that every backend supports.

``dist.batch_isend_irecv`` handles the vendor differences for us: it coalesces the
batch where the backend supports it and otherwise issues the same sends and
receives one at a time, so the same code is correct on XCCL, NCCL, RCCL, Gloo and
MPI.
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
    not by rank, and must already be the right size -- nothing is allocated,
    reordered or copied here, so a caller timing this call is timing the
    transport and nothing else.

    Not differentiable; see :func:`neighbor_exchange` for the autograd version.

    Receives are posted before sends so that the batch is already drainable when
    the sends land. Empty buffers are skipped rather than posted, which is the
    entire point of this module.
    """
    me = dist.get_rank(group=group)

    # int() because neighbor lists arrive as numpy int64 (np.unique over
    # halo_info), which P2POp will not accept as a rank.
    ops = []
    for k, (buf, peer) in enumerate(zip(recv_list, neighbors)):
        peer = int(peer)
        if not buf.numel():
            continue
        if peer == me:
            # A rank can list itself (a single-rank job, or a partition whose
            # halo closes on its own ranks). Posting a send to oneself is not
            # reliably supported across backends and would at best be a slow
            # way to spell copy_, so short-circuit it.
            buf.copy_(send_list[k])
            continue
        ops.append(dist.P2POp(dist.irecv, buf, peer, group))
    for k, (buf, peer) in enumerate(zip(send_list, neighbors)):
        peer = int(peer)
        if buf.numel() and peer != me:
            ops.append(dist.P2POp(dist.isend, buf, peer, group))

    if not ops:
        # A rank with no neighbors still has to not hang here. batch_isend_irecv
        # rejects an empty list, and there is nothing to wait on anyway.
        return

    for req in dist.batch_isend_irecv(ops):
        req.wait()


class _NeighborExchange(torch.autograd.Function):
    """Autograd wrapper around :func:`neighbor_exchange_`.

    The exchange just copies rows from one rank to another, so as a linear
    operator it is a permutation and its adjoint is the same permutation run
    backwards: the gradient of a send is a receive from that same peer, and vice
    versa. That is the same duality ``torch.distributed.nn``'s ``_AlltoAll`` uses,
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
    the matching send never comes. That hangs with no error, which at a few
    thousand ranks is a miserable thing to debug, so pay for one collective at
    setup to rule it out. This is called once outside the step loop, not per
    exchange.

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
