"""
Create halo swap info.
"""

import argparse
import numpy as np
from typing import Tuple
import time

import torch
from torch_geometric.data import Data
import torch_geometric.utils as utils

import mpi4py

mpi4py.rc.initialize = False
from mpi4py import MPI


# helper Cantor-pairing on two 1D int64 tensors
def cantor_pair(k1: torch.Tensor, k2: torch.Tensor) -> torch.Tensor:
    # (0.5*(k1+k2)*(k1+k2+1) + k2) exactly as before, but keep it isolated
    s = k1.to(torch.float64) + k2.to(torch.float64)
    return (0.5 * s * (s + 1) + k2.to(torch.float64)).to(torch.int64)


def make_reduced_graph(
    COMM: MPI.COMM_WORLD, RANK: int, SIZE: int
) -> Tuple[Data, Data, torch.Tensor]:
    path_to_pos_full = main_path + "pos_node_rank_%d_size_%d" % (RANK, SIZE)
    path_to_ei = main_path + "edge_index_rank_%d_size_%d" % (RANK, SIZE)
    path_to_glob_ids = main_path + "global_ids_rank_%d_size_%d" % (RANK, SIZE)
    path_to_unique = main_path + "local_unique_mask_rank_%d_size_%d" % (
        RANK,
        SIZE,
    )
    path_to_halo_ids = None
    if SIZE > 1:
        path_to_halo_ids = main_path + "halo_ids_rank_%d_size_%d" % (RANK, SIZE)
        path_to_unique_halo = main_path + "halo_unique_mask_rank_%d_size_%d" % (
            RANK,
            SIZE,
        )

    # ~~~~ Get positions and global node index
    # if args.LOG=='debug': print('[RANK %d]: Loading positions and global node index' %(RANK), flush=True)
    pos = np.fromfile(path_to_pos_full + ".bin", dtype=np.float64).reshape((
        -1,
        3,
    ))
    gli = np.fromfile(path_to_glob_ids + ".bin", dtype=np.int64).reshape((
        -1,
        1,
    ))

    # ~~~~ Back-out number of elements
    Ne = int(pos.shape[0] / Np)
    # if args.LOG=='debug': print('[RANK %d]: Number of elements is %d' %(RANK, Ne), flush=True)

    # ~~~~ Get edge index
    # if args.LOG=='debug': print('[RANK %d]: Loading edge index' %(RANK), flush=True)
    ei = np.fromfile(path_to_ei + ".bin", dtype=np.int32).reshape((-1, 2)).T
    ei = ei.astype(np.int64)

    # ~~~~ Get local unique mask
    # if args.LOG=='debug': print('[RANK %d]: Loading local unique mask' %(RANK), flush=True)
    local_unique_mask = np.fromfile(path_to_unique + ".bin", dtype=np.int32)

    # ~~~~ Get halo unique mask
    halo_unique_mask = np.array([])
    if SIZE > 1:
        halo_unique_mask = np.fromfile(
            path_to_unique_halo + ".bin", dtype=np.int32
        )
    COMM.Barrier()

    # ~~~~ Make graph:
    data = Data(
        x=torch.tensor(pos),
        edge_index=torch.tensor(ei),
        pos=torch.tensor(pos),
        global_ids=torch.tensor(gli.squeeze()),
        local_unique_mask=torch.tensor(local_unique_mask),
        halo_unique_mask=torch.tensor(halo_unique_mask),
    )
    data.edge_index = utils.remove_self_loops(data.edge_index)[0]
    data.edge_index = utils.coalesce(data.edge_index)
    data.edge_index = utils.to_undirected(data.edge_index)

    # ~~~~ Append list of graphs
    # graph_list.append(data)
    COMM.Barrier()
    if RANK == 0:
        print("Done making graph \n", flush=True)

    # ~~~~ Reduce size of graph
    # X: [First isolate local nodes]
    idx_local_unique = torch.nonzero(data.local_unique_mask).squeeze(-1)
    idx_halo_unique = torch.tensor([], dtype=idx_local_unique.dtype)
    if SIZE > 1:
        idx_halo_unique = torch.nonzero(data.halo_unique_mask).squeeze(-1)
    idx_keep = torch.cat((idx_local_unique, idx_halo_unique))

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # PYGEOM FUNCTION -- this gets the reduced edge_index
    num_nodes = data.x.shape[0]
    perm = idx_keep
    mask = perm.new_full((num_nodes,), -1)
    i = torch.arange(perm.size(0), dtype=torch.long, device=perm.device)
    mask[perm] = i

    row, col = data.edge_index
    row, col = mask[row], mask[col]
    mask = (row >= 0) & (col >= 0)
    row, col = row[mask], col[mask]
    edge_index_reduced = torch.stack([row, col], dim=0)
    edge_index_reduced = utils.coalesce(edge_index_reduced)
    edge_index_reduced = utils.to_undirected(edge_index_reduced)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    pos_reduced = data.pos[idx_keep]
    gid_reduced = data.global_ids[idx_keep]
    data_reduced = Data(
        x=pos_reduced,
        pos=pos_reduced,
        edge_index=edge_index_reduced,
        global_ids=gid_reduced,
    )
    n_not_halo = len(idx_local_unique)
    n_halo = len(idx_halo_unique)
    data_reduced.local_unique_mask = torch.zeros(
        n_not_halo + n_halo, dtype=torch.int64
    )
    data_reduced.local_unique_mask[:n_not_halo] = 1
    data_reduced.halo_unique_mask = torch.zeros(
        n_not_halo + n_halo, dtype=torch.int64
    )
    data_reduced.halo_unique_mask[n_not_halo:] = 1
    gid = data.global_ids
    zero_indices = torch.where(gid == 0)[0]
    consecutive_negatives = -1 * torch.arange(1, len(zero_indices) + 1)
    gid[zero_indices] = consecutive_negatives
    data.global_ids = gid
    data_reduced.global_ids = gid[idx_keep]
    if RANK == 0:
        print("Done making reduced graph \n", flush=True)
    return data, data_reduced, idx_keep

    # graph_reduced_list.append(data_reduced)


# ~~~~ Get the new halo_ids
def get_reduced_halo_ids(
    COMM: MPI.COMM_WORLD, RANK: int, SIZE: int, data_reduced: Data
) -> torch.Tensor:
    """Build this rank's halo triples [local_id, global_id, rank].
    """
    if SIZE == 1:
        return torch.zeros((0, 3), dtype=torch.int64)

    # What are the local ids of the halo nodes ?
    # (the reduced graph is laid out locals-first, then halos)
    n_local = int(data_reduced.local_unique_mask.sum().item())
    n_halo = int(data_reduced.halo_unique_mask.sum().item())
    idx_halo_unique = torch.arange(n_local, n_local + n_halo, dtype=torch.int64)

    # What are the corresponding global ids? The sign of the global id marks
    # ownership of a coincident node upstream, so canonicalize with abs().
    gid_halo_unique = torch.abs(data_reduced.global_ids[idx_halo_unique]).to(
        torch.int64
    )

    # What is the current rank?
    rank_array = torch.full_like(gid_halo_unique, RANK)

    # [local id, global id, rank]
    return torch.stack(
        (idx_halo_unique, gid_halo_unique, rank_array), dim=1
    )


def _alltoallv_rows(
    COMM: MPI.COMM_WORLD, SIZE: int, send_rows: np.ndarray, send_counts
) -> np.ndarray:
    """Alltoallv of a [M, W] int64 row block already grouped by destination.

    send_counts[i] is the number of *rows* destined for rank i. Only the
    O(SIZE) counts/displacement arrays scale with rank count here; the payload
    is O(local halo size).
    """
    width = int(send_rows.shape[1])
    send_rows = np.ascontiguousarray(send_rows, dtype=np.int64)

    scount = np.ascontiguousarray(send_counts, dtype=np.int32)
    rcount = np.empty(SIZE, dtype=np.int32)
    COMM.Alltoall([scount, MPI.INT], [rcount, MPI.INT])

    sdispl = np.zeros(SIZE, dtype=np.int32)
    rdispl = np.zeros(SIZE, dtype=np.int32)
    if SIZE > 1:
        sdispl[1:] = np.cumsum(scount[:-1])
        rdispl[1:] = np.cumsum(rcount[:-1])

    recv_rows = np.empty((int(rcount.sum()), width), dtype=np.int64)
    COMM.Alltoallv(
        [send_rows, (scount * width, sdispl * width), MPI.INT64_T],
        [recv_rows, (rcount * width, rdispl * width), MPI.INT64_T],
    )
    return recv_rows


def _group_by_dest(rows: np.ndarray, dest: np.ndarray, SIZE: int):
    """Stable-sort rows into destination-contiguous order.

    Returns (rows, counts).
    """
    order = np.argsort(dest, kind="stable")
    counts = np.bincount(dest, minlength=SIZE).astype(np.int32)
    return np.ascontiguousarray(rows[order]), counts


def _all_pairs_within_runs(starts: np.ndarray, counts: np.ndarray):
    """For each run of length c, all ordered (owner, neighbor) pairs.

    Fully vectorized: the previous implementation looped over every
    globally unique halo node in Python and built a meshgrid per node.
    """
    npairs = counts * (counts - 1)
    total = int(npairs.sum())
    if total == 0:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty

    run = np.repeat(np.arange(counts.shape[0], dtype=np.int64), npairs)
    # offset of each pair inside its own run's block of c*(c-1) pairs
    block_start = np.cumsum(npairs) - npairs
    off = np.arange(total, dtype=np.int64) - np.repeat(block_start, npairs)

    c = counts[run].astype(np.int64)
    i = off // (c - 1)
    j = off % (c - 1)
    j = j + (j >= i)  # skip the diagonal

    base = starts[run].astype(np.int64)
    return base + i, base + j


# Prepares the halo_info matrix for halo swap -- reference implementation.
# Kept as the replicated oracle the distributed version is checked against;
# it does its own Allgatherv so it still takes the local triples.
def get_halo_info(
    COMM: MPI.COMM_WORLD,
    RANK: int,
    SIZE: int,
    data_reduced: Data,
    halo_ids_local: torch.Tensor,
) -> list:
    if SIZE == 1:
        return [torch.zeros((0, 4), dtype=torch.int64)]

    # Replicate every rank's halo triples (this is the O(SIZE) step that
    # get_halo_info_fast avoids).
    halo_ids = halo_ids_local.to(torch.int64).reshape(-1, 3)
    shape_list = COMM.allgather(halo_ids.shape[0])
    halo_ids_full = torch.zeros(sum(shape_list), 3, dtype=torch.int64)
    count = [shape_list[i] * 3 for i in range(SIZE)]
    displ = [sum(count[:i]) for i in range(SIZE)]
    COMM.Allgatherv(
        [halo_ids.contiguous(), MPI.INT64_T],
        [halo_ids_full, count, displ, MPI.INT64_T],
    )

    n_nodes_glob = COMM.allgather(data_reduced.pos.shape[0])

    halo_ids_full[:, 1] = torch.abs(halo_ids_full[:, 1])
    gid = halo_ids_full[:, 1].numpy()
    rnk = halo_ids_full[:, 2].numpy()
    loc = halo_ids_full[:, 0].numpy()

    # canonical order: (global id, rank)
    order = np.lexsort((rnk, gid))
    gid, rnk, loc = gid[order], rnk[order], loc[order]

    _, starts, counts = np.unique(gid, return_index=True, return_counts=True)
    own_pos, nbr_pos = _all_pairs_within_runs(starts, counts)

    halo_info_glob = [torch.empty(0)] * SIZE
    owner_ranks = rnk[own_pos]
    for r in np.unique(owner_ranks):
        m = owner_ranks == r
        rows = np.zeros((int(m.sum()), 4), dtype=np.int64)
        rows[:, 0] = loc[own_pos[m]]
        rows[:, 1] = np.arange(rows.shape[0], dtype=np.int64) + n_nodes_glob[r]
        rows[:, 2] = gid[own_pos[m]]
        rows[:, 3] = rnk[nbr_pos[m]]
        halo_info_glob[int(r)] = torch.from_numpy(rows)
    return halo_info_glob


# Prepares the halo_info matrix for halo swap
def get_halo_info_fast(
    COMM: MPI.COMM_WORLD,
    RANK: int,
    SIZE: int,
    data_reduced: Data,
    halo_ids_local: torch.Tensor,
) -> list:
    """Build halo_info without replicating the global halo id table.

    Each global id is assigned a rendezvous rank (gid % SIZE). Every rank
    ships its halo triples to the rendezvous owner, which sees all copies of
    the ids it owns, forms the (owner, neighbor) pairs, and ships each row
    back to the rank that owns it. Per-rank cost is O(local halo size) in both
    memory and work, independent of SIZE.

    Returns a list of length SIZE in which
      * entry RANK is this rank's full halo_info,
        [local id, halo slot, global id, neighbor rank];
      * entry S, for each neighbor S, holds only S's rows whose neighbor is
        RANK -- cols 0 (S's local id), 2 (global id) and 3 (== RANK). That is
        exactly the slice get_edge_weights filters out of it. Col 1 (S's halo
        slot) is not reconstructed here because no caller reads it.
      * all other entries are empty.
    Rows are ordered by (global id, neighbor rank), so for any pair of ranks
    R and S the rows R holds for S and the rows S holds for R agree
    element-wise on global id, which is the ordering contract get_edge_weights
    asserts and the halo swap masks depend on.
    """
    if SIZE == 1:
        return [torch.zeros((0, 4), dtype=torch.int64)]

    triples = halo_ids_local.to(torch.int64).reshape(-1, 3).numpy()
    local_ids = triples[:, 0]
    gids = np.abs(triples[:, 1])

    # ---- Phase 1: ship [gid, rank, local id] to each gid's rendezvous rank
    dest = (gids % SIZE).astype(np.int64)
    out = np.empty((triples.shape[0], 3), dtype=np.int64)
    out[:, 0] = gids
    out[:, 1] = RANK
    out[:, 2] = local_ids
    out, counts = _group_by_dest(out, dest, SIZE)
    recv = _alltoallv_rows(COMM, SIZE, out, counts)

    # ---- Phase 2: at the rendezvous, pair up every copy of each owned gid
    if recv.shape[0]:
        g, r, loc = recv[:, 0], recv[:, 1], recv[:, 2]
        order = np.lexsort((r, g))  # canonical order: (global id, rank)
        g, r, loc = g[order], r[order], loc[order]
        _, starts, counts = np.unique(
            g, return_index=True, return_counts=True
        )
        own_pos, nbr_pos = _all_pairs_within_runs(starts, counts)
        # [owner local id, global id, neighbor rank, neighbor local id]
        pairs = np.empty((own_pos.shape[0], 4), dtype=np.int64)
        pairs[:, 0] = loc[own_pos]
        pairs[:, 1] = g[own_pos]
        pairs[:, 2] = r[nbr_pos]
        pairs[:, 3] = loc[nbr_pos]
        back_dest = r[own_pos].astype(np.int64)
    else:
        pairs = np.zeros((0, 4), dtype=np.int64)
        back_dest = np.zeros(0, dtype=np.int64)

    # ---- Phase 3: ship each row home to the rank that owns it
    pairs, counts = _group_by_dest(pairs, back_dest, SIZE)
    rows = _alltoallv_rows(COMM, SIZE, pairs, counts)

    # ---- Phase 4: restore canonical order and assign halo slots
    rows = rows[np.lexsort((rows[:, 2], rows[:, 1]))]
    n_rows = rows.shape[0]

    halo_info = np.zeros((n_rows, 4), dtype=np.int64)
    halo_info[:, 0] = rows[:, 0]
    halo_info[:, 1] = (
        np.arange(n_rows, dtype=np.int64) + data_reduced.pos.shape[0]
    )
    halo_info[:, 2] = rows[:, 1]
    halo_info[:, 3] = rows[:, 2]

    halo_info_glob = [torch.empty(0)] * SIZE
    halo_info_glob[RANK] = torch.from_numpy(halo_info)

    # The mirror rows each neighbor holds for us, needed by get_edge_weights.
    for s in np.unique(rows[:, 2]):
        m = rows[:, 2] == s
        mirror = np.zeros((int(m.sum()), 4), dtype=np.int64)
        mirror[:, 0] = rows[m, 3]  # neighbor's local id
        mirror[:, 2] = rows[m, 1]  # global id
        mirror[:, 3] = RANK
        halo_info_glob[int(s)] = torch.from_numpy(mirror)

    return halo_info_glob


# ~~~~ Get node degree from halo_info
def get_node_degree(
    COMM: MPI.COMM_WORLD,
    RANK: int,
    SIZE: int,
    data_reduced: Data,
    halo_info_rank: torch.Tensor,
) -> torch.Tensor:
    if SIZE == 1:
        return torch.ones(data_reduced.pos.shape[0])
    else:
        sample = data_reduced
        n_nodes_local = sample.pos.shape[0]
        node_degree = torch.ones(n_nodes_local)
        # halo_info_rank = halo_info_glob[RANK]
        unique_local_indices, counts = torch.unique(
            halo_info_rank[:, 0], return_counts=True
        )
        node_degree[unique_local_indices] += counts
    return node_degree


# ~~~~ Get edge weights to account for duplicate edges
def get_edge_weights(
    COMM: MPI.COMM_WORLD,
    RANK: int,
    SIZE: int,
    data_reduced: Data,
    halo_info_glob: list,
) -> torch.Tensor:
    if SIZE == 1:
        return torch.ones(data_reduced.edge_index.shape[1])
    else:
        # Collect edge_index shape
        edge_index_shape_list = COMM.allgather(data_reduced.edge_index.shape)

        # Collect global_id shape
        global_ids_shape_list = COMM.allgather(data_reduced.global_ids.shape)

        sample = data_reduced
        halo_info_rank = halo_info_glob[RANK]

        # Get neighboring procs for this rank
        neighboring_procs = np.unique(halo_info_rank[:, 3])
        # if args.LOG == 'debug':
        #    print(f'[RANK {RANK}]: Found {len(neighboring_procs)} neighboring procs.: {neighboring_procs}',flush=True)

        # Initialize edge weights
        num_edges_own = sample.edge_index.shape[1]
        edge_weights = torch.ones(num_edges_own)

        # Send/receive the edge index
        for j in neighboring_procs:
            COMM.Isend([data_reduced.edge_index, MPI.INT], dest=j)
        edge_index_nei_list = []
        for j in neighboring_procs:
            tmp = torch.zeros(edge_index_shape_list[j], dtype=torch.int64)
            COMM.Recv([tmp, MPI.INT], source=j)
            edge_index_nei_list.append(tmp)
        COMM.Barrier()
        # if RANK == 0: print('Communicated the edge_index arrays', flush=True)

        # Send/receive the global ids
        for j in neighboring_procs:
            COMM.Isend([data_reduced.global_ids, MPI.INT], dest=j)
        global_ids_nei_list = []
        for j in neighboring_procs:
            tmp = torch.zeros(global_ids_shape_list[j], dtype=torch.int64)
            COMM.Recv([tmp, MPI.INT], source=j)
            global_ids_nei_list.append(tmp)
        COMM.Barrier()
        # if RANK == 0: print('Communicated the global_ids arrays', flush=True)

        for i, rank_nei in enumerate(neighboring_procs):
            # extract only the halo rows for this neighbor
            halo_own = halo_info_rank[halo_info_rank[:, 3] == rank_nei]
            halo_nei = halo_info_glob[rank_nei][
                halo_info_glob[rank_nei][:, 3] == RANK
            ]

            # sanity check ordering
            assert torch.equal(halo_own[:, 2], halo_nei[:, 2]), (
                "misordered halos"
            )

            # pick out just the edges that touch our out-going halo nodes
            edge_idx = data_reduced.edge_index
            local_own = halo_own[:, 0]
            mask_own = torch.isin(edge_idx[1], local_own)
            edge_own = edge_idx[:, mask_own]

            # and the corresponding ones from the neighbor
            nei_idx = edge_index_nei_list[i]
            local_nei = halo_nei[:, 0]
            mask_nei = torch.isin(nei_idx[1], local_nei)
            edge_nei = nei_idx[:, mask_nei]

            # convert to global
            gli_own = data_reduced.global_ids
            own_send, own_recv = edge_own
            own_pair = cantor_pair(gli_own[own_send], gli_own[own_recv])

            gli_nei = global_ids_nei_list[i]
            nei_send, nei_recv = edge_nei
            nei_pair = cantor_pair(gli_nei[nei_send], gli_nei[nei_recv])

            # ------------------------------------
            # vectorized duplicate counting:
            # Q: how many times do each of my pairs occur in the neighboring rank's pairs?
            # 1) find each unique pairing in the neighbor and how many times it occurs
            uniq, counts = torch.unique(nei_pair, return_counts=True)

            # 2) sort so we can searchsorted
            uniq_sorted, idx_sort = torch.sort(uniq)
            counts_sorted = counts[idx_sort]

            # 3) locate insertion positions
            # returns index of where own_pair would be inserted into uniq_sorted to keep it sorted
            pos = torch.searchsorted(uniq_sorted, own_pair)

            # 4) clamp into [0, N-1] so indexing is always safe
            max_idx = uniq_sorted.numel() - 1
            pos_clamped = torch.clamp(pos, max=max_idx)

            # 5) check which actually match
            is_match = uniq_sorted[pos_clamped] == own_pair

            # 6) build duplicate-count vector: how many matches?
            dup_count = torch.zeros_like(pos, dtype=torch.int64)
            dup_count[is_match] = counts_sorted[pos_clamped][is_match]

            # 7) accumulate duplicates into the full edge_weights
            edge_weights[mask_own] += dup_count
    return edge_weights


if __name__ == "__main__":
    # Init MPI
    if not MPI.Is_initialized():
        MPI.Init()
    COMM = MPI.COMM_WORLD
    RANK = COMM.Get_rank()
    SIZE = COMM.Get_size()

    parser = argparse.ArgumentParser(
        description="Process command line arguments."
    )
    parser.add_argument(
        "--POLY", type=int, required=True, help="Specify the polynomial order."
    )
    parser.add_argument(
        "--PATH",
        type=str,
        required=True,
        help="Specify the gnn_outputs folder path.",
    )
    parser.add_argument(
        "--LOG",
        type=str,
        default="info",
        required=False,
        help="Logging verbosity",
    )
    args = parser.parse_args()

    POLY = args.POLY
    DIM = 3
    Np = (POLY + 1) ** DIM
    main_path = args.PATH + "/"

    # Make graph and reduced graph
    data, data_reduced, idx_keep = make_reduced_graph(COMM, RANK, SIZE)

    # Get this rank's halo triples for the reduced graph (local, no comm)
    halo_ids_local = get_reduced_halo_ids(COMM, RANK, SIZE, data_reduced)

    # Compute the halo_info
    if RANK == 0:
        print("Computing halo_info ...", flush=True)
    COMM.Barrier()
    t_start = MPI.Wtime()
    # halo_info_glob = get_halo_info(COMM, RANK, SIZE, data_reduced, halo_ids_local)
    halo_info_glob = get_halo_info_fast(
        COMM, RANK, SIZE, data_reduced, halo_ids_local
    )
    t_end = MPI.Wtime()
    local_time = t_end - t_start
    max_time = np.array([0.0])
    COMM.Allreduce(np.array([local_time]), max_time, op=MPI.MAX)
    if RANK == 0:
        print(f"Done in {max_time} seconds\n", flush=True)

    # Compute the node_degree
    if RANK == 0:
        print("Computing node_degree ...", flush=True)
    COMM.Barrier()
    t_start = MPI.Wtime()
    node_degree = get_node_degree(
        COMM, RANK, SIZE, data_reduced, halo_info_glob[RANK]
    )
    t_end = MPI.Wtime()
    local_time = t_end - t_start
    max_time = np.array([0.0])
    COMM.Allreduce(np.array([local_time]), max_time, op=MPI.MAX)
    if RANK == 0:
        print(f"Done in {max_time} seconds\n", flush=True)

    # Compute the edge_weights
    if RANK == 0:
        print("Computing edge_weights ...", flush=True)
    COMM.Barrier()
    t_start = MPI.Wtime()
    edge_weights = get_edge_weights(
        COMM, RANK, SIZE, data_reduced, halo_info_glob
    )
    t_end = MPI.Wtime()
    local_time = t_end - t_start
    max_time = np.array([0.0])
    COMM.Allreduce(np.array([local_time]), max_time, op=MPI.MAX)
    if RANK == 0:
        print(f"Done in {max_time} seconds\n", flush=True)

    # Write files
    if RANK == 0:
        print("Writing halo_info, edge_weights, node_degree ...", flush=True)
    np.save(
        main_path + "halo_info_rank_%d_size_%d.npy" % (RANK, SIZE),
        halo_info_glob[RANK].numpy(),
    )
    np.save(
        main_path + "node_degree_rank_%d_size_%d.npy" % (RANK, SIZE),
        node_degree.numpy(),
    )
    np.save(
        main_path + "edge_weights_rank_%d_size_%d.npy" % (RANK, SIZE),
        edge_weights.numpy(),
    )
    COMM.Barrier()
    if RANK == 0:
        print("Done \n", flush=True)

    if MPI.Is_initialized():
        MPI.Finalize()
