"""
Create halo swap info.
"""

import argparse
import numpy as np
from typing import Tuple
import time

import mpi4py

mpi4py.rc.initialize = False
from mpi4py import MPI

if not MPI.Is_initialized():
    MPI.Init()
COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
SIZE = COMM.Get_size()

import torch
from torch_geometric.data import Data
import torch_geometric.utils as utils


# helper Cantor-pairing on two 1D int64 tensors
def cantor_pair(k1: torch.Tensor, k2: torch.Tensor) -> torch.Tensor:
    # (0.5*(k1+k2)*(k1+k2+1) + k2) exactly as before, but keep it isolated
    s = k1.to(torch.float64) + k2.to(torch.float64)
    return (0.5 * s * (s + 1) + k2.to(torch.float64)).to(torch.int64)


def make_reduced_graph() -> Tuple[Data, Data, torch.Tensor]:
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
def get_reduced_halo_ids(data_reduced) -> torch.Tensor:
    idx_halo_unique = torch.tensor([], dtype=torch.int64)
    halo_ids = torch.tensor([], dtype=torch.int64)
    halo_ids_full = torch.tensor([], dtype=torch.int64)
    if SIZE > 1:
        # gid = data.global_ids

        # What are the local ids of the halo nodes ?
        n_local = data_reduced.local_unique_mask.sum().item()
        n_halo = data_reduced.halo_unique_mask.sum().item()
        idx_halo_unique = torch.tensor(list(range(n_local, n_local + n_halo)))

        # What are the corresponding global ids?
        gid_halo_unique = data_reduced.global_ids[idx_halo_unique]

        # What is the current rank?
        rank_array = torch.ones_like(gid_halo_unique, dtype=torch.int64) * RANK

        # [Local ids, global ids, rank]
        halo_ids = torch.concat(
            (
                idx_halo_unique.view(-1, 1),
                gid_halo_unique.view(-1, 1),
                rank_array.view(-1, 1),
            ),
            dim=1,
        )

        halo_ids_shape_list = COMM.allgather(halo_ids.shape[0])
        halo_ids_full_length = sum(halo_ids_shape_list)
        halo_ids_full_width = halo_ids.shape[1]
        halo_ids_full_type = halo_ids.dtype
        halo_ids_full = torch.zeros(
            halo_ids_full_length, halo_ids_full_width, dtype=halo_ids_full_type
        )

        count = [
            halo_ids_shape_list[i] * halo_ids_full_width for i in range(SIZE)
        ]
        displ = [sum(count[:i]) for i in range(SIZE)]
        # if args.LOG == 'debug' and RANK==0:
        #    print(f'count={count}',flush=True)
        #    print(f'displ={displ}',flush=True)
        COMM.Allgatherv(
            [halo_ids, MPI.LONG], [halo_ids_full, count, displ, MPI.LONG]
        )
    return halo_ids_full


# Prepares the halo_info matrix for halo swap
def get_halo_info(data_reduced, halo_ids_full) -> list:
    if SIZE == 1:
        halo_info_glob = [torch.tensor([], dtype=torch.int64)]
    else:
        # Collect number of nodes
        n_nodes = []
        n_nodes.append(data_reduced.pos.shape[0])
        n_nodes_glob = COMM.allgather(n_nodes[0])

        # concatenate
        # halo_ids_full = torch.cat(halo_ids_list)
        # halo_ids_full = torch.cat(halo_ids_list_glob)
        # del halo_ids_list_glob

        # take absolute value of global id
        halo_ids_full[:, 1] = torch.abs(halo_ids_full[:, 1])

        # sort in ascending order of global id
        global_ids = halo_ids_full[:, 1]
        _, idx_sort = torch.sort(global_ids)
        halo_ids_full = halo_ids_full[idx_sort]

        # get the frequency of nodes
        global_ids = halo_ids_full[:, 1]
        output = torch.unique_consecutive(
            global_ids, return_inverse=True, return_counts=True
        )
        counts_unique = output[2]
        counts = output[2][output[1]]
        counts = counts.reshape((-1, 1))
        if RANK == 0:
            print(f"global_ids shape = {global_ids.shape}", flush=True)
        if RANK == 0:
            print(f"counts_unique shape = {counts_unique.shape}", flush=True)
        if RANK == 0:
            print(f"counts shape = {counts.shape}", flush=True)

        # append the counts to halo_ids_full
        halo_ids_full = torch.cat([halo_ids_full, counts], dim=1)
        if RANK == 0:
            print(f"halo_ids_full shape = {halo_ids_full.shape}", flush=True)

        # Get the number of halo nodes for each rank
        # halo_info = []
        halo_ids_rank = halo_ids_full[halo_ids_full[:, 2] == RANK]
        Nhalo_rank = torch.sum(halo_ids_rank[:, 3] - 1)
        # halo_info.append(torch.zeros((Nhalo_rank,4), dtype=torch.int64))
        # halo_info_glob = COMM.allgather(halo_info[0])

        # Halo_info_glob is a list of tensors. Each element is a tensor of shape (Nhalo_rank_glob[i],4).
        # Columns in each element:[local_id of non halo nodes, local_id of halo nodes, global_id of nodes (same for local and halo), neighboring rank]
        Nhalo_rank_glob = COMM.allgather(Nhalo_rank)
        halo_info_glob = [
            torch.zeros((Nhalo_rank_glob[i], 4), dtype=torch.int64)
            for i in range(SIZE)
        ]

        # Loop through counts
        halo_counts = [0] * SIZE
        idx = 0
        for i in range(len(counts_unique)):
            count = counts_unique[i].item()
            halo_temp = halo_ids_full[idx : idx + count]
            # for j in range(count):
            #    a = halo_ids_full[idx]

            rank_list = halo_temp[:, 2]
            for j in range(len(rank_list)):
                rank = rank_list[j].item()

                # get the current rank info
                node_local_id = halo_temp[
                    j, 0
                ]  # local node id of sender on "rank"
                node_global_id = halo_temp[
                    j, 1
                ]  # global node id of sender on "rank"

                # loop through the same nodes not on this rank index
                halo_temp_nbrs = halo_temp[torch.arange(len(halo_temp)) != j]
                for k in range(len(halo_temp_nbrs)):
                    neighbor_rank = halo_temp_nbrs[
                        k, 2
                    ]  # neighboring rank for this halo node
                    node_halo_id = (
                        n_nodes_glob[rank] + halo_counts[rank]
                    )  # local node id of halo node on "rank"

                    # update the halo info matrix
                    halo_info_glob[rank][halo_counts[rank]][0] = node_local_id
                    halo_info_glob[rank][halo_counts[rank]][1] = node_halo_id
                    halo_info_glob[rank][halo_counts[rank]][2] = node_global_id
                    halo_info_glob[rank][halo_counts[rank]][3] = neighbor_rank

                    # update the count
                    halo_counts[rank] += 1

                    # print('[RANK %d] \t %d \t %d \t %d \n' %(rank, node_local_id, node_halo_id, neighbor_rank))

            # print('count = %d, idx = %d' %(count, idx))
            # print(a)
            # print('\n')
            idx += count
    return halo_info_glob


# Prepares the halo_info matrix for halo swap
def get_halo_info_fast(data_reduced, halo_ids_full):
    if SIZE == 1:
        return [torch.zeros((0, 4), dtype=torch.int64)]
    # — 1) sort by global_id and extract the three columns into separate vectors
    halo_ids_full[:, 1] = torch.abs(halo_ids_full[:, 1])
    _, idx_sort = torch.sort(halo_ids_full[:, 1])
    halo_ids_full = halo_ids_full[idx_sort]
    local_ids = halo_ids_full[:, 0]
    global_ids = halo_ids_full[:, 1]
    ranks = halo_ids_full[:, 2]

    # — 2) find consecutive runs of the same global_id
    _, inverse_idx, counts = torch.unique_consecutive(
        global_ids, return_inverse=True, return_counts=True
    )
    # compute the start index of each run
    starts = torch.cat(
        (
            torch.tensor([0], device=counts.device),
            torch.cumsum(counts, dim=0)[:-1],
        ),
        dim=0,
    )

    # — 3) build ALL (owner_idx, neighbor_idx) pairs for each run at once
    pair_list = []
    for start, cnt in zip(starts.tolist(), counts.tolist()):
        idx = torch.arange(start, start + cnt, device=halo_ids_full.device)
        I, J = torch.meshgrid(idx, idx, indexing="ij")
        mask = I != J
        pair_list.append(torch.stack((I[mask], J[mask]), dim=1))
    pairs = torch.cat(pair_list, dim=0)  # [M,2] where M = Σ (cnt*(cnt-1))

    # — 4) pull out the columns we need
    owner_idx, nbr_idx = pairs[:, 0], pairs[:, 1]
    owner_ranks = ranks[owner_idx]
    owner_locals = local_ids[owner_idx]
    owner_globals = global_ids[owner_idx]
    neighbor_ranks = ranks[nbr_idx]

    # — 5) build a big halo‐info tensor [M×4] with a placeholder in col 1
    halo_flat = torch.zeros(
        (pairs.size(0), 4), dtype=torch.int64, device=halo_ids_full.device
    )
    halo_flat[:, 0] = owner_locals
    halo_flat[:, 2] = owner_globals
    halo_flat[:, 3] = neighbor_ranks

    # — 6) split out each rank’s rows, and assign the proper halo‐node IDs
    #     (they start at n_nodes_glob[r] and count up by 1)
    n_nodes_glob = COMM.allgather(data_reduced.pos.shape[0])
    neighboring_procs = np.unique(halo_flat[owner_ranks == RANK, 3]).tolist()
    neighboring_procs = [RANK] + neighboring_procs
    halo_info_glob = [torch.empty(0)] * SIZE
    for r in neighboring_procs:
        mask_r = owner_ranks == r
        Hr = halo_flat[mask_r]
        cnt_r = Hr.size(0)
        if cnt_r:
            Hr[:, 1] = (
                torch.arange(cnt_r, dtype=torch.int64, device=Hr.device)
                + n_nodes_glob[r]
            )
        halo_info_glob[r] = Hr

    return halo_info_glob


# ~~~~ Get node degree from halo_info
def get_node_degree(data_reduced, halo_info_rank) -> torch.Tensor:
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
def get_edge_weights(data_reduced, halo_info_glob) -> torch.Tensor:
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
    data, data_reduced, idx_keep = make_reduced_graph()

    # Get halo_ids for reduced graph
    halo_ids_full = get_reduced_halo_ids(data_reduced)

    # Compute the halo_info
    if RANK == 0:
        print("Computing halo_info ...", flush=True)
    COMM.Barrier()
    t_start = MPI.Wtime()
    # halo_info_glob = get_halo_info(data_reduced, halo_ids_full)
    halo_info_glob = get_halo_info_fast(data_reduced, halo_ids_full)
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
    node_degree = get_node_degree(data_reduced, halo_info_glob[RANK])
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
    edge_weights = get_edge_weights(data_reduced, halo_info_glob)
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
