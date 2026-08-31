"""
Create halo swap info.
"""
import argparse 
import numpy as np
import torch

from torch_geometric.data import Data
import torch_geometric.utils as utils
import torch_geometric.nn as tgnn


parser = argparse.ArgumentParser(description='Process command line arguments.')
parser.add_argument('--SIZE', type=int, required=True, help='Specify the mpi world size corresponding to files in gnn_outputs directory.')
parser.add_argument('--POLY', type=int, required=True, help='Specify the polynomial order.')
parser.add_argument('--PATH', type=str, required=True, help='Specify the gnn_outputs folder path.')
args = parser.parse_args()

SIZE = args.SIZE
POLY = args.POLY
DIM = 3
Np = (POLY+1)**DIM
main_path = args.PATH + '/' # './gnn_outputs/' 

Ne_list = []
graph_list = []
graph_reduced_list = []
halo_ids_list = [] 
for RANK in range(SIZE): 
    path_to_pos_full = main_path + 'pos_node_rank_%d_size_%d' %(RANK,SIZE)
    path_to_ei = main_path + 'edge_index_rank_%d_size_%d' %(RANK,SIZE)
    path_to_glob_ids = main_path + 'global_ids_rank_%d_size_%d' %(RANK,SIZE)
    path_to_unique = main_path + 'local_unique_mask_rank_%d_size_%d' %(RANK,SIZE)
    path_to_halo_ids = None 
    if SIZE > 1:
        path_to_halo_ids =  main_path + 'halo_ids_rank_%d_size_%d' %(RANK,SIZE)
        path_to_unique_halo = main_path + 'halo_unique_mask_rank_%d_size_%d' %(RANK,SIZE)

    # ~~~~ Get positions and global node index  
    print('[RANK %d]: Loading positions and global node index' %(RANK))
    pos = np.fromfile(path_to_pos_full + ".bin", dtype=np.float64).reshape((-1,3))
    gli = np.fromfile(path_to_glob_ids + ".bin", dtype=np.int64).reshape((-1,1))
     
    # ~~~~ Back-out number of elements 
    Ne = int(pos.shape[0]/Np)
    Ne_list.append(Ne)
    print('[RANK %d]: Number of elements is %d' %(RANK, Ne))

    # ~~~~ Get edge index
    print('[RANK %d]: Loading edge index' %(RANK))
    ei = np.fromfile(path_to_ei + ".bin", dtype=np.int32).reshape((-1,2)).T
    ei = ei.astype(np.int64)

    # ~~~~ Get local unique mask 
    print('[RANK %d]: Loading local unique mask' %(RANK))
    local_unique_mask = np.fromfile(path_to_unique + ".bin", dtype=np.int32)

    # ~~~~ Get halo unique mask 
    halo_unique_mask = np.array([])
    if SIZE > 1:
        halo_unique_mask = np.fromfile(path_to_unique_halo + ".bin", dtype=np.int32)
    
    # ~~~~ Make graph:
    print('[RANK %d]: Making graph' %(RANK))
    data = Data(x = torch.tensor(pos), edge_index = torch.tensor(ei), pos = torch.tensor(pos), global_ids = torch.tensor(gli.squeeze()), local_unique_mask = torch.tensor(local_unique_mask), halo_unique_mask = torch.tensor(halo_unique_mask))
    data.edge_index = utils.remove_self_loops(data.edge_index)[0]
    data.edge_index = utils.coalesce(data.edge_index)
    data.edge_index = utils.to_undirected(data.edge_index)

    # ~~~~ Append list of graphs
    graph_list.append(data)

    # ~~~~ Reduce size of graph 
    print('[RANK %d]: Reduced size of edge_index based on unique node ids' %(RANK))
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
    mask = perm.new_full((num_nodes, ), -1)
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
    data_reduced = Data(x = pos_reduced, pos = pos_reduced, edge_index = edge_index_reduced, global_ids = gid_reduced)
    n_not_halo = len(idx_local_unique)
    n_halo = len(idx_halo_unique)
    data_reduced.local_unique_mask = torch.zeros(n_not_halo + n_halo, dtype=torch.int64)
    data_reduced.local_unique_mask[:n_not_halo] = 1
    data_reduced.halo_unique_mask = torch.zeros(n_not_halo + n_halo, dtype=torch.int64)
    data_reduced.halo_unique_mask[n_not_halo:] = 1
    graph_reduced_list.append(data_reduced)

    # ~~~~ Get the new halo_ids_list 
    print('[RANK %d]: Assembling halo_ids_list using reduced graph' %(RANK))
    idx_halo_unique = torch.tensor([], dtype=torch.int64)
    halo_ids = torch.tensor([], dtype=torch.int64)
    if SIZE > 1:
        gid = data.global_ids

        # What are the local ids of the halo nodes ? 
        n_local = data_reduced.local_unique_mask.sum().item()
        n_halo = data_reduced.halo_unique_mask.sum().item()
        idx_halo_unique = torch.tensor(list(range(n_local, n_local+n_halo)))

        # What are the corresponding global ids? 
        gid_halo_unique = data_reduced.global_ids[idx_halo_unique] 

        # What is the current rank? 
        rank_array = torch.ones_like(gid_halo_unique, dtype=torch.int64) * RANK 

        # [Local ids, global ids, rank] 
        halo_ids = torch.concat((idx_halo_unique.view(-1,1),
                                 gid_halo_unique.view(-1,1),
                                 rank_array.view(-1,1)), dim=1)
        
        # ~~~~ append list of halo id matrices
        halo_ids_list.append(halo_ids)

    # ~~~~ Generate the map back to original, non-reduced graph
    print('[RANK %d]: Demonstrating procedure for mapping back non-coincident to coincident graph' %(RANK))
    # Step 1: replace global id 0 with negative numbers 
    gid = data.global_ids 
    zero_indices = torch.where(gid == 0)[0]
    consecutive_negatives = -1 * torch.arange(1, len(zero_indices) + 1)
    gid[zero_indices] = consecutive_negatives
    data.global_ids = gid
    data_reduced.global_ids = gid[idx_keep]

    # Step 2: Get QoIs 
    # Get quantities from full graph (coincident nodes)  
    lid = torch.tensor(range(data.x.shape[0]))
    gid = data.global_ids
    pos = data.pos

    # Get quantities from reduced graph (no coincident nodes)  
    lid_reduced = torch.tensor(range(data_reduced.x.shape[0]))
    gid_reduced = data_reduced.global_ids
    pos_reduced = data_reduced.pos

    # Step 3: Sorting 
    # Sort full graph based on global id
    _, idx_sort = torch.sort(gid)
    gid = gid[idx_sort]
    lid = lid[idx_sort]
    pos = pos[idx_sort]

    # Sort reduced graph based on global id 
    _, idx_sort_reduced = torch.sort(gid_reduced)
    gid_reduced = gid_reduced[idx_sort_reduced]
    lid_reduced = lid_reduced[idx_sort_reduced]
    pos_reduced = pos_reduced[idx_sort_reduced]

    # Step 4: Get the scatter assignments 
    count = 0
    scatter_ids = torch.zeros(data.x.shape[0], dtype=torch.int64)
    scatter_ids[0] = count
    for i in range(1,len(gid)):

        gid_prev = gid[i-1]
        gid_curr = gid[i]

        if (gid_curr > gid_prev):
            count += 1

        scatter_ids[i] = count

    # Step 5: Scatter back 
    pos_recon = pos_reduced[scatter_ids]
    
    # Step 6: Un-sort, and compute error  
    _, idx_sort = torch.sort(lid)
    gid = gid[idx_sort]
    lid = lid[idx_sort]
    pos = pos[idx_sort]
    pos_recon = pos_recon[idx_sort]

    error = pos_recon - pos



n_nodes = []
for RANK in range(SIZE):
    data_reduced = graph_reduced_list[RANK]
    n_nodes.append(data_reduced.pos.shape[0])

# Prepares the halo_info matrix for halo swap 
halo_info = [torch.tensor([], dtype=torch.int64)]
if SIZE > 1:
    # concatenate 
    halo_ids_full = torch.cat(halo_ids_list)

    # take absolute value of global id 
    halo_ids_full[:,1] = torch.abs(halo_ids_full[:,1])

    # sort in ascending order of global id 
    global_ids = halo_ids_full[:,1] 
    _, idx_sort = torch.sort(global_ids)
    halo_ids_full = halo_ids_full[idx_sort]

    # get the frequency of nodes 
    global_ids = halo_ids_full[:,1] 
    output = torch.unique_consecutive(global_ids, return_inverse=True, return_counts=True)
    counts_unique = output[2]
    counts = output[2][output[1]]
    counts = counts.reshape((-1,1))

    # append the counts to halo_ids_full
    halo_ids_full = torch.cat([halo_ids_full, counts], dim=1)

    # Get the number of halo nodes for each rank
    halo_info = []
    for rank in range(SIZE):
        halo_ids_rank = halo_ids_full[halo_ids_full[:,2] == rank]
        Nhalo_rank = torch.sum(halo_ids_rank[:,3] - 1)
        halo_info.append(torch.zeros((Nhalo_rank,4), dtype=torch.int64))
        print('rank = %d, halo info shape is ' %(rank), halo_info[rank].shape)

    # Loop through counts 
    halo_counts = [0]*SIZE
    idx = 0 
    for i in range(len(counts_unique)):
        count = counts_unique[i].item()
        halo_temp = halo_ids_full[idx:idx+count]
        #for j in range(count): 
        #    a = halo_ids_full[idx]
    
        rank_list = halo_temp[:,2]
        for j in range(len(rank_list)):
            rank = rank_list[j].item()

            # get the current rank info 
            node_local_id = halo_temp[j,0] # local node id of sender on "rank"
            node_global_id = halo_temp[j,1] # global node id of sender on "rank" 

            # loop through the same nodes not on this rank index
            halo_temp_nbrs = halo_temp[torch.arange(len(halo_temp)) != j]
            for k in range(len(halo_temp_nbrs)):
                neighbor_rank = halo_temp_nbrs[k,2] # neighboring rank for this halo node 
                node_halo_id = n_nodes[rank] + halo_counts[rank] # local node id of halo node on "rank"  

                # update the halo info matrix
                halo_info[rank][halo_counts[rank]][0] = node_local_id
                halo_info[rank][halo_counts[rank]][1] = node_halo_id
                halo_info[rank][halo_counts[rank]][2] = node_global_id
                halo_info[rank][halo_counts[rank]][3] = neighbor_rank

                # update the count 
                halo_counts[rank] += 1

                # print('[RANK %d] \t %d \t %d \t %d \n' %(rank, node_local_id, node_halo_id, neighbor_rank))


        #print('count = %d, idx = %d' %(count, idx))
        #print(a)
        #print('\n')
        idx+=count


# ~~~~ Get node degree from halo_info
node_degree = []
if SIZE == 1:
    node_degree = [torch.ones( graph_reduced_list[0].pos.shape[0] )]
else:
    for RANK in range(SIZE):
        sample = graph_reduced_list[RANK]
        n_nodes_local = sample.pos.shape[0]
        node_degree_rank = torch.ones(n_nodes_local)
        halo_info_rank = halo_info[RANK]
        unique_local_indices, counts = torch.unique(halo_info_rank[:,0], return_counts=True)
        node_degree_rank[unique_local_indices] += counts
        node_degree.append(node_degree_rank)


# ~~~~ Get edge weights to account for duplicate edges 
edge_weights = []
if SIZE == 1:
    edge_weights = [torch.ones( graph_reduced_list[0].edge_index.shape[1] )]
else: 
    for RANK in range(SIZE):
        sample = graph_reduced_list[RANK]
        halo_info_rank = halo_info[RANK]

        # Get neighboring procs for this rank 
        neighboring_procs = np.unique(halo_info_rank[:,3])
        
        # Initialize edge weights 
        num_edges_own = sample.edge_index.shape[1]
        ew_own = torch.ones(num_edges_own)

        for j in neighboring_procs:
            rank_own = RANK
            rank_nei = j 

            halo_info_own = halo_info_rank
            halo_info_nei = halo_info[rank_nei]

            halo_info_own = halo_info_own[halo_info_own[:,3] == rank_nei, :] 
            halo_info_nei = halo_info_nei[halo_info_nei[:,3] == rank_own, :] 

            # check the global id ordering 
            if not torch.equal(halo_info_own[:,2], halo_info_nei[:,2]):
                raise AssertionError("halo_info tensors are not properly ordered by global id")
            
            # Get connectivities 
            edge_index_own = graph_reduced_list[rank_own].edge_index
            edge_index_nei = graph_reduced_list[rank_nei].edge_index

            #num_edges_nei = edge_index_nei.shape[1]
            #ew_nei = torch.ones(num_edges_nei)
            #edge_id_own = torch.arange(num_edges_own, dtype=torch.int64) 
            #edge_id_nei = torch.arange(num_edges_nei, dtype=torch.int64) 

            # Get the edges of owner rank nodes 
            local_id_own = halo_info_own[:,0]
            mask_own = torch.isin(edge_index_own[1], local_id_own)
            edge_index_own_subset = edge_index_own[:, mask_own]

            # Get the edges of neighbor rank nodes 
            local_id_nei = halo_info_nei[:,0]
            mask_nei = torch.isin(edge_index_nei[1], local_id_nei)
            edge_index_nei_subset = edge_index_nei[:, mask_nei]

            # Convert to global ids
            gli_own = graph_reduced_list[rank_own].global_ids
            send, recv = edge_index_own_subset
            send_global = gli_own[send]
            recv_global = gli_own[recv]
            edge_index_own_subset_global = torch.stack((send_global, recv_global))  

            gli_nei = graph_reduced_list[rank_nei].global_ids
            send, recv = edge_index_nei_subset
            send_global = gli_nei[send]
            recv_global = gli_nei[recv]
            edge_index_nei_subset_global = torch.stack((send_global, recv_global))  

            # Compute pairing functions: Cantor pair = 0.5 * (k1 + k2) * (k1 + k2 + 1) + k2 
            k1, k2 = edge_index_own_subset_global
            k1 = k1.to(torch.float64)
            k2 = k2.to(torch.float64)
            cpair_own = (0.5 * (k1 + k2) * (k1 + k2 + 1) + k2).to(torch.int64)

            k1, k2 = edge_index_nei_subset_global
            k1 = k1.to(torch.float64)
            k2 = k2.to(torch.float64)
            cpair_nei = (0.5 * (k1 + k2) * (k1 + k2 + 1) + k2).to(torch.int64)

            # Find which owner edges are duplicated in neighbor edges 
            duplicates_count = torch.zeros_like(cpair_own)
            for eid in range(len(cpair_own)):
                edge = cpair_own[eid]
                duplicates_count[eid] = (cpair_nei == edge).sum().item()

            # place in edge weighst 
            ew_own[mask_own] += duplicates_count 

        edge_weights.append(ew_own)


# ~~~~ Write halo_info, edge_weights, and node degree
for RANK in range(SIZE):
    halo_info_rank = halo_info[RANK]
    print('Writing %s' %(main_path + 'halo_info_rank_%d_size_%d.npy' %(RANK,SIZE)))
    np.save(main_path + 'halo_info_rank_%d_size_%d.npy' %(RANK,SIZE), halo_info_rank.numpy())

    edge_weights_rank = edge_weights[RANK]
    print('Writing %s' %(main_path + 'edge_weights_rank_%d_size_%d.npy' %(RANK,SIZE)))
    np.save(main_path + 'edge_weights_rank_%d_size_%d.npy' %(RANK,SIZE), edge_weights_rank.numpy())

    node_degree_rank = node_degree[RANK]
    print('Writing %s' %(main_path + 'node_degree_rank_%d_size_%d.npy' %(RANK,SIZE)))
    np.save(main_path + 'node_degree_rank_%d_size_%d.npy' %(RANK,SIZE), node_degree_rank.numpy())







