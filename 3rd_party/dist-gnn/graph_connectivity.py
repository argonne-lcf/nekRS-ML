import torch
from torch_geometric.data import Data
import torch_geometric.utils as utils
from typing import List, Tuple
Tensor = torch.Tensor


def get_reduced_graph(data_full: Data) -> Tuple[Data, Tensor]:
    """
    This function takes as input the GLL-based overlapping graph, and eliminates 
    overlapping nodes and edges to produce a REDUCED (non-overlapping) GLL-based graph. 
    This is equivalent to a graph pooling operation -- we are reducing the number of nodes 
    and edges in the input graph such that the output graph has no coincident local nodes. 
    """
    # X: [First isolate local nodes]
    idx_local_unique = torch.nonzero(data_full.local_unique_mask).squeeze(-1)
    idx_halo_unique = torch.nonzero(data_full.halo_unique_mask).squeeze(-1)
    idx_keep = torch.cat((idx_local_unique, idx_halo_unique))

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # PYGEOM FUNCTION -- this gets the reduced edge_index
    num_nodes = data_full.pos.shape[0]
    perm = idx_keep
    mask = perm.new_full((num_nodes, ), -1)
    i = torch.arange(perm.size(0), dtype=torch.long, device=perm.device)
    mask[perm] = i

    row, col = data_full.edge_index
    row, col = mask[row], mask[col]
    mask = (row >= 0) & (col >= 0)
    row, col = row[mask], col[mask]
    edge_index_reduced = torch.stack([row, col], dim=0)
    edge_index_reduced = utils.coalesce(edge_index_reduced)
    edge_index_reduced = utils.to_undirected(edge_index_reduced)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    pos_reduced = data_full.pos[idx_keep]
    pos_orig_reduced = data_full.pos_orig[idx_keep]
    gid_reduced = data_full.global_ids[idx_keep]
    data_reduced = Data(x = None, pos_orig = pos_orig_reduced, pos = pos_reduced, edge_index = edge_index_reduced, global_ids = gid_reduced)
    n_not_halo = len(idx_local_unique)
    n_halo = len(idx_halo_unique)
    data_reduced.local_unique_mask = torch.zeros(n_not_halo + n_halo, dtype=torch.int64)
    data_reduced.local_unique_mask[:n_not_halo] = 1
    data_reduced.halo_unique_mask = torch.zeros(n_not_halo + n_halo, dtype=torch.int64)
    data_reduced.halo_unique_mask[n_not_halo:] = 1
    data_reduced.local_ids = torch.tensor(range(data_full.pos.shape[0]))

    return data_reduced, idx_keep


def update_global_ids(data_full: Data, data_reduced: Data, idx_full2reduced: Tensor ) -> Data:
    """
    This function replaces the zero entries in global IDs in the full graph with consecutive negative 
    numbers. This is used to produce the upsampling indices.  
    """
    gid = data_full.global_ids
    zero_indices = torch.where(gid == 0)[0]
    consecutive_negatives = -1 * torch.arange(1, len(zero_indices) + 1)
    gid[zero_indices] = consecutive_negatives
    data_full.global_ids = gid 
    data_reduced.global_ids = gid[idx_full2reduced]
    
    return data_full, data_reduced

def get_upsample_indices(data_full: Data, data_reduced: Data, idx_full2reduced: Tensor) -> Tensor:
    """
    This function produces the indices that allow to go from the reduced (non-overlapping) node 
    representation, to the full (overlapping) node representation. 
    """
    # 1. Make all global IDs unique so coincident nodes share the same ID
    data_full, data_reduced = update_global_ids(
        data_full, data_reduced, idx_full2reduced
    )

    gid_full: Tensor = data_full.global_ids  # (N_full,)

    # Build mapping from global id -> *first* index in reduced graph (original order)
    gid_reduced: Tensor = data_reduced.global_ids  # (N_reduced,)

    # Sort reduced gids to group identical IDs
    gid_red_sorted, idx_sort_red = torch.sort(gid_reduced)

    # Identify first occurrence of each distinct gid
    is_first = torch.ones_like(gid_red_sorted, dtype=torch.bool)
    is_first[1:] = gid_red_sorted[1:] != gid_red_sorted[:-1]
    gid_unique = gid_red_sorted[is_first]  # ascending unique gids
    red_first_orig_idx = idx_sort_red[is_first]  # corresponding original indices

    # Map every full-graph node to the first occurrence in reduced graph
    # (gid_unique is sorted ascending, required by searchsorted)
    pos = torch.searchsorted(gid_unique, gid_full)  # (N_full,)
    scatter_ids = red_first_orig_idx[pos]

    # Sanity check: the mapping must reproduce gid_full exactly
    assert torch.equal(gid_reduced[scatter_ids], gid_full), (
        "get_upsample_indices: Inconsistent mapping - gid_reduced[scatter_ids] "
        "does not match gid_full"
    )
    return scatter_ids

def upsample_node_attributes(x_reduced: Tensor, 
                             gid_reduced: Tensor, lid_reduced: Tensor,
                             gid_full: Tensor, lid_full: Tensor, 
                             idx_reduced2full: Tensor) -> Tensor:
    """ 
    Performs the node upsampling to go back to the coincident representation.
    """

    # Sort full graph based on global id
    _, idx_sort = torch.sort(gid_full)
    gid_full = gid_full[idx_sort]
    lid_full = lid_full[idx_sort]

    # Sort reduced graph based on global id 
    _, idx_sort_reduced = torch.sort(gid_reduced)
    gid_reduced = gid_reduced[idx_sort_reduced]
    lid_reduced = lid_reduced[idx_sort_reduced]
    x_reduced = x_reduced[idx_sort_reduced]

    # Scatter back 
    # -- this is ordered by ascending global ID 
    x_full = x_reduced[idx_reduced2full]

    # Un-sort 
    _, idx_sort = torch.sort(lid_full)
    x_full = x_full[idx_sort]

    return x_full



