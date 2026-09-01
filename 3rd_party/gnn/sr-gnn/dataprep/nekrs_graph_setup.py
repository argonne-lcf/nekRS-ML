import numpy as np
import sys, time
import torch
import torch.nn.functional as F
import torch_geometric
from torch_geometric.data import Data
import torch_geometric.utils as utils
import torch_geometric.nn as tgnn
from typing import Optional, Union, Callable, List, Tuple
from pymech.neksuite import readnek

import logging

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)

Tensor = torch.Tensor
TORCH_FLOAT = torch.float32
NP_FLOAT = np.float32
TORCH_INT = torch.int64
NP_INT = np.int64


class DataLoHi(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == "edge_index_lo":
            return self.x.size(0)
        if key == "edge_index_coin":
            return self.x.size(0)
        if key == "edge_index_hi":
            return self.y.size(0)
        return super().__inc__(key, value, *args, **kwargs)


class DataLoHiIncr(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == "edge_index_lo_0":
            return self.x_lo_0.size(0)
        if key == "edge_index_coin_0":
            return self.x_lo_0.size(0)
        if key == "edge_index_hi_0":
            return self.x_hi_0.size(0)
        if key == "edge_index_lo_1":
            return self.x_lo_1.size(0)
        if key == "edge_index_coin_1":
            return self.x_lo_1.size(0)
        if key == "edge_index_hi_1":
            return self.x_hi_1.size(0)
        if key == "edge_index_lo_2":
            return self.x_lo_2.size(0)
        if key == "edge_index_coin_2":
            return self.x_lo_2.size(0)
        if key == "edge_index_hi_2":
            return self.x_hi_2.size(0)
        return super().__inc__(key, value, *args, **kwargs)


def get_rms(x_batch: Tensor) -> Tensor:
    u_var = x_batch.var(dim=1, keepdim=True)
    tke = 0.5 * u_var.sum(dim=2)
    u_rms = torch.sqrt(tke / 1.5)
    return u_rms


def get_stats(x_batch: Tensor) -> Tuple[Tensor, Tensor]:
    x_batch_mean = torch.mean(x_batch, dim=1)
    x_batch_std = torch.std(x_batch, dim=1)
    return x_batch_mean, x_batch_std


def get_element_lengthscale(pos_batch: Tensor) -> Tensor:
    pos_min = pos_batch.min(dim=1)[0]
    pos_max = pos_batch.max(dim=1)[0]
    return torch.norm(pos_max - pos_min, p=2, dim=1)


def get_edge_index_coincident(batch, pos, edge_index):
    if batch is None:
        batch = edge_index.new_zeros(pos.size(0))

    pos_unbatch = utils.unbatch(pos, batch)
    ei_coin_unbatch = []
    n_nodes_unbatch = []
    n_nodes_incr = [0]
    for b in range(batch.max() + 1):
        pos = pos_unbatch[b]
        ei_coin = tgnn.radius_graph(pos, r=1e-9, max_num_neighbors=32)
        n_nodes_unbatch.append(pos.shape[0])
        ei_coin_unbatch.append(ei_coin)
        if b > 0:
            n_nodes_incr.append(n_nodes_unbatch[b] + n_nodes_incr[b - 1])

    for b in range(batch.max() + 1):
        ei_coin_unbatch[b] = ei_coin_unbatch[b] + n_nodes_incr[b]

    ei_coin = torch.concat(ei_coin_unbatch, dim=1)
    return ei_coin


def get_pygeom_dataset_lo_hi_pymech(
    data_xlo_path: str,
    data_xhi_path: str,
    edge_index_path_lo: str,
    edge_index_path_hi: str,
    node_weight: Optional[float] = 1.0,
    device_for_loading: Optional[str] = "cpu",
    fraction_valid: Optional[float] = 0.1,
    n_element_neighbors: Optional[int] = 0,
) -> Tuple[List, List]:
    t_load = time.time()
    logger.debug(
        "In get_pygeom_dataset_lo_hi_pymech. Loading data and making pygeom dataset..."
    )
    edge_index_lo = np.loadtxt(edge_index_path_lo, dtype=np.int64).T
    edge_index_lo = torch.tensor(edge_index_lo)
    edge_index_hi = np.loadtxt(edge_index_path_hi, dtype=np.int64).T
    edge_index_hi = torch.tensor(edge_index_hi)

    edge_index = edge_index_lo
    n_nodes_per_element = edge_index.max() + 1
    if n_element_neighbors > 0:
        node_max_per_element = edge_index.max()
        n_edges_per_element = edge_index.shape[1]
        edge_index_full = torch.zeros(
            (2, n_edges_per_element * (n_element_neighbors + 1)),
            dtype=edge_index.dtype,
        )
        edge_index_full[:, :n_edges_per_element] = edge_index
        for i in range(1, n_element_neighbors + 1):
            start = n_edges_per_element * i
            end = n_edges_per_element * (i + 1)
            edge_index_full[:, start:end] = (
                edge_index + (node_max_per_element + 1) * i
            )
        edge_index = edge_index_full

    # Load data
    xlo_field = readnek(data_xlo_path)
    xhi_field = readnek(data_xhi_path)

    # Prune data if needed
    elm_id = list(range(xhi_field.nel))
    elm_id = np.array(elm_id, dtype=np.longlong)
    centroids = np.zeros((xhi_field.nel, 3), dtype=np.float64)
    for i in range(xhi_field.nel):
        centroids[i, :] = xhi_field.elem[i].centroid[:]
    # Now can prune elements based on coordinates of centroids
    eid_keep = elm_id

    # Train / valid split
    n_snaps = len(eid_keep)
    if fraction_valid > 0:
        # How many total snapshots to extract
        n_full = n_snaps
        n_valid = int(np.floor(fraction_valid * n_full))

        # Get validation set indices
        idx_valid = np.sort(np.random.choice(n_full, n_valid, replace=False))

        # Get training set indices
        idx_train = np.array(
            list(set(list(range(n_full))) - set(list(idx_valid)))
        )

        n_train = len(idx_train)
        n_valid = len(idx_valid)
    else:
        n_full = n_snaps
        n_valid = 0
        n_train = n_full
        idx_train = list(range(n_train))
        idx_valid = []

    idx_train_mask = np.zeros(n_snaps, dtype=int)
    idx_train_mask[idx_train] = 1

    # Get the element neighborhoods
    if n_element_neighbors > 0:
        Nelements = len(xlo_field.elem)
        pos_c = torch.zeros((Nelements, 3))
        for i in range(Nelements):
            pos_c[i] = torch.tensor(xlo_field.elem[i].centroid)
        edge_index_c = tgnn.knn_graph(x=pos_c, k=n_element_neighbors)

    # Get the element masks
    central_element_mask = torch.concat((
        torch.ones((n_nodes_per_element), dtype=torch.int64),
        torch.zeros(
            (n_nodes_per_element * n_element_neighbors), dtype=torch.int64
        ),
    ))
    central_element_mask = central_element_mask.to(torch.bool)

    data_train_list = []
    data_valid_list = []
    for it in range(len(eid_keep)):
        i = eid_keep[it]  # element id

        pos_xlo_i = (
            torch.tensor(xlo_field.elem[i].pos).reshape((3, -1)).T
        )  # pygeom pos format -- [N, 3]
        vel_xlo_i = torch.tensor(xlo_field.elem[i].vel).reshape((3, -1)).T
        pos_xhi_i = (
            torch.tensor(xhi_field.elem[i].pos).reshape((3, -1)).T
        )  # pygeom pos format -- [N, 3]
        vel_xhi_i = torch.tensor(xhi_field.elem[i].vel).reshape((3, -1)).T

        elm_vtx_lo_i = get_element_vertices(pos_xlo_i)
        elm_vtx_hi_i = get_element_vertices(pos_xhi_i)

        dx_min = (elm_vtx_lo_i.max(dim=0)[0] - elm_vtx_lo_i.min(dim=0)[0]).min()
        error_max = (
            elm_vtx_lo_i.max(dim=0)[0] - elm_vtx_hi_i.max(dim=0)[0]
        ).max()
        error_min = (
            elm_vtx_lo_i.min(dim=0)[0] - elm_vtx_hi_i.min(dim=0)[0]
        ).max()
        rel_error_max = torch.abs(error_max / dx_min) * 100
        rel_error_min = torch.abs(error_min / dx_min) * 100

        # Check positions
        if (rel_error_max > 1e-2) or (rel_error_min > 1e-2):
            logger.error(
                f"Relative error in positions exceeds 0.01% in element i={i}."
            )
            sys.exit()
        if pos_xlo_i.max() == 0.0 and pos_xlo_i.min() == 0.0:
            logger.error(f"Node positions are not stored in {data_xlo_path}.")
            sys.exit()
        if pos_xhi_i.max() == 0.0 and pos_xhi_i.min() == 0.0:
            logger.error(f"Node positions are not stored in {data_xhi_path}.")
            sys.exit()

        # get x_mean and x_std
        x_mean_element_lo = (
            torch
            .mean(vel_xlo_i, dim=0)
            .unsqueeze(0)
            .repeat(central_element_mask.shape[0], 1)
        )
        x_std_element_lo = (
            torch
            .std(vel_xlo_i, dim=0)
            .unsqueeze(0)
            .repeat(central_element_mask.shape[0], 1)
        )
        x_mean_element_hi = (
            torch
            .mean(vel_xlo_i, dim=0)
            .unsqueeze(0)
            .repeat(vel_xhi_i.shape[0], 1)
        )
        x_std_element_hi = (
            torch
            .std(vel_xlo_i, dim=0)
            .unsqueeze(0)
            .repeat(vel_xhi_i.shape[0], 1)
        )

        # element lengthscale
        lengthscale_element = torch.norm(
            pos_xlo_i.max(dim=0)[0] - pos_xlo_i.min(dim=0)[0], p=2
        )

        # node weight
        nw = torch.ones((vel_xhi_i.shape[0], 1)) * node_weight

        # Get the element neighbors for the input
        if n_element_neighbors > 0:
            send = edge_index_c[0, :]
            recv = edge_index_c[1, :]
            nbrs = send[recv == i]

            pos_x_full = [pos_xlo_i]
            vel_x_full = [vel_xlo_i]
            for j in nbrs:
                pos_x_full.append(
                    torch.tensor(xlo_field.elem[j].pos).reshape((3, -1)).T
                )
                vel_x_full.append(
                    torch.tensor(xlo_field.elem[j].vel).reshape((3, -1)).T
                )
            pos_x_full = torch.concat(pos_x_full)
            vel_x_full = torch.concat(vel_x_full)

            # reset pos
            pos_xlo_i = pos_x_full
            vel_xlo_i = vel_x_full

        # create data
        data_temp = DataLoHi(
            x=vel_xlo_i.to(dtype=TORCH_FLOAT),
            y=vel_xhi_i.to(dtype=TORCH_FLOAT),
            x_mean_lo=x_mean_element_lo.to(dtype=TORCH_FLOAT),
            x_std_lo=x_std_element_lo.to(dtype=TORCH_FLOAT),
            x_mean_hi=x_mean_element_hi.to(dtype=TORCH_FLOAT),
            x_std_hi=x_std_element_hi.to(dtype=TORCH_FLOAT),
            node_weight=nw.to(dtype=TORCH_FLOAT),
            L=lengthscale_element.to(dtype=TORCH_FLOAT),
            pos_norm_lo=(pos_xlo_i / lengthscale_element).to(dtype=TORCH_FLOAT),
            pos_norm_hi=(pos_xhi_i / lengthscale_element).to(dtype=TORCH_FLOAT),
            edge_index_lo=edge_index,
            edge_index_hi=edge_index_hi,
            central_element_mask=central_element_mask,
            eid=torch.tensor(i),
        )

        # for synchronizing across element boundaries
        if n_element_neighbors > 0:
            batch = None
            edge_index_coin = get_edge_index_coincident(
                batch, data_temp.pos_norm_lo, data_temp.edge_index_lo
            )
            degree = utils.degree(
                edge_index_coin[1, :], num_nodes=data_temp.pos_norm_lo.shape[0]
            )
            degree += 1.0
            data_temp.edge_index_coin = edge_index_coin
            data_temp.degree = degree
        else:
            data_temp.edge_index_coin = None
            data_temp.degree = None

        data_temp = data_temp.to(device_for_loading)

        if idx_train_mask[it] == 1:
            data_train_list.append(data_temp)
        else:
            data_valid_list.append(data_temp)

    t_load = time.time() - t_load
    logger.info(f"Created PT Data in {t_load:.2f} sec")
    return {
        "train": {"data": data_train_list, "num_samples": n_train},
        "valid": {"data": data_valid_list, "num_samples": n_valid},
    }


def get_element_vertices(pos: torch.Tensor) -> torch.Tensor:
    """Given the position of the GLL points within the element,
    compute the polynomial order and return the vertices of the element.
    """
    Ngll = round(pos.shape[0] ** (1 / 3))
    N = Ngll - 1

    if N == 1:
        return pos
    else:
        indices = torch.tensor([
            0,
            Ngll - 1,
            Ngll**2 - N - 1,
            Ngll**2 - 1,
            Ngll**3 - Ngll**2,
            Ngll**3 - Ngll**2 + N,
            Ngll**3 - N - 1,
            Ngll**3 - 1,
        ])
        elm_vtx = pos[indices, :]
        return elm_vtx
