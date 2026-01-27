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


def get_pygeom_dataset_pymech(
    data_x_path: str,
    data_y_path: str,
    edge_index_path: str,
    edge_index_vertex_path: Optional[str] = None,
    node_weight: Optional[float] = 1.0,
    device_for_loading: Optional[str] = "cpu",
    fraction_valid: Optional[float] = 0.1,
    n_element_neighbors: Optional[int] = 0,
) -> Tuple[List, List]:
    t_load = time.time()

    logger.info("Loading data and making pygeom dataset...")
    edge_index = np.loadtxt(edge_index_path, dtype=np.int64).T
    if edge_index_vertex_path:
        logger.info("Adding p1 connectivity...")
        logger.info("\tEdge index shape before: ", edge_index.shape)
        edge_index_vertex = np.loadtxt(edge_index_vertex_path, dtype=np.int64).T
        edge_index = np.concatenate((edge_index, edge_index_vertex), axis=1)
        logger.info("\tEdge index shape after: ", edge_index.shape)
    edge_index = torch.tensor(edge_index)

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
    x_field = readnek(data_x_path)
    y_field = readnek(data_y_path)

    # Train / valid split
    n_snaps = len(x_field.elem)

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
        Nelements = len(x_field.elem)
        pos_c = torch.zeros((Nelements, 3))
        for i in range(Nelements):
            pos_c[i] = torch.tensor(x_field.elem[i].centroid)
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
    for i in range(n_snaps):
        N_i = x_field.elem[i].pos.shape[1]
        pos_x_i = (
            torch.tensor(x_field.elem[i].pos).reshape((3, -1)).T
        )  # pygeom pos format -- [N, 3]
        pos_y_i = (
            torch.tensor(y_field.elem[i].pos).reshape((3, -1)).T
        )  # pygeom pos format -- [N, 3]
        vel_x_i = torch.tensor(x_field.elem[i].vel).reshape((3, -1)).T
        vel_y_i = torch.tensor(y_field.elem[i].vel).reshape((3, -1)).T

        x_gll = x_field.elem[i].pos[0, 0, 0, :]
        dx_min = x_gll[1] - x_gll[0]

        error_max = (pos_x_i - pos_y_i).max()
        rel_error = torch.abs(error_max / dx_min) * 100

        # Check positions
        if rel_error > 1e-2:
            logger.error(
                f"Relative error in positions exceeds 0.01% in element i={i}."
            )
            sys.exit()
        if pos_x_i.max() == 0.0 and pos_x_i.min() == 0.0:
            logger.error(f"Node positions are not stored in {data_x_path}.")
            sys.exit()
        if pos_y_i.max() == 0.0 and pos_y_i.min() == 0.0:
            logger.error(f"Node positions are not stored in {data_y_path}.")
            sys.exit()
        pos_i = pos_x_i

        # get x_mean and x_std
        x_mean_element = (
            torch
            .mean(vel_x_i, dim=0)
            .unsqueeze(0)
            .repeat(central_element_mask.shape[0], 1)
        )
        x_std_element = (
            torch
            .std(vel_x_i, dim=0)
            .unsqueeze(0)
            .repeat(central_element_mask.shape[0], 1)
        )

        # element lengthscale
        lengthscale_element = torch.norm(
            pos_i.max(dim=0)[0] - pos_i.min(dim=0)[0], p=2
        )

        # node weight
        nw = torch.ones((vel_x_i.shape[0], 1)) * node_weight

        # Get the element neighbors for the input
        if n_element_neighbors > 0:
            send = edge_index_c[0, :]
            recv = edge_index_c[1, :]
            nbrs = send[recv == i]

            pos_x_full = [pos_x_i]
            vel_x_full = [vel_x_i]
            for j in nbrs:
                pos_x_full.append(
                    torch.tensor(x_field.elem[j].pos).reshape((3, -1)).T
                )
                vel_x_full.append(
                    torch.tensor(x_field.elem[j].vel).reshape((3, -1)).T
                )
            pos_x_full = torch.concat(pos_x_full)
            vel_x_full = torch.concat(vel_x_full)

            # reset pos
            pos_i = pos_x_full
            vel_x_i = vel_x_full

        # create data
        data_temp = Data(
            x=vel_x_i.to(dtype=TORCH_FLOAT),
            y=vel_y_i.to(dtype=TORCH_FLOAT),
            x_mean=x_mean_element.to(dtype=TORCH_FLOAT),
            x_std=x_std_element.to(dtype=TORCH_FLOAT),
            node_weight=nw.to(dtype=TORCH_FLOAT),
            L=lengthscale_element.to(dtype=TORCH_FLOAT),
            pos=pos_i.to(dtype=TORCH_FLOAT),
            pos_norm=(pos_i / lengthscale_element).to(dtype=TORCH_FLOAT),
            edge_index=edge_index,
            central_element_mask=central_element_mask,
            eid=torch.tensor(i),
        )

        data_temp = data_temp.to(device_for_loading)

        if idx_train_mask[i] == 1:
            data_train_list.append(data_temp)
        else:
            data_valid_list.append(data_temp)

    t_load = time.time() - t_load
    logger.info("\ttook %g sec" % (t_load))
    return data_train_list, data_valid_list


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

        print(f"element id: {i}")
        print(f"pos_xlo_i: {pos_xlo_i.shape}")
        print(f"pos_xhi_i: {pos_xhi_i.shape}")
        print(f"pos_xlo_i: {pos_xlo_i[:8]}")
        print(f"pos_xhi_i: {pos_xhi_i[:16]}")

        elm_vtx_lo = get_element_vertices(pos_xlo_i)
        elm_vtx_hi = get_element_vertices(pos_xhi_i)
        print(f"elm_vtx_lo: {elm_vtx_lo}")
        print(f"elm_vtx_hi: {elm_vtx_hi}")

        x_gll = xhi_field.elem[i].pos[0, 0, 0, :]
        dx_min = x_gll[1] - x_gll[0]

        print("Max pos_xlo_i: ", pos_xlo_i.max(dim=0)[0])
        print("Max pos_xhi_i: ", pos_xhi_i.max(dim=0)[0])
        print("Min pos_xlo_i: ", pos_xlo_i.min(dim=0)[0])
        print("Min pos_xhi_i: ", pos_xhi_i.min(dim=0)[0])

        error_max = (pos_xlo_i.max(dim=0)[0] - pos_xhi_i.max(dim=0)[0]).max()
        error_min = (pos_xlo_i.min(dim=0)[0] - pos_xhi_i.min(dim=0)[0]).max()
        rel_error_max = torch.abs(error_max / dx_min) * 100
        rel_error_min = torch.abs(error_min / dx_min) * 100
        print(f"error_max: {error_max}")
        print(f"error_min: {error_min}")
        print(f"dx_min: {dx_min}")
        print(f"rel_error_max: {rel_error_max}")
        print(f"rel_error_min: {rel_error_min}")
        print("--------------------------------\n\n")

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
    logger.info("\ttook %g sec" % (t_load))
    return {
        "train": {"data": data_train_list, "num_samples": n_train},
        "valid": {"data": data_valid_list, "num_samples": n_valid},
    }


def get_pygeom_dataset_lo_hi_pymech_incr(
    data_x_path: List[str],
    edge_index_path: List[str],
    node_weight: Optional[float] = 1.0,
    device_for_loading: Optional[str] = "cpu",
    fraction_valid: Optional[float] = 0.1,
    n_element_neighbors: Optional[int] = 0,
) -> Tuple[List, List]:
    t_load = time.time()
    logger.debug(
        "In get_pygeom_dataset_lo_hi_pymech_incr. Loading data and making pygeom dataset..."
    )

    data_xlo_path = data_x_path[:-1]
    data_xhi_path = data_x_path[1:]
    edge_index_path_lo = edge_index_path[:-1]
    edge_index_path_hi = edge_index_path[1:]

    edge_index_lo = []
    edge_index_hi = []

    n_grids = len(data_x_path)  # total number of p grids
    # n_levels = len(data_xlo_path) # number of different transfers -- p=1 to 3, 3 to 5, 5 to 3 gives n_levels = 3
    n_levels = n_grids - 1

    for i in range(n_levels):
        edge_index_lo.append(
            np.loadtxt(edge_index_path_lo[i], dtype=np.int64).T
        )
        edge_index_lo[i] = torch.tensor(edge_index_lo[i])
        edge_index_hi.append(
            np.loadtxt(edge_index_path_hi[i], dtype=np.int64).T
        )
        edge_index_hi[i] = torch.tensor(edge_index_hi[i])

    edge_index_list = []
    n_nodes_per_element = []
    for i in range(n_levels):
        edge_index = edge_index_lo[i]
        n_nodes_per_element.append(edge_index.max() + 1)
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
        edge_index_list.append(edge_index)
    edge_index = edge_index_list

    # Load data
    x_fields = []
    for i in range(n_grids):
        logger.info(f"\t readnek({data_x_path[i]}) ...")
        x_fields.append(readnek(data_x_path[i]))

    xlo_field = x_fields[:-1]
    xhi_field = x_fields[1:]

    n_snaps_list = [len(a.elem) for a in x_fields]

    if not all(a == n_snaps_list[0] for a in n_snaps_list):
        raise SystemExit(
            "Number of elements across snapshots is not the same! Aborting..."
        )

    # Train / valid split
    n_snaps = n_snaps_list[0]

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
        edge_index_c = []
        for lev in range(n_levels):
            Nelements = len(xlo_field[lev].elem)
            pos_c = torch.zeros((Nelements, 3))
            for i in range(Nelements):
                pos_c[i] = torch.tensor(xlo_field[lev].elem[i].centroid)
            edge_index_c.append(tgnn.knn_graph(x=pos_c, k=n_element_neighbors))

    # Get the element masks
    central_element_mask = []
    for lev in range(n_levels):
        central_element_mask.append(
            torch.concat((
                torch.ones((n_nodes_per_element[lev]), dtype=torch.int64),
                torch.zeros(
                    (n_nodes_per_element[lev] * n_element_neighbors),
                    dtype=torch.int64,
                ),
            ))
        )
        central_element_mask[lev] = central_element_mask[lev].to(torch.bool)

    data_train_list = []
    data_valid_list = []

    for i in range(n_snaps):
        data_temp_list = []
        # data_temp = Data()
        for lev in range(n_levels):
            pos_xlo_i = (
                torch.tensor(xlo_field[lev].elem[i].pos).reshape((3, -1)).T
            )  # pygeom pos format -- [N, 3]
            vel_xlo_i = (
                torch.tensor(xlo_field[lev].elem[i].vel).reshape((3, -1)).T
            )
            pos_xhi_i = (
                torch.tensor(xhi_field[lev].elem[i].pos).reshape((3, -1)).T
            )  # pygeom pos format -- [N, 3]
            vel_xhi_i = (
                torch.tensor(xhi_field[lev].elem[i].vel).reshape((3, -1)).T
            )

            x_gll = xhi_field[lev].elem[i].pos[0, 0, 0, :]
            dx_min = x_gll[1] - x_gll[0]

            error_max = (
                pos_xlo_i.max(dim=0)[0] - pos_xhi_i.max(dim=0)[0]
            ).max()
            error_min = (
                pos_xlo_i.min(dim=0)[0] - pos_xhi_i.min(dim=0)[0]
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
                logger.error(
                    f"Node positions are not stored in {data_xlo_path}."
                )
                sys.exit()
            if pos_xhi_i.max() == 0.0 and pos_xhi_i.min() == 0.0:
                logger.error(
                    f"Node positions are not stored in {data_xhi_path}."
                )
                sys.exit()

            # get x_mean and x_std
            x_mean_element_lo = (
                torch
                .mean(vel_xlo_i, dim=0)
                .unsqueeze(0)
                .repeat(central_element_mask[lev].shape[0], 1)
            )
            x_std_element_lo = (
                torch
                .std(vel_xlo_i, dim=0)
                .unsqueeze(0)
                .repeat(central_element_mask[lev].shape[0], 1)
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
                send = edge_index_c[lev][0, :]
                recv = edge_index_c[lev][1, :]
                nbrs = send[recv == i]

                pos_x_full = [pos_xlo_i]
                vel_x_full = [vel_xlo_i]
                for j in nbrs:
                    pos_x_full.append(
                        torch
                        .tensor(xlo_field[lev].elem[j].pos)
                        .reshape((3, -1))
                        .T
                    )
                    vel_x_full.append(
                        torch
                        .tensor(xlo_field[lev].elem[j].vel)
                        .reshape((3, -1))
                        .T
                    )
                pos_x_full = torch.concat(pos_x_full)
                vel_x_full = torch.concat(vel_x_full)

                # reset pos
                pos_xlo_i = pos_x_full
                vel_xlo_i = vel_x_full

            # create data
            data_temp = Data(
                x=vel_xlo_i.to(dtype=TORCH_FLOAT),
                y=vel_xhi_i.to(dtype=TORCH_FLOAT),
                x_mean_lo=x_mean_element_lo.to(dtype=TORCH_FLOAT),
                x_std_lo=x_std_element_lo.to(dtype=TORCH_FLOAT),
                x_mean_hi=x_mean_element_hi.to(dtype=TORCH_FLOAT),
                x_std_hi=x_std_element_hi.to(dtype=TORCH_FLOAT),
                node_weight=nw.to(dtype=TORCH_FLOAT),
                L=lengthscale_element.to(dtype=TORCH_FLOAT),
                pos_norm_lo=(pos_xlo_i / lengthscale_element).to(
                    dtype=TORCH_FLOAT
                ),
                pos_norm_hi=(pos_xhi_i / lengthscale_element).to(
                    dtype=TORCH_FLOAT
                ),
                edge_index_lo=edge_index[lev],
                edge_index_hi=edge_index_hi[lev],
                central_element_mask=central_element_mask[lev],
                eid=torch.tensor(i),
            )

            # for synchronizing across element boundaries
            if n_element_neighbors > 0:
                batch = None
                edge_index_coin = get_edge_index_coincident(
                    batch,
                    (pos_xlo_i / lengthscale_element).to(dtype=TORCH_FLOAT),
                    edge_index[lev],
                )
                degree = utils.degree(
                    edge_index_coin[1, :], num_nodes=pos_xlo_i.shape[0]
                )
                degree += 1.0
                data_temp.edge_index_coin = edge_index_coin
                data_temp.degree = degree
            else:
                data_temp.edge_index_coin = torch.Tensor([])
                data_temp.degree = torch.Tensor([])

            data_temp_list.append(data_temp)

            # if lev == 0:
            #     data_temp.x_0 = vel_xlo_i.to(dtype=TORCH_FLOAT),
            #     data_temp.y_0 = vel_xhi_i.to(dtype=TORCH_FLOAT),
            #     data_temp.x_mean_lo_0 = x_mean_element_lo.to(dtype=TORCH_FLOAT),
            #     data_temp.x_std_lo_0 = x_std_element_lo.to(dtype=TORCH_FLOAT),
            #     data_temp.x_mean_hi_0 = x_mean_element_hi.to(dtype=TORCH_FLOAT),
            #     data_temp.x_std_hi_0 = x_std_element_hi.to(dtype=TORCH_FLOAT),
            #     data_temp.node_weight_0 = nw.to(dtype=TORCH_FLOAT),
            #     data_temp.L_0 = lengthscale_element.to(dtype=TORCH_FLOAT),
            #     data_temp.pos_norm_lo_0 = (pos_xlo_i/lengthscale_element).to(dtype=TORCH_FLOAT),
            #     data_temp.pos_norm_hi_0 = (pos_xhi_i/lengthscale_element).to(dtype=TORCH_FLOAT),
            #     data_temp.edge_index_lo_0 = edge_index[lev],
            #     data_temp.edge_index_hi_0 = edge_index_hi[lev],
            #     data_temp.central_element_mask_0 = central_element_mask[lev],
            #     data_temp.eid_0 = torch.tensor(i)
            # elif lev == 1:
            #     data_temp.x_1 = vel_xlo_i.to(dtype=TORCH_FLOAT),
            #     data_temp.y_1 = vel_xhi_i.to(dtype=TORCH_FLOAT),
            #     data_temp.x_mean_lo_1 = x_mean_element_lo.to(dtype=TORCH_FLOAT),
            #     data_temp.x_std_lo_1 = x_std_element_lo.to(dtype=TORCH_FLOAT),
            #     data_temp.x_mean_hi_1 = x_mean_element_hi.to(dtype=TORCH_FLOAT),
            #     data_temp.x_std_hi_1 = x_std_element_hi.to(dtype=TORCH_FLOAT),
            #     data_temp.node_weight_1 = nw.to(dtype=TORCH_FLOAT),
            #     data_temp.L_1 = lengthscale_element.to(dtype=TORCH_FLOAT),
            #     data_temp.pos_norm_lo_1 = (pos_xlo_i/lengthscale_element).to(dtype=TORCH_FLOAT),
            #     data_temp.pos_norm_hi_1 = (pos_xhi_i/lengthscale_element).to(dtype=TORCH_FLOAT),
            #     data_temp.edge_index_lo_1 = edge_index[lev],
            #     data_temp.edge_index_hi_1 = edge_index_hi[lev],
            #     data_temp.central_element_mask_1 = central_element_mask[lev],
            #     data_temp.eid_1 = torch.tensor(i)
            # elif lev == 2:
            #     data_temp.x_2 = vel_xlo_i.to(dtype=TORCH_FLOAT),
            #     data_temp.y_2 = vel_xhi_i.to(dtype=TORCH_FLOAT),
            #     data_temp.x_mean_lo_2 = x_mean_element_lo.to(dtype=TORCH_FLOAT),
            #     data_temp.x_std_lo_2 = x_std_element_lo.to(dtype=TORCH_FLOAT),
            #     data_temp.x_mean_hi_2 = x_mean_element_hi.to(dtype=TORCH_FLOAT),
            #     data_temp.x_std_hi_2 = x_std_element_hi.to(dtype=TORCH_FLOAT),
            #     data_temp.node_weight_2 = nw.to(dtype=TORCH_FLOAT),
            #     data_temp.L_2 = lengthscale_element.to(dtype=TORCH_FLOAT),
            #     data_temp.pos_norm_lo_2 = (pos_xlo_i/lengthscale_element).to(dtype=TORCH_FLOAT),
            #     data_temp.pos_norm_hi_2 = (pos_xhi_i/lengthscale_element).to(dtype=TORCH_FLOAT),
            #     data_temp.edge_index_lo_2 = edge_index[lev],
            #     data_temp.edge_index_hi_2 = edge_index_hi[lev],
            #     data_temp.central_element_mask_2 = central_element_mask[lev],
            #     data_temp.eid_2 = torch.tensor(i)

            # # for synchronizing across element boundaries
            # if n_element_neighbors > 0:
            #     batch = None
            #     edge_index_coin = get_edge_index_coincident(
            #             batch, (pos_xlo_i/lengthscale_element).to(dtype=TORCH_FLOAT), edge_index[lev])
            #     degree = utils.degree(edge_index_coin[1,:], num_nodes = pos_xlo_i.shape[0])
            #     degree += 1.
            #     if lev == 0:
            #         data_temp.edge_index_coin_0 = edge_index_coin
            #         data_temp.degree_0 = degree
            #     elif lev == 1:
            #         data_temp.edge_index_coin_1 = edge_index_coin
            #         data_temp.degree_1 = degree
            #     elif lev == 2:
            #         data_temp.edge_index_coin_2 = edge_index_coin
            #         data_temp.degree_2 = degree
            # else:
            #     if lev == 0:
            #         data_temp.edge_index_coin_0 = torch.Tensor([])
            #         data_temp.degree_0 = torch.Tensor([])
            #     elif lev == 1:
            #         data_temp.edge_index_coin_1 = torch.Tensor([])
            #         data_temp.degree_1 = torch.Tensor([])
            #     elif lev == 2:
            #         data_temp.edge_index_coin_2 = torch.Tensor([])
            #         data_temp.degree_2 = torch.Tensor([])

        # aggregate data
        # data_temp_agg = Data(
        #         x_lo = [data.x for data in data_temp_list],
        #         x_hi = [data.y for data in data_temp_list],
        #         x_mean_lo = [data.x_mean_lo for data in data_temp_list],
        #         x_std_lo = [data.x_std_lo for data in data_temp_list],
        #         x_mean_hi = [data.x_mean_hi for data in data_temp_list],
        #         x_std_hi = [data.x_std_hi for data in data_temp_list],
        #         node_weight = [data.node_weight for data in data_temp_list],
        #         L = [data.L for data in data_temp_list],
        #         pos_norm_lo = [data.pos_norm_lo for data in data_temp_list],
        #         pos_norm_hi = [data.pos_norm_hi for data in data_temp_list],
        #         edge_index_lo = [data.edge_index_lo for data in data_temp_list],
        #         edge_index_hi = [data.edge_index_hi for data in data_temp_list],
        #         central_element_mask = [data.central_element_mask for data in data_temp_list],
        #         eid = [data.eid for data in data_temp_list],
        #         )

        data_temp_agg = DataLoHiIncr()

        lev = 0
        data_temp_agg.x_lo_0 = data_temp_list[lev].x
        data_temp_agg.x_hi_0 = data_temp_list[lev].y
        data_temp_agg.x_mean_lo_0 = data_temp_list[lev].x_mean_lo
        data_temp_agg.x_std_lo_0 = data_temp_list[lev].x_std_lo
        data_temp_agg.x_mean_hi_0 = data_temp_list[lev].x_mean_hi
        data_temp_agg.x_std_hi_0 = data_temp_list[lev].x_std_hi
        data_temp_agg.node_weight_0 = data_temp_list[lev].node_weight
        data_temp_agg.L_0 = data_temp_list[lev].L
        data_temp_agg.pos_norm_lo_0 = data_temp_list[lev].pos_norm_lo
        data_temp_agg.pos_norm_hi_0 = data_temp_list[lev].pos_norm_hi
        data_temp_agg.edge_index_lo_0 = data_temp_list[lev].edge_index_lo
        data_temp_agg.edge_index_hi_0 = data_temp_list[lev].edge_index_hi
        data_temp_agg.central_element_mask_0 = data_temp_list[
            lev
        ].central_element_mask
        data_temp_agg.eid_0 = data_temp_list[lev].eid
        data_temp_agg.edge_index_coin_0 = data_temp_list[lev].edge_index_coin
        data_temp_agg.degree_0 = data_temp_list[lev].degree

        lev = 1
        data_temp_agg.x_lo_1 = data_temp_list[lev].x
        data_temp_agg.x_hi_1 = data_temp_list[lev].y
        data_temp_agg.x_mean_lo_1 = data_temp_list[lev].x_mean_lo
        data_temp_agg.x_std_lo_1 = data_temp_list[lev].x_std_lo
        data_temp_agg.x_mean_hi_1 = data_temp_list[lev].x_mean_hi
        data_temp_agg.x_std_hi_1 = data_temp_list[lev].x_std_hi
        data_temp_agg.node_weight_1 = data_temp_list[lev].node_weight
        data_temp_agg.L_1 = data_temp_list[lev].L
        data_temp_agg.pos_norm_lo_1 = data_temp_list[lev].pos_norm_lo
        data_temp_agg.pos_norm_hi_1 = data_temp_list[lev].pos_norm_hi
        data_temp_agg.edge_index_lo_1 = data_temp_list[lev].edge_index_lo
        data_temp_agg.edge_index_hi_1 = data_temp_list[lev].edge_index_hi
        data_temp_agg.central_element_mask_1 = data_temp_list[
            lev
        ].central_element_mask
        data_temp_agg.eid_1 = data_temp_list[lev].eid
        data_temp_agg.edge_index_coin_1 = data_temp_list[lev].edge_index_coin
        data_temp_agg.degree_1 = data_temp_list[lev].degree

        lev = 2
        data_temp_agg.x_lo_2 = data_temp_list[lev].x
        data_temp_agg.x_hi_2 = data_temp_list[lev].y
        data_temp_agg.x_mean_lo_2 = data_temp_list[lev].x_mean_lo
        data_temp_agg.x_std_lo_2 = data_temp_list[lev].x_std_lo
        data_temp_agg.x_mean_hi_2 = data_temp_list[lev].x_mean_hi
        data_temp_agg.x_std_hi_2 = data_temp_list[lev].x_std_hi
        data_temp_agg.node_weight_2 = data_temp_list[lev].node_weight
        data_temp_agg.L_2 = data_temp_list[lev].L
        data_temp_agg.pos_norm_lo_2 = data_temp_list[lev].pos_norm_lo
        data_temp_agg.pos_norm_hi_2 = data_temp_list[lev].pos_norm_hi
        data_temp_agg.edge_index_lo_2 = data_temp_list[lev].edge_index_lo
        data_temp_agg.edge_index_hi_2 = data_temp_list[lev].edge_index_hi
        data_temp_agg.central_element_mask_2 = data_temp_list[
            lev
        ].central_element_mask
        data_temp_agg.eid_2 = data_temp_list[lev].eid
        data_temp_agg.edge_index_coin_2 = data_temp_list[lev].edge_index_coin
        data_temp_agg.degree_2 = data_temp_list[lev].degree

        # print(i)
        # if i == 500:
        #    break
        data_temp_agg = data_temp_agg.to(device_for_loading)
        if idx_train_mask[i] == 1:
            data_train_list.append(data_temp_agg)
        else:
            data_valid_list.append(data_temp_agg)

    t_load = time.time() - t_load
    print("\ttook %g sec" % (t_load))
    return data_train_list, data_valid_list


def get_pygeom_dataset(
    data_x_path: str,
    data_y_path: str,
    edge_index_path: str,
    node_element_ids_path: str,
    global_ids_path: str,
    pos_path: str,
    edge_index_vertex_path: Optional[str] = None,
    device_for_loading: Optional[str] = "cpu",
    fraction_valid: Optional[float] = 0.1,
) -> Tuple[List, List, List, List]:
    t_load = time.time()

    print("Loading data and making pygeom dataset...")
    edge_index = np.loadtxt(edge_index_path, dtype=NP_INT).T
    node_element_ids = np.loadtxt(node_element_ids_path, dtype=NP_INT)
    global_ids = np.loadtxt(global_ids_path, dtype=NP_INT)
    pos = np.loadtxt(pos_path, dtype=NP_FLOAT)
    x = np.loadtxt(data_x_path, dtype=NP_FLOAT)
    y = np.loadtxt(data_y_path, dtype=NP_FLOAT)

    # Add extra edges if defined -- produces multiscale graph (adding P1 connectivity)
    if edge_index_vertex_path:
        print("Adding p1 connectivity...")
        print("\tEdge index shape before: ", edge_index.shape)
        edge_index_vertex = np.loadtxt(edge_index_vertex_path, dtype=NP_INT).T
        edge_index = np.concatenate((edge_index, edge_index_vertex), axis=1)
        print("\tEdge index shape after: ", edge_index.shape)

    # Retain only N_gll = Np*Ne elements
    N_gll = pos.shape[0]
    x = x[:N_gll, :]
    y = y[:N_gll, :]
    print("N_gll: ", N_gll)

    # Make tensor
    edge_index = torch.tensor(edge_index)
    node_element_ids = torch.tensor(node_element_ids)
    global_ids = torch.tensor(global_ids)
    pos = torch.tensor(pos, dtype=TORCH_FLOAT)
    x = torch.tensor(x, dtype=TORCH_FLOAT)
    y = torch.tensor(y, dtype=TORCH_FLOAT)

    # Coalesce
    edge_index = utils.coalesce(edge_index)

    # un-batch elements
    pos_tuple = utils.unbatch(pos, node_element_ids, dim=0)
    x_tuple = utils.unbatch(x, node_element_ids, dim=0)
    y_tuple = utils.unbatch(y, node_element_ids, dim=0)
    global_ids_tuple = utils.unbatch(global_ids, node_element_ids, dim=0)

    # get element-local stats -- vector per element
    x_mean_element, x_std_element = get_stats(torch.stack(x_tuple))
    # y_rms_element = get_rms(torch.stack(y_tuple))

    # get element lengthscale -- single scalar per element
    lengthscale_element = get_element_lengthscale(torch.stack(pos_tuple))

    # # add ln(rms) to the input data (x)
    # x_rms = x_rms_element[node_element_ids]
    # x_rms = torch.log(x_rms) # take log -- account for range
    # x = torch.cat([x, x_rms], dim=1)
    # x_tuple = utils.unbatch(x, node_element_ids, dim=0)

    # Get training data statistics
    n_samples = x.shape[0]
    x_mean = x.mean(dim=0, keepdim=False)
    x_var = x.var(dim=0, keepdim=False)
    y_mean = y.mean(dim=0, keepdim=False)
    y_var = y.var(dim=0, keepdim=False)
    data_mean = [x_mean, y_mean]
    data_var = [x_var, y_var]

    # Train / valid split
    n_snaps = len(pos_tuple)

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

    data_train_list = []
    for i in idx_train:
        n_nodes_i = x_tuple[i].shape[0]
        data_temp = Data(
            x=x_tuple[i],
            y=y_tuple[i],
            x_mean=x_mean_element[i]
            .unsqueeze(0)
            .repeat(x_tuple[i].shape[0], 1),
            x_std=x_std_element[i].unsqueeze(0).repeat(x_tuple[i].shape[0], 1),
            L=lengthscale_element[i],
            pos=pos_tuple[i],
            pos_norm=pos_tuple[i] / lengthscale_element[i],
            edge_index=edge_index,
            global_ids=global_ids_tuple[i],
        )
        data_temp = data_temp.to(device_for_loading)
        data_train_list.append(data_temp)

    data_valid_list = []
    for i in idx_valid:
        data_temp = Data(
            x=x_tuple[i],
            y=y_tuple[i],
            x_mean=x_mean_element[i]
            .unsqueeze(0)
            .repeat(x_tuple[i].shape[0], 1),
            x_std=x_std_element[i].unsqueeze(0).repeat(x_tuple[i].shape[0], 1),
            L=lengthscale_element[i],
            pos=pos_tuple[i],
            pos_norm=pos_tuple[i] / lengthscale_element[i],
            edge_index=edge_index,
            global_ids=global_ids_tuple[i],
        )
        data_temp = data_temp.to(device_for_loading)
        data_valid_list.append(data_temp)

    t_load = time.time() - t_load
    print("\ttook %g sec" % (t_load))
    return data_train_list, data_valid_list, data_mean, data_var, n_samples


def get_element_vertices(pos: Tensor) -> Tensor:
    """Given the position of the GLL points within the element,
       compute the polynomial order and return the vertices of the element.
    """
    Np1 = round(pos.shape[0]**(1/3))
    N = Np1 - 1
    print(f"N: {N}")
    print(f"Np1: {Np1}")
    
    if N == 1:
        return pos
    else:
        # write posd to file to 4 decimal places
        np.savetxt("pos.txt", pos.numpy(), fmt='%.4f')
        indices = torch.tensor([0,
                                Np1-1,
                                Np1**2-N-1,
                                Np1**2-1,
                                Np1**3-Np1**2-1,
                                Np1**3-Np1**2+N-1,
                                Np1**3-N-1,
                                Np1**3-1])
        print(f"indices: {indices}")
        elm_vtx = pos[indices, :]
        return elm_vtx
