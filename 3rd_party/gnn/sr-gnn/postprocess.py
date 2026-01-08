import numpy as np
import os
import sys
import argparse
from typing import Optional

import torch
import torch_geometric
import torch_geometric.utils as utils
import torch_geometric.nn as tgnn

from pymech.neksuite import readnek, writenek

from gnn import GNN_Element_Neighbor_Lo_Hi
import dataprep.nekrs_graph_setup as ngs

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)

try:
    WITH_CUDA = torch.cuda.is_available()
except:
    WITH_CUDA = False
    logger.warning("Found no CUDA devices")
    pass

try:
    WITH_XPU = torch.xpu.is_available()
except:
    WITH_XPU = False
    logger.warning("Found no XPU devices")
    pass

if WITH_CUDA:
    DEVICE = torch.device("cuda")
    N_DEVICES = torch.cuda.device_count()
    DEVICE_ID = 0
elif WITH_XPU:
    DEVICE = torch.device("xpu")
    N_DEVICES = torch.xpu.device_count()
    DEVICE_ID = 0
else:
    DEVICE = torch.device("cpu")
    DEVICE_ID = 0

TORCH_FLOAT = torch.float32


def count_parameters(mdl):
    return sum(p.numel() for p in mdl.parameters() if p.requires_grad)


def get_edge_index(
    edge_index_path: str, edge_index_vertex_path: Optional[str] = None
) -> torch.Tensor:
    edge_index = np.loadtxt(edge_index_path, dtype=np.int64).T
    if edge_index_vertex_path:
        logger.info("Adding p1 connectivity...")
        logger.info("\tEdge index shape before: ", edge_index.shape)
        edge_index_vertex = np.loadtxt(edge_index_vertex_path, dtype=np.int64).T
        edge_index = np.concatenate((edge_index, edge_index_vertex), axis=1)
        logger.info("\tEdge index shape after: ", edge_index.shape)
    edge_index = torch.tensor(edge_index)
    return edge_index


def main():
    """
    Inference script: save predicted flowfield into .f file.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case_path", type=str, help="Path to the case to load"
    )
    parser.add_argument(
        "--model_path", type=str, help="Path to the model to load"
    )
    parser.add_argument(
        "--input_snap_list",
        type=str,
        nargs="+",
        help="List of input (low-resolution) snapshots to load",
    )
    parser.add_argument(
        "--target_snap_list",
        type=str,
        nargs="+",
        help="List of target (high-resolution) snapshots to load",
    )
    parser.add_argument(
        "--output_name", type=str, help="Output name for the .f files"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for torch and numpy"
    )
    parser.add_argument(
        "--target_poly_order",
        type=int,
        default=7,
        help="Polynomial order of the target (high-resolution) field",
    )
    parser.add_argument(
        "--input_poly_order",
        type=int,
        default=1,
        help="Polynomial order of the input (low-resolution) field",
    )
    parser.add_argument(
        "--n_element_neighbors",
        type=int,
        default=6,
        help="Number of element neighbors",
    )
    parser.add_argument(
        "--use_residual",
        type=bool,
        default=True,
        help="Use residual mode for inference",
    )
    args = parser.parse_args()

    # ~~~~ Some initializations ~~~~ #
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_grad_enabled(False)

    # ~~~~ Load model state and instantiate model ~~~~ #
    a = torch.load(args.model_path, map_location="cpu", weights_only=False)
    input_dict = a["input_dict"]
    input_node_channels = input_dict["input_node_channels"]
    input_edge_channels_coarse = input_dict["input_edge_channels_coarse"]
    input_edge_channels_fine = input_dict["input_edge_channels_fine"]
    hidden_channels = input_dict["hidden_channels"]
    output_node_channels = input_dict["output_node_channels"]
    n_mlp_hidden_layers = input_dict["n_mlp_hidden_layers"]
    n_messagePassing_layers = input_dict["n_messagePassing_layers"]
    use_fine_messagePassing = input_dict["use_fine_messagePassing"]
    name = input_dict["name"]

    model = GNN_Element_Neighbor_Lo_Hi(
        input_node_channels=input_node_channels,
        input_edge_channels_coarse=input_edge_channels_coarse,
        input_edge_channels_fine=input_edge_channels_fine,
        hidden_channels=hidden_channels,
        output_node_channels=output_node_channels,
        n_mlp_hidden_layers=n_mlp_hidden_layers,
        n_messagePassing_layers=n_messagePassing_layers,
        use_fine_messagePassing=use_fine_messagePassing,
        device=DEVICE,
        name=name,
    )
    logger.info(
        f"Number of parameters in the SR-GNN model: {count_parameters(model)}"
    )
    model.load_state_dict(a["state_dict"])
    model.to(DEVICE)
    model.eval()

    # ~~~~ Load in edge index ~~~~ #
    edge_index_path_lo = f"{args.case_path}/gnn_outputs_poly_{args.input_poly_order}/edge_index_element_local"
    edge_index_path_hi = f"{args.case_path}/gnn_outputs_poly_{args.target_poly_order}/edge_index_element_local"
    edge_index_lo = get_edge_index(edge_index_path_lo)
    edge_index_hi = get_edge_index(edge_index_path_hi)
    logger.info("Loaded edge index for input and target graphs")

    # Get full edge index
    edge_index = edge_index_lo
    n_nodes_per_element = edge_index.max() + 1
    if args.n_element_neighbors > 0:
        node_max_per_element = edge_index.max()
        n_edges_per_element = edge_index.shape[1]
        edge_index_full = torch.zeros(
            (2, n_edges_per_element * (args.n_element_neighbors + 1)),
            dtype=edge_index.dtype,
        )
        edge_index_full[:, :n_edges_per_element] = edge_index
        for i in range(1, args.n_element_neighbors + 1):
            start = n_edges_per_element * i
            end = n_edges_per_element * (i + 1)
            edge_index_full[:, start:end] = (
                edge_index + (node_max_per_element + 1) * i
            )
        edge_index = edge_index_full

    # ~~~~ For each input snapshot, perform model inference ~~~~ #
    for isnap in range(len(args.input_snap_list)):
        time_id = args.input_snap_list[isnap].split(".f")[-1]

        target_snap = args.target_snap_list[isnap]
        input_snap = args.input_snap_list[isnap]
        input_path = f"{args.case_path}/{input_snap}"
        target_path = f"{args.case_path}/{target_snap}"
        xlo_field = readnek(input_path)
        xhi_field = readnek(target_path)
        xhi_field_pred = readnek(target_path)
        xhi_field_error = readnek(target_path)

        n_snaps = len(xlo_field.elem)

        # Get the element neighborhoods
        if args.n_element_neighbors > 0:
            Nelements = len(xlo_field.elem)
            pos_c = torch.zeros((Nelements, 3))
            for i in range(Nelements):
                pos_c[i] = torch.tensor(xlo_field.elem[i].centroid)
            edge_index_c = tgnn.knn_graph(x=pos_c, k=args.n_element_neighbors)

        # Get the element masks
        central_element_mask = torch.concat((
            torch.ones((n_nodes_per_element), dtype=torch.int64),
            torch.zeros(
                (n_nodes_per_element * args.n_element_neighbors),
                dtype=torch.int64,
            ),
        ))
        central_element_mask = central_element_mask.to(torch.bool)

        # Model inference
        prediction_path = (
            args.case_path + f"/predictions/{model.get_save_header()}"
        )
        with torch.no_grad():
            for i in range(n_snaps):
                print(f"Evaluating element {i}/{n_snaps}")

                pos_xlo_i = (
                    torch.tensor(xlo_field.elem[i].pos).reshape((3, -1)).T
                )  # pygeom pos format -- [N, 3]
                vel_xlo_i = (
                    torch.tensor(xlo_field.elem[i].vel).reshape((3, -1)).T
                )
                pos_xhi_i = (
                    torch.tensor(xhi_field.elem[i].pos).reshape((3, -1)).T
                )  # pygeom pos format -- [N, 3]
                vel_xhi_i = (
                    torch.tensor(xhi_field.elem[i].vel).reshape((3, -1)).T
                )

                x_gll = xhi_field.elem[i].pos[0, 0, 0, :]
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
                    print(
                        f"Relative error in positions exceeds 0.01% in element i={i}."
                    )
                    sys.exit()
                if pos_xlo_i.max() == 0.0 and pos_xlo_i.min() == 0.0:
                    print(f"Node positions are not stored in {input_path}.")
                    sys.exit()
                if pos_xhi_i.max() == 0.0 and pos_xhi_i.min() == 0.0:
                    print(f"Node positions are not stored in {target_path}.")
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
                # nw = torch.ones((vel_xhi_i.shape[0], 1)) * node_weight

                # Get the element neighbors for the input
                if args.n_element_neighbors > 0:
                    send = edge_index_c[0, :]
                    recv = edge_index_c[1, :]
                    nbrs = send[recv == i]

                    pos_x_full = [pos_xlo_i]
                    vel_x_full = [vel_xlo_i]
                    for j in nbrs:
                        pos_x_full.append(
                            torch
                            .tensor(xlo_field.elem[j].pos)
                            .reshape((3, -1))
                            .T
                        )
                        vel_x_full.append(
                            torch
                            .tensor(xlo_field.elem[j].vel)
                            .reshape((3, -1))
                            .T
                        )
                    pos_x_full = torch.concat(pos_x_full)
                    vel_x_full = torch.concat(vel_x_full)

                    # reset pos
                    pos_xlo_i = pos_x_full
                    vel_xlo_i = vel_x_full

                # create data
                data = ngs.DataLoHi(
                    x=vel_xlo_i.to(dtype=TORCH_FLOAT),
                    y=vel_xhi_i.to(dtype=TORCH_FLOAT),
                    x_mean_lo=x_mean_element_lo.to(dtype=TORCH_FLOAT),
                    x_std_lo=x_std_element_lo.to(dtype=TORCH_FLOAT),
                    x_mean_hi=x_mean_element_hi.to(dtype=TORCH_FLOAT),
                    x_std_hi=x_std_element_hi.to(dtype=TORCH_FLOAT),
                    # node_weight = nw.to(dtype=TORCH_FLOAT),
                    L=lengthscale_element.to(dtype=TORCH_FLOAT),
                    pos_norm_lo=(pos_xlo_i / lengthscale_element).to(
                        dtype=TORCH_FLOAT
                    ),
                    pos_norm_hi=(pos_xhi_i / lengthscale_element).to(
                        dtype=TORCH_FLOAT
                    ),
                    edge_index_lo=edge_index,
                    edge_index_hi=edge_index_hi,
                    central_element_mask=central_element_mask,
                    eid=torch.tensor(i),
                )

                # for synchronizing across element boundaries
                if args.n_element_neighbors > 0:
                    batch = None
                    edge_index_coin = ngs.get_edge_index_coincident(
                        batch, data.pos_norm_lo, data.edge_index_lo
                    )
                    degree = utils.degree(
                        edge_index_coin[1, :],
                        num_nodes=data.pos_norm_lo.shape[0],
                    )
                    degree += 1.0
                    data.edge_index_coin = edge_index_coin
                    data.degree = degree
                else:
                    data.edge_index_coin = None
                    data.degree = None

                data = data.to(DEVICE)

                # Model evaluation
                # 1) Preprocessing: scale input
                eps = 1e-10
                x_scaled = (data.x - data.x_mean_lo) / (data.x_std_lo + eps)

                # 2) Evaluate model
                out_gnn = model(
                    x=x_scaled,
                    mask=data.central_element_mask,
                    edge_index_lo=data.edge_index_lo,
                    edge_index_hi=data.edge_index_hi,
                    pos_lo=data.pos_norm_lo,
                    pos_hi=data.pos_norm_hi,
                    # batch_lo = data.x_batch,
                    # batch_hi = data.y_batch,
                    edge_index_coin=data.edge_index_coin
                    if args.n_element_neighbors > 0
                    else None,
                    degree=data.degree
                    if args.n_element_neighbors > 0
                    else None,
                )

                # 3) set the target
                if args.use_residual:
                    mask = data.central_element_mask
                    data.x_batch = data.edge_index_lo.new_zeros(
                        data.pos_norm_lo.size(0)
                    )
                    data.y_batch = data.edge_index_hi.new_zeros(
                        data.pos_norm_hi.size(0)
                    )
                    x_interp = tgnn.unpool.knn_interpolate(
                        x=data.x[mask, :].cpu(),
                        pos_x=data.pos_norm_lo[mask, :].cpu(),
                        pos_y=data.pos_norm_hi.cpu(),
                        batch_x=data.x_batch[mask].cpu(),
                        batch_y=data.y_batch.cpu(),
                        k=8,
                    )
                    x_interp = x_interp.to(DEVICE)
                    # target = (data.y - x_interp)/(data.x_std_hi + eps)
                    # gnn = (data.y - x_interp)/(data.x_std_hi + eps)
                    # gnn * (data.x_std_hi + eps) = (data.y - x_interp)
                    # data.y = x_interp + gnn * (data.x_std_hi + eps)
                    y_pred = x_interp + out_gnn * (data.x_std_hi + eps)
                else:
                    # target = (data.y - data.x_mean_hi)/(data.x_std_hi + eps)
                    # gnn = (data.y - data.x_mean_hi)/(data.x_std_hi + eps)
                    # gnn * (data.x_std_hi + eps) = data.y - data.x_mean_hi
                    # data.y = data.x_mean_hi + gnn * (data.x_std_hi + eps)
                    y_pred = data.x_mean_hi + out_gnn * (data.x_std_hi + eps)

                # Making the .f file
                # Re-shape the prediction, convert back to fp64 numpy
                y_pred = y_pred.cpu()
                orig_shape = xhi_field.elem[i].vel.shape
                y_pred_rs = (
                    torch
                    .reshape(y_pred.T, orig_shape)
                    .to(dtype=torch.float64)
                    .numpy()
                )
                target = data.y.cpu()
                target_rs = (
                    torch
                    .reshape(target.T, orig_shape)
                    .to(dtype=torch.float64)
                    .numpy()
                )

                # Place prediction back in the snapshot data
                xhi_field_pred.elem[i].vel[:, :, :, :] = y_pred_rs

                # Place error back in snapshot data
                xhi_field_error.elem[i].vel[:, :, :, :] = target_rs - y_pred_rs

                # Sanity check to make sure reshape is correct.
                # target_orig = xhi_field.elem[i].vel
                # err_sanity = target_orig - target_rs

            # Write
            logger.info(f"Writing {time_id} prediction to {prediction_path}...")
            if not os.path.exists(prediction_path):
                os.makedirs(prediction_path)
                logger.info(f"Directory '{prediction_path}' created.")
            writenek(
                prediction_path + f"/{args.output_name}_pred0.f{time_id}",
                xhi_field_pred,
            )
            writenek(
                prediction_path + f"/{args.output_name}_error0.f{time_id}",
                xhi_field_error,
            )
            logger.info("Done!")


if __name__ == "__main__":
    main()
