"""
Convert neKRS output files into PyTorch Geometric dataset

Adapted from Shivam Barwey (ANL) at https://github.com/sbarwey/DDP_PyGeom.
"""

import os
import sys
import numpy as np
import argparse
import torch

from mpi4py import MPI

COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()
LOCAL_RANK = int(os.getenv("PALS_LOCAL_RANKID", 0))
LOCAL_SIZE = int(os.getenv("PALS_LOCAL_SIZE", 1))
HOST_NAME = MPI.Get_processor_name()

import dataprep.nekrs_graph_setup as ngs

import logging

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)


def write_dataset(args: argparse.Namespace):
    """
    Write the nekRS data to a PyTorch Geometric dataset.
    """
    # Take care of some initializations
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(1)
    device_for_loading = "cpu"
    n_train_local = 0
    n_valid_local = 0
    train_dataset_local = []
    test_dataset_local = []
    edge_index_path_lo = f"{args.case_path}/gnn_outputs_poly_{args.input_poly_order}/edge_index_element_local"
    edge_index_path_hi = f"{args.case_path}/gnn_outputs_poly_{args.target_poly_order}/edge_index_element_local"

    # Loop over snapshots (distributed across MPI ranks).
    n_files = len(args.target_snap_list)
    if RANK == 0:
        logger.info(
            f"MPI layout: SIZE={SIZE}, distributing {n_files} files across ranks"
        )
        if SIZE > n_files:
            logger.warning(
                f"SIZE={SIZE} exceeds number of files ({n_files}); ranks "
                f"{n_files}..{SIZE - 1} will be idle. Consider launching with "
                f"-n {n_files} to avoid wasted resources."
            )
    for i in range(RANK, n_files, SIZE):
        target_snap = args.target_snap_list[i]
        input_snap = args.input_snap_list[i]
        input_path = f"{args.case_path}/{input_snap}"
        target_path = f"{args.case_path}/{target_snap}"

        logger.info(
            f"[rank {RANK}] Loading data from {input_snap} and {target_snap}"
        )
        dataset = ngs.get_pygeom_dataset_lo_hi_pymech(
            data_xlo_path=input_path,
            data_xhi_path=target_path,
            edge_index_path_lo=edge_index_path_lo,
            edge_index_path_hi=edge_index_path_hi,
            device_for_loading=device_for_loading,
            fraction_valid=args.validation_fraction,
            n_element_neighbors=args.n_element_neighbors,
        )

        train_dataset_local += dataset["train"]["data"]
        test_dataset_local += dataset["valid"]["data"]
        n_train_local += dataset["train"]["num_samples"]
        n_valid_local += dataset["valid"]["num_samples"]

    # Gather per-rank datasets onto rank 0 for a single unified output.
    train_gathered = COMM.gather(train_dataset_local, root=0)
    test_gathered = COMM.gather(test_dataset_local, root=0)
    n_train = COMM.reduce(n_train_local, op=MPI.SUM, root=0)
    n_valid = COMM.reduce(n_valid_local, op=MPI.SUM, root=0)

    if RANK == 0:
        train_dataset = [d for chunk in train_gathered for d in chunk]
        test_dataset = [d for chunk in test_gathered for d in chunk]

        # Create output directory if it doesn't exist
        data_dir = args.case_path + f"/pt_datasets"
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        # try torch.save
        logger.info(f"Saving dataset to {data_dir}")
        logger.info(f"Number of training samples: {n_train}")
        logger.info(f"Number of validation samples: {n_valid}")
        torch.save(train_dataset, data_dir + f"/train_dataset.pt")
        torch.save(test_dataset, data_dir + f"/valid_dataset.pt")
        logger.info("Done!")


def main() -> None:
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case_path", type=str, help="Path to the case to load"
    )
    parser.add_argument(
        "--target_snap_list",
        type=str,
        nargs="+",
        help="List of target (high-resolution) snapshots to load",
    )
    parser.add_argument(
        "--input_snap_list",
        type=str,
        nargs="+",
        help="List of input (low-resolution) snapshots to load",
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
        "--seed", type=int, default=42, help="Random seed for torch and numpy"
    )
    parser.add_argument(
        "--validation_fraction",
        type=float,
        default=0.1,
        help="Fraction of data to use for validation",
    )
    parser.add_argument(
        "--n_element_neighbors",
        type=int,
        default=6,
        help="Number of element neighbors",
    )
    args = parser.parse_args()
    assert len(args.target_snap_list) == len(args.input_snap_list), (
        f"Number of target and input snapshots must be the same ({len(args.target_snap_list)} targets and {len(args.input_snap_list)} inputs)"
    )

    # Convert neKRS output files into PyTorch Geometric dataset
    write_dataset(args)


if __name__ == "__main__":
    main()
