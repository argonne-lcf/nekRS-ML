"""
Convert neKRS output files into PyTorch Geometric dataset

Adapted from Shivam Barwey (ANL) at https://github.com/sbarwey/DDP_PyGeom.
"""
import os
import numpy as np
import argparse
import torch

import dataprep.nekrs_graph_setup as ngs

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', force=True)
logger = logging.getLogger(__name__)


def write_dataset(args: argparse.Namespace):
    """
    Write the nekRS data to a PyTorch Geometric dataset.
    """
    # Take care of some initializations
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device_for_loading = 'cpu'
    n_train = 0
    n_valid = 0
    train_dataset = []
    test_dataset = [] 
    edge_index_path_lo = f"{args.case_path}/gnn_outputs_poly_{args.input_poly_order}/edge_index_element_local"
    edge_index_path_hi = f"{args.case_path}/gnn_outputs_poly_{args.target_poly_order}/edge_index_element_local"

    # Loop over snapshots (could parallelize with MPI if needed)
    for i in range(len(args.target_snap_list)):
        target_snap = args.target_snap_list[i]
        input_snap = args.input_snap_list[i]
        input_path = f"{args.case_path}/{input_snap}" 
        target_path = f"{args.case_path}/{target_snap}"

        logger.info(f"Loading data from {input_snap} and {target_snap}")
        dataset = ngs.get_pygeom_dataset_lo_hi_pymech(
                                data_xlo_path = input_path, 
                                data_xhi_path = target_path,
                                edge_index_path_lo = edge_index_path_lo,
                                edge_index_path_hi = edge_index_path_hi,
                                device_for_loading = device_for_loading,
                                fraction_valid = args.validation_fraction,
                                n_element_neighbors = args.n_element_neighbors)

        train_dataset += dataset["train"]["data"]
        test_dataset += dataset["valid"]["data"]
        n_train += dataset["train"]["num_samples"]
        n_valid += dataset["valid"]["num_samples"]

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
    parser.add_argument('--case_path', type=str, help='Path to the case to load')
    parser.add_argument('--target_snap_list', type=str, nargs='+', help='List of target (high-resolution) snapshots to load')
    parser.add_argument('--input_snap_list', type=str, nargs='+', help='List of input (low-resolution) snapshots to load')
    parser.add_argument('--target_poly_order', type=int, default=7, help='Polynomial order of the target (high-resolution) field')
    parser.add_argument('--input_poly_order', type=int, default=1, help='Polynomial order of the input (low-resolution) field')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for torch and numpy')
    parser.add_argument('--validation_fraction', type=float, default=0.1, help='Fraction of data to use for validation')
    parser.add_argument('--n_element_neighbors', type=int, default=6, help='Number of element neighbors')
    args = parser.parse_args()
    assert len(args.target_snap_list) == len(args.input_snap_list), \
        f"Number of target and input snapshots must be the same ({len(args.target_snap_list)} targets and {len(args.input_snap_list)} inputs)"
    
    # Convert neKRS output files into PyTorch Geometric dataset
    write_dataset(args)

if __name__ == '__main__':
    main()
