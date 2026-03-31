import argparse
from mpi4py import MPI
from parrsb import Mesh
import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process command line arguments."
    )
    parser.add_argument(
        "-p",
        "--poly",
        type=int,
        required=True,
        help="Specify the polynomial order",
    )
    parser.add_argument(
        "-c",
        "--case",
        type=str,
        required=True,
        help="Specify the mesh file path (.re2 file, without the extension)",
    )
    parser.add_argument(
        "-l",
        "--log",
        type=str,
        default="info",
        choices=["debug", "info", "warning", "error"],
        required=False,
        help="Specify the log level (default: info)",
    )
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    m = Mesh(args.case, comm)
    partitions = m.partition()
