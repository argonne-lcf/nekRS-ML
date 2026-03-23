import argparse
import mpi4py
from mpi4py import MPI
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

    p = args.poly
    re2_path = args.case if args.case[-4:] != ".re2" else args.case[:-4]
    log_level = args.log

    comm = MPI.COMM_WORLD
