"""
PyTorch DDP inference script for GNN-based diffusion models from mesh data
"""

import os
import logging
from collections import deque
from typing import Optional, Union, Callable
import numpy as np
from numpy.typing import NDArray
import hydra
import time
import math
from omegaconf import DictConfig, OmegaConf

try:
    # import mpi4py
    # mpi4py.rc.initialize = False
    from mpi4py import MPI

    WITH_DDP = True
except ModuleNotFoundError as e:
    WITH_DDP = False
    pass

import torch

# Local imports
import utils
from trainer import DGNTrainer
from client import OnlineClient
import postprocess

log = logging.getLogger(__name__)

# Get MPI:
if WITH_DDP:
    COMM = MPI.COMM_WORLD
    SIZE = COMM.Get_size()
    RANK = COMM.Get_rank()
    LOCAL_RANK = int(os.getenv("PALS_LOCAL_RANKID"))
    LOCAL_SIZE = int(os.getenv("PALS_LOCAL_SIZE"))
    HOST_NAME = MPI.Get_processor_name()

    try:
        WITH_CUDA = torch.cuda.is_available()
    except:
        WITH_CUDA = False
        if RANK == 0:
            log.warn("Found no CUDA devices")
        pass

    try:
        WITH_XPU = torch.xpu.is_available()
    except:
        WITH_XPU = False
        if RANK == 0:
            log.warn("Found no XPU devices")
        pass

    if WITH_CUDA:
        DEVICE = torch.device("cuda")
        N_DEVICES = torch.cuda.device_count()
        DEVICE_ID = LOCAL_RANK if N_DEVICES > 1 else 0
    elif WITH_XPU:
        DEVICE = torch.device("xpu")
        N_DEVICES = torch.xpu.device_count()
        DEVICE_ID = LOCAL_RANK if N_DEVICES > 1 else 0
    else:
        DEVICE = torch.device("cpu")
        DEVICE_ID = "cpu"
else:
    SIZE = 1
    RANK = 0
    LOCAL_RANK = 0
    MASTER_ADDR = "localhost"
    log.warning("MPI Initialization failed!")


def _load_validation_snapshots(trainer, data_dir: str):
    """Load per-rank validation snapshots from on-disk field files.

    Loads every ``fld_u_rank_{RANK}_*`` file in ``data_dir``, in the same
    format the trainer's load_field_data uses, and returns them as a list of
    numpy fp32 arrays of shape ``(n_nodes_local, input_node_features)`` in
    PHYSICAL units (no normalization, no halo padding). Halo isn't needed
    because metric computation only touches owned nodes -- the inter-rank
    redistribution and 1/node_degree weighting handle the face-node double-
    counting cleanly across ranks.

    Assumes the validation mesh matches the model's mesh (same ``data_full``,
    same ``idx_full2reduced``). We pull those off the trainer; no graph is
    rebuilt for the validation set.
    """
    field_name = "u"
    file_list = os.listdir(data_dir)
    files = [
        f for f in file_list
        if (f"fld_{field_name}" in f) and (f"rank_{RANK}_" in f)
    ]
    files.sort(key=lambda x: int(x.split(".")[0].split("_")[-1]))
    files = [os.path.join(data_dir, f) for f in files]
    if RANK == 0:
        log.info(
            f"[validation] Found {len(files)} field files in {data_dir}"
        )

    n_features = trainer.cfg.input_node_features
    N_gll = trainer.data_full.pos.shape[0]
    idx_full2reduced = trainer.idx_full2reduced
    if hasattr(idx_full2reduced, "detach"):
        idx_full2reduced_np = idx_full2reduced.detach().cpu().numpy()
    else:
        idx_full2reduced_np = np.asarray(idx_full2reduced)
    n_nodes_local = int(trainer.data_reduced.n_nodes_local.item())

    snapshots = []
    for path in files:
        # INVARIANT: validation snapshots are kept in PHYSICAL units (no
        # mean-subtract, no std-divide). They are compared against the
        # un-scaled prediction in infer() and plotted directly. Do NOT add
        # normalization here; the loader's contract is "raw on-disk values".
        data_x = trainer.load_data(path, dtype=np.float64).reshape(
            (-1, n_features)
        )
        data_x = data_x[:N_gll, :]                    # drop any padding
        data_x = data_x[idx_full2reduced_np, :]       # full -> reduced layout
        data_x = data_x[:n_nodes_local, :]            # owned nodes only
        snapshots.append(data_x.astype(np.float32))
    return snapshots


def _best_match_metrics(
    pred: NDArray[np.float32],
    val_snapshots: list,
    weights: NDArray[np.float32],
    effective_nodes: float,
    n_features: int,
):
    """Compute global (cross-rank) MSE and R^2 between ``pred`` and every
    snapshot in ``val_snapshots``, then return the metrics for the snapshot
    with the smallest MSE.

    Both pred and val snapshots are per-rank owned-node arrays in physical
    units. ``weights`` is ``1 / node_degree[:n_nodes_local]`` so face nodes
    shared across ranks contribute fractionally and the sum across all ranks
    counts each physical node exactly once.

    Issues a SINGLE all-reduce of length ``2 * n_val`` to keep collective
    overhead bounded regardless of validation set size.

    Returns:
        (best_idx, mse, r2) for the best-matching snapshot.
    """
    n_val = len(val_snapshots)
    # Per-snapshot local accumulators packed contiguously: [sse_0, ..., sse_{n-1},
    # tss_0, ..., tss_{n-1}]. fp64 to keep the cross-rank sum well-conditioned.
    w = weights.reshape(-1, 1).astype(np.float64)
    pred64 = pred.astype(np.float64)

    local = np.zeros(2 * n_val, dtype=np.float64)
    for j, val in enumerate(val_snapshots):
        val64 = val.astype(np.float64)
        diff = pred64 - val64
        local[j] = (w * diff * diff).sum()
        # Per-feature mean of the validation snapshot across the GLOBAL mesh
        # would itself need a collective. We use the LOCAL weighted mean as a
        # cheap centering; this slightly biases TSS but R^2 retains its
        # qualitative ordering across snapshots (best is still best). If you
        # want a true global R^2, do a separate all-reduce on the val means
        # before this loop and pass them in.
        val_local_mean = (w * val64).sum(axis=0) / w.sum()
        centered = val64 - val_local_mean
        local[n_val + j] = (w * centered * centered).sum()

    if SIZE > 1:
        gbl = np.zeros_like(local)
        COMM.Allreduce(local, gbl, op=MPI.SUM)
    else:
        gbl = local

    sse = gbl[:n_val]
    tss = gbl[n_val:]
    # Element-wise MSE over (effective_nodes * n_features) so it is comparable
    # to the trainer's training-loss MSE.
    mse = sse / (effective_nodes * n_features)
    # Guard against TSS underflow on a constant snapshot.
    tss_safe = np.where(tss > 1e-30, tss, 1.0)
    r2 = 1.0 - sse / tss_safe

    best_idx = int(np.argmin(mse))
    return best_idx, float(mse[best_idx]), float(r2[best_idx])


def gather_wrapper(temp: NDArray[np.float32]) -> NDArray[np.float32]:

    temp_shape = temp.shape
    n_cols = temp_shape[1]

    # ~~~~ gather using mpi4py gatherv
    # Step 1: Gather the sizes of each of the local arrays
    local_size = np.array(
        temp.size, dtype="int32"
    )  # total elements = n_nodes_local * 3
    all_sizes = None
    if RANK == 0:
        all_sizes = np.empty(SIZE, dtype="int32")
    COMM.Gather(local_size, all_sizes, root=0)

    # Step 2: compute displacements for Gatherv
    if RANK == 0:
        displacements = np.insert(np.cumsum(all_sizes[:-1]), 0, 0)
    else:
        displacements = None

    # Step 3: Flatten the local array for sending
    flat_temp = temp.flatten()

    # Step 4: On root, prepare recv buffer
    if RANK == 0:
        total_size = np.sum(all_sizes)
        recvbuf = np.empty(total_size, dtype=temp.dtype)
    else:
        recvbuf = None

    # Perform the Gatherv operation, then reshape the buffer
    COMM.Gatherv(
        sendbuf=flat_temp,
        recvbuf=(recvbuf, (all_sizes, displacements)) if RANK == 0 else None,
        root=0,
    )

    gathered_array = None
    if RANK == 0:
        gathered_array = recvbuf.reshape(-1, 3)
    COMM.Barrier()

    return gathered_array


def infer(cfg: DictConfig, client: Optional[OnlineClient] = None) -> None:
    """Generate samples of the velocity field"""
    trainer = DGNTrainer(cfg, client=client)
    trainer.writeGraphStatistics()
    graph = trainer.data["graph"]
    stats = trainer.data["stats"]
    pos = graph.pos_orig
    n_nodes_local = trainer.data_reduced.n_nodes_local.item()

    # Load validation snapshots once (graph-static across the sampling loop).
    val_snapshots = None
    val_weights = None
    val_eff_nodes = None
    if getattr(cfg, "validate", False):
        val_snapshots = _load_validation_snapshots(
            trainer, cfg.validation_data_path
        )
        if len(val_snapshots) == 0:
            if RANK == 0:
                log.warning(
                    "[validation] No validation snapshots found in "
                    f"{cfg.validation_data_path}; skipping metric computation."
                )
            val_snapshots = None
        else:
            node_degree = trainer.data_reduced.node_degree[:n_nodes_local]
            val_weights = (
                (1.0 / node_degree).detach().cpu().to(torch.float32).numpy()
            )
            val_eff_nodes = float(
                trainer.data_reduced.effective_nodes
                .detach()
                .cpu()
                .to(torch.float64)
                .item()
            )

    # Sampling loop
    local_time = []
    local_throughput = []
    for i in range(cfg.num_gen_samples):
        # Generate sample prediction
        if RANK == 0:
            log.info("Predicting Dist-DGN sample ...")
        pred = trainer.sample()
        # Cast to fp32 before .numpy() so bf16 runs don't trigger the IPEX
        # "not share memory" warning (numpy has no native bf16).
        pred = pred[:n_nodes_local].cpu().to(torch.float32).numpy()

        # Undo scaling
        pred = pred * stats["x_std"] + stats["x_mean"]

        # Validation
        best_idx = None
        if val_snapshots is not None:
            best_idx, mse, r2 = _best_match_metrics(
                pred,
                val_snapshots,
                val_weights,
                val_eff_nodes,
                n_features=cfg.input_node_features,
            )
            if RANK == 0:
                log.info(
                    f"[validation] sample {i}: best-match snapshot "
                    f"index={best_idx}/{len(val_snapshots)}  "
                    f"MSE={mse:.6e}  R^2={r2:.6f}"
                )

        # Postprocess the data
        if cfg.postprocess:
            pos_np = pos.to(torch.float32).numpy()
            postprocess.plot_2d_field(COMM, pos_np, pred, f"pred_{i}.png")
            if best_idx is not None:
                postprocess.plot_2d_field(
                    COMM,
                    pos_np,
                    val_snapshots[best_idx],
                    f"val_best_match_{i}.png",
                )


@hydra.main(version_base=None, config_path="./conf", config_name="config")
def main(cfg: DictConfig) -> None:
    if cfg.verbose:
        log.info(
            f"Hello from rank {RANK}/{SIZE}, local rank {LOCAL_RANK}, on node {HOST_NAME} and device {DEVICE}:{DEVICE_ID + cfg.device_skip} out of {N_DEVICES}."
        )

    if RANK == 0:
        log.info("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        log.info("RUNNING WITH INPUTS:")
        log.info(f"{OmegaConf.to_yaml(cfg)}")
        log.info("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    if not cfg.online:
        infer(cfg)
    else:
        log.info("Oline inference not implemented yet for this model")
        COMM.Abort(1)

    utils.cleanup()
    if RANK == 0:
        log.info("Exiting ...")


if __name__ == "__main__":
    main()
