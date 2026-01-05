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
    #import mpi4py
    #mpi4py.rc.initialize = False
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
        if RANK == 0: log.warn('Found no CUDA devices')
        pass

    try:
        WITH_XPU = torch.xpu.is_available()
    except:
        WITH_XPU = False
        if RANK == 0: log.warn('Found no XPU devices')
        pass

    if WITH_CUDA:
        DEVICE = torch.device('cuda')
        N_DEVICES = torch.cuda.device_count()
        DEVICE_ID = LOCAL_RANK if N_DEVICES>1 else 0
    elif WITH_XPU:
        DEVICE = torch.device('xpu')
        N_DEVICES = torch.xpu.device_count()
        DEVICE_ID = LOCAL_RANK if N_DEVICES>1 else 0
    else:
        DEVICE = torch.device('cpu')
        DEVICE_ID = 'cpu'
else:
    SIZE = 1
    RANK = 0
    LOCAL_RANK = 0
    MASTER_ADDR = 'localhost'
    log.warning('MPI Initialization failed!')


def gather_wrapper(temp: NDArray[np.float32]) -> NDArray[np.float32]:

    temp_shape = temp.shape
    n_cols = temp_shape[1]

    # ~~~~ gather using mpi4py gatherv
    # Step 1: Gather the sizes of each of the local arrays
    local_size = np.array(temp.size, dtype='int32')  # total elements = n_nodes_local * 3
    all_sizes = None
    if RANK == 0:
        all_sizes = np.empty(SIZE, dtype='int32')
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
        root=0
    )

    gathered_array = None
    if RANK == 0:
        gathered_array = recvbuf.reshape(-1, 3)
    COMM.Barrier()

    return gathered_array


def infer(cfg: DictConfig,
          client: Optional[OnlineClient] = None
    ) -> None:
    """Generate samples of the velocity field
    """
    trainer = DGNTrainer(cfg, client=client)
    trainer.writeGraphStatistics()
    graph = trainer.data['graph']
    stats = trainer.data['stats']
    pos = graph.pos_orig
    n_nodes_local = trainer.data_reduced.n_nodes_local.item()

    # Sampling loop
    local_time = []
    local_throughput = []
    n_features = cfg.input_node_features
    for i in range(cfg.num_gen_samples):
        # Initialize new random field and generate sample prediction
        field_r = torch.randn(pos.size(0), n_features, dtype=trainer.torch_dtype)
        pred = trainer.sample(field_r)
        pred = pred.cpu().numpy()
        
        # Undo scaling
        pred = pred * stats['x_std'] + stats['x_mean']

        # Postprocess the data
        if cfg.postprocess:
            postprocess.plot_2d_field(COMM, pos, pred, f"pred_{i}.png")


@hydra.main(version_base=None, config_path='./conf', config_name='config')
def main(cfg: DictConfig) -> None:
    if cfg.verbose:
        log.info(f'Hello from rank {RANK}/{SIZE}, local rank {LOCAL_RANK}, on node {HOST_NAME} and device {DEVICE}:{DEVICE_ID+cfg.device_skip} out of {N_DEVICES}.')
    
    if RANK == 0:
        log.info('\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        log.info('RUNNING WITH INPUTS:')
        log.info(f'{OmegaConf.to_yaml(cfg)}') 
        log.info('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

    if not cfg.online:
        infer(cfg)
    else:
        log.info('Oline inference not implemented yet for this model')
        COMM.Abort(1)
    
    utils.cleanup()
    if RANK == 0:
        log.info('Exiting ...')


if __name__ == '__main__':
    main()
