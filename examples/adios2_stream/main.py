"""
PyTorch DDP training script for GNN-based surrogates from mesh data
"""
import sys
import os
import logging
from collections import deque
from typing import Optional, Union, Callable
import numpy as np
import hydra
import time
import math
from omegaconf import DictConfig, OmegaConf

try:
    import mpi4py.rc
    mpi4py.rc.initialize = False
    from mpi4py import MPI
except ModuleNotFoundError as e:
    sys.exit('MPI is required! Please install MPI and try again.')

import torch

# Local imports
# This import causes the adios client to not initialize 
# or hang when loading the .bp file.
# The issue is importing torch_geometric
from trainer import Trainer 
from client import OnlineClient

log = logging.getLogger(__name__)

# Get MPI:
MPI.Init()
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
    if RANK == 0: log.warning('Found no CUDA devices')
    pass

try:
    WITH_XPU = torch.xpu.is_available()
except:
    WITH_XPU = False
    if RANK == 0: log.warning('Found no XPU devices')
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


def train(cfg: DictConfig,
          client: Optional[OnlineClient] = None
    ) -> None:
    # This import here makes the adios2 client initialize and .bp file to be read
    # but makes adios2 hang when opening the SST stream
    #_root_logger = logging.getLogger()
    #_prev_level = _root_logger.level
    #_root_logger.setLevel(logging.WARNING)
    #from trainer import Trainer
    #_root_logger.setLevel(_prev_level)
    trainer = Trainer(cfg, COMM, client=client)
    N = trainer.N
    
    ## Training loop: 
    stream_times = np.zeros(cfg.phase1_steps)
    iteration = 0
    while True:
        stream_time = trainer.train_step()
        if RANK == 0: print(f"[Trainer] Loaded data for step {iteration} in {stream_time:.4f} seconds", flush=True)
        stream_times[iteration] = stream_time
        iteration += 1

        # Break loop over dataloader
        if iteration >= cfg.phase1_steps:
            break

    # Tell simulation to exit
    if cfg.online:
        if RANK == 0: print(f"[Trainer] Telling simulation to quit ...", flush=True)
        client.stop_nekRS()
    COMM.Barrier()

    # Clean up
    trainer.cleanup()

    # Compute average stream time across all ranks
    global_stream_times = np.zeros(cfg.phase1_steps*SIZE)
    COMM.Allgather(stream_times, global_stream_times)
    avg_stream_times = np.mean(global_stream_times)

    # Print performance stats
    time.sleep(5)
    if RANK == 0:
        print("\n=== Communication Performance Summary ===")
        data_size_gb = N * 8 / 1e9
        recv_bw = N * 8 / avg_stream_times / 1e9
        print(f"Data size per message: {data_size_gb.item():.4e} GB")
        print(f"Total iterations: {cfg.phase1_steps}")
        print(f"Average receive time: {avg_stream_times:.6f} seconds")
        print(f"Average receive bandwidth: {recv_bw.item():.6f} GB/s",flush=True)


@hydra.main(version_base=None, config_path='./conf', config_name='config')
def main(cfg: DictConfig) -> None:
    if RANK == 0: print(f'[Trainer] Running with {SIZE} MPI ranks on head node {HOST_NAME}', flush=True)

    if not cfg.online:
        print('[Trainer] This example is only for online training!', flush=True)
        COMM.Abort(1)
    else:
        client = OnlineClient(cfg, COMM)
        COMM.Barrier()
        if RANK == 0: print('[Trainer] Initialized Online Client!', flush=True)
        train(cfg, client)

    if RANK == 0:
        print('[Trainer] Exiting ...', flush=True)


if __name__ == '__main__':
    main()
