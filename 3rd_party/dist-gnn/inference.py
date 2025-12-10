"""
PyTorch DDP inference script for GNN-based surrogates from mesh data
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
from trainer import Trainer
from client import OnlineClient

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

    ## pytorch will look for these
    #os.environ['RANK'] = str(RANK)
    #os.environ['WORLD_SIZE'] = str(SIZE)
    ## -----------------------------------------------------------
    ## NOTE: Get the hostname of the master node, and broadcast
    ## it to all other nodes It will want the master address too,
    ## which we'll broadcast:
    ## -----------------------------------------------------------
    #MASTER_ADDR = socket.gethostname() if RANK == 0 else None
    #MASTER_ADDR = MPI.COMM_WORLD.bcast(MASTER_ADDR, root=0)
    #os.environ['MASTER_ADDR'] = MASTER_ADDR
    #os.environ['MASTER_PORT'] = str(2345)

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
    #log.info(f"[RANK {RANK}] -- STEP 1: all_sizes = {all_sizes}")

    # Step 2: compute displacements for Gatherv 
    if RANK == 0:
        displacements = np.insert(np.cumsum(all_sizes[:-1]), 0, 0)
    else:
        displacements = None
    #log.info(f"[RANK {RANK}] -- STEP 2: displacements = {displacements}")

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
        # reshape all at once: 
        gathered_array = recvbuf.reshape(-1, 3)

        # # reshape rank-wise, then concatenate 
        # gathered_arrays = []
        # start = 0
        # for proc_size in all_sizes:
        #     proc_rows = proc_size // n_cols
        #     proc_data = recvbuf[start:start+proc_size].reshape(proc_rows, n_cols)
        #     gathered_arrays.append(proc_data)
        #     start += proc_size 
        # gathered_array = np.concatenate(gathered_arrays, axis=0)

    COMM.Barrier()

    return gathered_array

def inference(cfg: DictConfig) -> None:
    """Perform 'a-priori' inference from a set of loaded input files
    """
    trainer = Trainer(cfg, COMM)
    trainer.writeGraphStatistics()

    if RANK == 0:
        if not os.path.exists(cfg.inference_dir + trainer.model.module.get_save_header()):
            os.makedirs(cfg.inference_dir + trainer.model.module.get_save_header())

    graph = trainer.data['graph']
    stats = trainer.data['stats']
    loader = trainer.data['test']['loader']
    pos = graph.pos_orig 

    trainer.model.eval()
    with torch.no_grad():
        for bidx, data in enumerate(loader):
            if RANK == 0: log.info(f"~~~~ ROLLOUT STEP {bidx} ~~~~")
            x = data['x']
            pred_scaled = trainer.inference_step(x)
            
            # unscale 
            pred = pred_scaled * stats['std'] + stats['mean']

            # get target 
            target = data['y'][0]
        
            n_nodes_local = graph.n_nodes_local
            pred = pred[:n_nodes_local]
            target = target[:n_nodes_local]
            error = pred - target.to(trainer.device)
            pos = pos[:n_nodes_local]

            if RANK == 0: 
                log.info(f"Shape of pred: {pred.shape}")
                log.info(f"Shape of target: {target.shape}")
                log.info(f"Shape of pos: {pos.shape}")
        
            # Gather the prediction and target with mpi4py gatherv
            if RANK == 0: log.info("Gathering data...")
            x_gathered = gather_wrapper(x.cpu().numpy())
            pred_gathered = gather_wrapper(pred.cpu().numpy())
            target_gathered = gather_wrapper(target.cpu().numpy())
            error_gathered = gather_wrapper(error.cpu().numpy())
            pos_gathered = gather_wrapper(pos.cpu().numpy())

            # Write the data
            if RANK == 0:
                log.info("Writing...")
                np.save(cfg.inference_dir + trainer.model.module.get_save_header() + f"/x_{bidx}", x_gathered)
                np.save(cfg.inference_dir + trainer.model.module.get_save_header() + f"/pred_{bidx}", pred_gathered)
                np.save(cfg.inference_dir + trainer.model.module.get_save_header() + f"/target_{bidx}", target_gathered)
                np.save(cfg.inference_dir + trainer.model.module.get_save_header() + f"/error_{bidx}", error_gathered)
                np.save(cfg.inference_dir + trainer.model.module.get_save_header() + f"/pos_{bidx}", pos_gathered)

    # Torch distributed cleanup
    trainer.cleanup()

def inference_rollout(cfg: DictConfig,
                      client: Optional[OnlineClient] = None
    ) -> None:
    """Perform 'a-posteriori' inference by rolling out in time an initial condition
    """
    trainer = Trainer(cfg, COMM, client=client)
    trainer.writeGraphStatistics()

    dataloader = trainer.data['train']['loader']
    data = next(iter(dataloader))
    x = data['x']
    graph = trainer.data['graph']
    stats = trainer.data['stats']
    n_nodes_local = graph.n_nodes_local
    pos = graph.pos_orig[:n_nodes_local]

    # Roll-out loop
    trainer.model.eval()
    local_time = []
    local_throughput = []
    with torch.no_grad():
        while True:
            t_step = time.time()
            x = trainer.inference_step(x)
            t_step = time.time() - t_step
            if trainer.iteration > 0:
                local_time.append(t_step)
                local_throughput.append(n_nodes_local/t_step/1.0e6)
            trainer.iteration += 1

            # Logging 
            if RANK == 0:
                summary = ' '.join([
                    f'[STEP {trainer.iteration}]',
                    f't_step={t_step:.4g}sec',
                    f'throughput={n_nodes_local/t_step:.4g}nodes/sec'
                ])
                log.info(summary)

            # Checkpoint  
            if trainer.iteration % cfg.ckptfreq == 0:
                trainer.checkpoint()
            
            # Break loop
            if trainer.iteration >= cfg.rollout_steps:
                break
    
    # Save solution checkpoint 
    x = x.cpu()
    x = x * stats['x_std'] + stats['x_mean']
    if not cfg.online:
        # Gather the prediction and target with mpi4py gatherv
        if RANK == 0: log.info("Gathering data...")
        x_gathered = gather_wrapper(x.numpy())
        pos_gathered = gather_wrapper(pos.cpu().numpy())

        # Write the data
        save_path = cfg.inference_dir + trainer.model.module.get_save_header()
        if RANK == 0:
            log.info("Writing...")
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            np.save(save_path + f"/x_{trainer.iteration}", x_gathered)
            np.save(save_path + f"/pos_{trainer.iteration}", pos_gathered)
    else:
        client.put_array(f'checkpt_u_rank_{RANK}_size_{SIZE}',x.to(torch.float32).numpy())

    # Torch distributed cleanup
    trainer.cleanup()

    # Print performance stats
    global_stats = utils.collect_stats(COMM, n_nodes_local, local_time, local_throughput)
    if RANK == 0:
        log.info('Performance metrics:')
        log.info(f'\tTotal number of graph nodes: {global_stats["n_nodes"]}')
        log.info(f'\tTotal number of iterations: {trainer.iteration-1}')
        min_val, max_val, avg_val = utils.min_max_avg(global_stats["time"])
        log.info(f'\tStep time [sec]: min={min_val:.4g}, max={max_val:.4g}, mean={avg_val:.4g}')
        min_val, max_val, avg_val = utils.min_max_avg(global_stats["throughput"])
        log.info(f'\tStep throughput [million nodes / sec]: min={min_val:.4g}, max={max_val:.4g}, mean={avg_val:.4g}')
        min_val, max_val, avg_val = utils.min_max_avg(global_stats["glob_throughput"])
        log.info(f'\tParallel throughput [million nodes / sec]: min={min_val:.4g}, max={max_val:.4g}, mean={avg_val:.4g}')

    # Print FOM
    fom_local = (global_stats["n_nodes"] / 1.0e6) * (trainer.iteration-1) / sum(local_time)
    fom_gather = COMM.gather(fom_local, root=0)
    if RANK == 0:
        log.info('FOM:')
        min_val, max_val, avg_val = utils.min_max_avg(fom_gather)
        log.info(f'\tFOM_inference [million graph nodes x inference steps / inference time]: min={min_val:.4g}, max={max_val:.4g}, mean={avg_val:.4g}')


@hydra.main(version_base=None, config_path='./conf', config_name='config')
def main(cfg: DictConfig) -> None:
    if cfg.verbose:
        log.info(f'Hello from rank {RANK}/{SIZE}, local rank {LOCAL_RANK}, on node {HOST_NAME} and device {DEVICE}:{DEVICE_ID+cfg.device_skip} out of {N_DEVICES}.')
    
    if RANK == 0:
        log.info('\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        log.info('RUNNING WITH INPUTS:')
        log.info(OmegaConf.to_yaml(cfg)) 
        log.info('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

    if not cfg.online:
        inference(cfg)
    else:
        client = OnlineClient(cfg,COMM)
        COMM.Barrier()
        if RANK == 0: print('Initialized Online Client!\n', flush=True)
        inference_rollout(cfg, client)

    if RANK == 0:
        log.info('Exiting ...')

if __name__ == '__main__':
    main()
