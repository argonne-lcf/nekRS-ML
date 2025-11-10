"""
PyTorch DDP training script for GNN-based surrogates from mesh data
"""
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
    #import mpi4py
    #mpi4py.rc.initialize = False
    from mpi4py import MPI
    WITH_DDP = True
except ModuleNotFoundError as e:
    WITH_DDP = False
    pass

import torch
#try:
#    import intel_extension_for_pytorch as ipex
#except Exception as e:
#    pass
#try:
#    import oneccl_bindings_for_pytorch as ccl
#except Exception as e:
#    pass

# Local imports
import utils
from trainer import DGNTrainer
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


def train(cfg: DictConfig,
          client: Optional[OnlineClient] = None
    ) -> None:
    trainer = DGNTrainer(cfg, client=client)
    trainer.writeGraphStatistics()
    n_nodes_local = trainer.data_reduced.n_nodes_local.item()

    # Training loop: 
    trainer.model.train()
    loss_window = deque(maxlen=10)
    local_time = []
    local_throughput = []
    while True:
        train_loader = trainer.data['train']['loader']
        val_loader = trainer.data['validation']['loader']
        for bidx, data in enumerate(train_loader):
            t_step = time.time()
            loss = trainer.train_step(data)
            t_step = time.time() - t_step 
            if trainer.iteration > 0:
                local_time.append(t_step)
                local_throughput.append(n_nodes_local/t_step/1.0e6)
            trainer.loss_hist_train[trainer.iteration] = loss.item() 
            loss_window.append(loss.item())
            running_loss = sum(loss_window) / len(loss_window)
            trainer.iteration += 1 
            
            # Calculate gradients  
            if cfg.postprocess: postproc_out = trainer.postprocess()

            # Logging 
            if RANK == 0:
                summary_train = ' '.join([
                    f'[STEP {trainer.iteration}]',
                    f'loss={loss:.4e}',
                    f'r_loss={running_loss:.4e}',   # Include average loss in your logging
                    f't_step={t_step:.4g}sec',
                    f"lr={trainer.optimizer.param_groups[0]['lr']:.3e}"
                ])
                sepstr = '-' * len(summary_train)
                log.info(sepstr)
                log.info(summary_train)
                if cfg.timers:
                    t_dataTransfer = trainer.timers['dataTransfer'][trainer.timer_step-1]
                    t_bufferInit = trainer.timers['bufferInit'][trainer.timer_step-1]
                    t_forwardPass = trainer.timers['forwardPass'][trainer.timer_step-1]
                    t_loss = trainer.timers['loss'][trainer.timer_step-1]
                    t_backwardPass = trainer.timers['backwardPass'][trainer.timer_step-1]
                    t_optimizerStep = trainer.timers['optimizerStep'][trainer.timer_step-1]
                    log.info(f"t_dataTransfer: {t_dataTransfer:.4g} sec") 
                    log.info(f"t_bufferInit: {t_bufferInit:.4g} sec")
                    log.info(f"t_forwardPass: {t_forwardPass:.4g} sec [{n_nodes_local/t_forwardPass:.4e} nodes/sec]")
                    log.info(f"t_loss: {t_loss:.4g} sec [{n_nodes_local/t_loss:.4e} nodes/sec]")
                    log.info(f"t_backwardPass: {t_backwardPass:.4g} sec [{n_nodes_local/t_backwardPass:.4e} nodes/sec]")
                    log.info(f"t_optimizerStep: {t_optimizerStep:.4g} sec")
                if cfg.postprocess:
                    log.info(f"grad norm: {postproc_out[0]:.6g}")

            # Checkpoint  
            if trainer.iteration % cfg.ckptfreq == 0:
                trainer.checkpoint()

            # Update data loader when online training (and flush stream if using ADIOS)
            if cfg.online:
                if trainer.iteration % cfg.online_update_freq == 0:
                    trainer.update_data()
                    break

            # Break loop over dataloader
            if trainer.iteration >= trainer.total_iterations:
                break
        
        # Break while loop
        if trainer.iteration >= trainer.total_iterations:
            break
    
    # Correctness validation
    if cfg.target_loss != 0:
        if math.isclose(cfg.target_loss,loss.item(),rel_tol=0.001):
            if RANK==0: print('\n\nSUCCESS! GNN training validated!\n\n')
        else:
            if RANK==0: 
                print('\n\nWARNING! GNN training failed validation!')
                print(f'Target loss: {cfg.target_loss}, obtained loss: {loss.item()}\n\n')
    
    # Save model
    trainer.save_model()

    # Tell simulation to exit
    if cfg.online:
        if RANK == 0: log.info(f"[RANK {RANK}] -- Telling NekRS to quit ...")
        client.stop_nekRS()
    COMM.Barrier()

    # Print performance stats
    global_stats = utils.collect_stats(n_nodes_local, local_time, local_throughput)
    if RANK == 0:
        log.info('Performance metrics:')
        log.info(f'\tTotal number of graph nodes: {global_stats["n_nodes"]}')
        log.info(f'\tTotal number of iterations: {trainer.iteration-1}')
        min_val, max_val, avg_val = utils.min_max_avg(global_stats["time"])
        log.info(f'\tStep time [sec]: min={min_val:.4g}, max={max_val:.4g}, mean={avg_val:.4g}')
        min_val, max_val, avg_val = utils.min_max_avg(global_stats["throughput"])
        log.info(f'\tLocal step throughput [million nodes / sec]: min={min_val:.4g}, max={max_val:.4g}, mean={avg_val:.4g}')
        min_val, max_val, avg_val = utils.min_max_avg(global_stats["glob_throughput"])
        log.info(f'\tParallel step throughput [million nodes / sec]: min={min_val:.4g}, max={max_val:.4g}, mean={avg_val:.4g}')
    if cfg.online:
        glob_online_stats = utils.collect_online_stats(trainer.online_timers['trainDataTime'],trainer.online_timers['trainDataThroughput'])
        if RANK == 0:
            min_val, max_val, avg_val = utils.min_max_avg(glob_online_stats["time"])
            log.info(f'\tTransfer time per stream [sec]: min={min_val:.4g}, max={max_val:.4g}, mean={avg_val:.4g}')
            min_val, max_val, avg_val = utils.min_max_avg(glob_online_stats["tot_time"])
            log.info(f'\tTotal transfer time [sec]: min={min_val:.4g}, max={max_val:.4g}, mean={avg_val:.4g}')
            min_val, max_val, avg_val = utils.min_max_avg(glob_online_stats["throughput"])
            log.info(f'\tLocal transfer throughput [GB / sec]: min={min_val:.4g}, max={max_val:.4g}, mean={avg_val:.4g}')
            min_val, max_val, avg_val = utils.min_max_avg(glob_online_stats["glob_throughput"])
            log.info(f'\tParallel transfer throughput [GB / sec]: min={min_val:.4g}, max={max_val:.4g}, mean={avg_val:.4g}')

    # Print FOM
    gnn_fom = (global_stats["n_nodes"] / 1.0e6) * (trainer.iteration-1) / sum(local_time)
    gnn_fom_gather = COMM.gather(gnn_fom, root=0)
    if cfg.online:
        data_transfer_fom = glob_online_stats["glob_throughput"]
    if RANK == 0:
        log.info('FOM:')
        min_val, max_val, avg_val = utils.min_max_avg(gnn_fom_gather)
        log.info(f'\tFOM_train [million graph nodes x train steps / train time]: min={min_val:.4g}, max={max_val:.4g}, mean={avg_val:.4g}')
        if cfg.online:
            min_val, max_val, avg_val = utils.min_max_avg(data_transfer_fom)
            log.info(f'\tFOM_transfer [GB / transfer time]: min={min_val:.4g}, max={max_val:.4g}, mean={avg_val:.4g}')


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
        train(cfg)
    else:
        log.info('Oline training not implemented yet for this model')
        COMM.Abort(1)
    
    utils.cleanup()
    if RANK == 0:
        log.info('Exiting ...')


if __name__ == '__main__':
    main()
