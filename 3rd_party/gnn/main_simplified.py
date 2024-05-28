from __future__ import absolute_import, division, print_function, annotations
import os
import sys
import socket
import logging
import hydra
from omegaconf import DictConfig, OmegaConf
log = logging.getLogger(__name__)

# Get MPI:
try:
    from mpi4py import MPI
    import torch
    import torch.distributed as dist
    import torch.distributed.nn as distnn

    LOCAL_RANK = os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK', '0')
    # LOCAL_RANK = os.environ['OMPI_COMM_WORLD_LOCAL_RANK']
    SIZE = MPI.COMM_WORLD.Get_size()
    RANK = MPI.COMM_WORLD.Get_rank()
    COMM = MPI.COMM_WORLD

    WITH_CUDA = torch.cuda.is_available()

    # # Override gpu utilization
    # WITH_CUDA = False

    DEVICE = 'gpu' if WITH_CUDA else 'cpu'
    if DEVICE == 'gpu':
        DEVICE_ID = 'cuda:0' 
    else:
        DEVICE_ID = 'cpu'

    # pytorch will look for these
    os.environ['RANK'] = str(RANK)
    os.environ['WORLD_SIZE'] = str(SIZE)
    # -----------------------------------------------------------
    # NOTE: Get the hostname of the master node, and broadcast
    # it to all other nodes It will want the master address too,
    # which we'll broadcast:
    # -----------------------------------------------------------
    MASTER_ADDR = socket.gethostname() if RANK == 0 else None
    MASTER_ADDR = MPI.COMM_WORLD.bcast(MASTER_ADDR, root=0)
    os.environ['MASTER_ADDR'] = MASTER_ADDR
    os.environ['MASTER_PORT'] = str(2345)

except (ImportError, ModuleNotFoundError) as e:
    SIZE = 1
    RANK = 0
    LOCAL_RANK = 0
    MASTER_ADDR = 'localhost'
    log.warning('MPI Initialization failed!')
    log.warning(e)

def init_process_group(
    rank: Union[int, str],
    world_size: Union[int, str],
    backend: Optional[str] = None,
) -> None:
    if WITH_CUDA:
        backend = 'nccl' if backend is None else str(backend)

    else:
        backend = 'gloo' if backend is None else str(backend)

    dist.init_process_group(
        backend,
        rank=int(rank),
        world_size=int(world_size),
        init_method='env://',
    )

def cleanup():
    dist.destroy_process_group()

def trace_handler(p):
    output = p.key_averages().table(sort_by="cpu_time_total", row_limit=20)
    print(temp_test)

def halo_swap(
          input_tensor,
          mask_send,
          mask_recv,
          buff_send,
          buff_recv,
          neighboring_procs,
          SIZE,
          halo_swap_mode):
    """ 
    Performs halo swap using send/receive buffers
    """
    if SIZE > 1:
        if halo_swap_mode == 'all_to_all':
            log.info(f"[RANK {RANK}] -- in all_to_all")
            # Fill send buffer
            for i in neighboring_procs:
                n_send = len(mask_send[i])
                buff_send[i][:n_send,:] = input_tensor[mask_send[i]]

            # # Perform all_to_all
            distnn.all_to_all(buff_recv, buff_send)

            # Fill halo nodes
            for i in neighboring_procs:
                n_recv = len(mask_recv[i])
                input_tensor[mask_recv[i]] = buff_recv[i][:n_recv,:]

        elif halo_swap_mode == 'send_recv':
            log.info(f"[RANK {RANK}] -- in send_recv")
            # Fill send buffer
            for i in neighboring_procs:
                n_send = len(mask_send[i])
                buff_send[i][:n_send,:] = input_tensor[mask_send[i]]
            
            # distnn.send_recv(buff_recv, buff_send, neighboring_procs)

            # Perform sendrecv
            send_req = []
            for dst in neighboring_procs:
                tmp = dist.isend(buff_send[dst], dst)
                send_req.append(tmp)
            recv_req = []
            for src in neighboring_procs:
                tmp = dist.irecv(buff_recv[src], src)
                recv_req.append(tmp)

            for req in send_req:
                req.wait()
            for req in recv_req:
                req.wait()
            dist.barrier()

            # Fill halo nodes
            for i in neighboring_procs:
                n_recv = len(mask_recv[i])
                input_tensor[mask_recv[i]] = buff_recv[i][:n_recv,:]
        elif halo_swap_mode == 'none':
            pass
        else:
            raise ValueError("halo_swap_mode %s not valid. Valid options: all_to_all, sendrecv" %(self.halo_swap_mode))
    return input_tensor

def halo_test(cfg: DictConfig) -> None:
    """ 
    checking if halo exchange works 
    """
    if SIZE == 2:
        init_process_group(RANK, SIZE, backend=cfg.backend)
        nlocal = 5
        if RANK == 0:
            #                    0    1    2    3    4    5    6    7    8
            x = torch.tensor(  [1.0, 2.0, 3.0, 4.0, 5.0, 0.0]).unsqueeze(-1)
            neighboring_procs = torch.tensor([1])
            mask_send = [torch.tensor([]), torch.tensor([4])]
            mask_recv = [torch.tensor([]), torch.tensor([5])]
            buffer_send = [torch.tensor([0.0]).unsqueeze(-1), # 0 
                           torch.tensor([0.0]).unsqueeze(-1)  # 1
                           ]
            buffer_recv = [torch.tensor([0.0]).unsqueeze(-1), # 0
                           torch.tensor([0.0]).unsqueeze(-1)  # 1
                           ]
        if RANK == 1:
            #                    0    1    2    3    4    5    6    7    8
            x = torch.tensor(  [5.0, 6.0, 7.0, 8.0, 9.0, 0.0]).unsqueeze(-1)
            neighboring_procs = torch.tensor([0])
            mask_send = [torch.tensor([0]), torch.tensor([])]
            mask_recv = [torch.tensor([5]), torch.tensor([])]
            buffer_send = [torch.tensor([0.0]).unsqueeze(-1), # 0 
                           torch.tensor([0.0]).unsqueeze(-1)  # 1
                           ]
            buffer_recv = [torch.tensor([0.0]).unsqueeze(-1), # 0
                           torch.tensor([0.0]).unsqueeze(-1)  # 1
                           ]
        if WITH_CUDA:
            x = x.cuda()
            for i in range(len(buffer_send)):
                buffer_send[i] = buffer_send[i].cuda()
                buffer_recv[i] = buffer_recv[i].cuda()
        
        xh = torch.clone(x)
        xh = halo_swap(
          xh,
          mask_send,
          mask_recv,
          buffer_send,
          buffer_recv,
          neighboring_procs,
          SIZE,
          cfg.halo_swap_mode)

        log.info(f"[RANK {RANK}] -- x after swap:\n{xh}")

    else:
        raise ValueError("Number of ranks must be 2 for this script.")

    return 

@hydra.main(version_base=None, config_path='./conf', config_name='config')
def main(cfg: DictConfig) -> None:
    if RANK == 0:
        log.info(f"halo_swap_mode: {cfg.halo_swap_mode} \t backend: {cfg.backend}")
    halo_test(cfg)
    if RANK == 0:
        log.info(f"done.")
    cleanup()

if __name__ == '__main__':
    main()
