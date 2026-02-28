"""
Trainer for distributed, consistent graph neural network
"""
import sys
import os
import socket
from typing import Optional, Union, Callable
import logging
import numpy as np
import time
from omegaconf import DictConfig, OmegaConf

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.cuda.amp.grad_scaler import GradScaler
import torch.nn as nn
import torch.optim as optim

# PyTorch Geometric
import torch_geometric
from torch_geometric.data import Data
import torch_geometric.utils as pyg_utils

# Local imports
from client import OnlineClient

log = logging.getLogger(__name__)
Tensor = torch.Tensor
NP_FLOAT_DTYPE = np.float32
SMALL = 1e-12
GB_SIZE = 1024**3


class Trainer:
    def __init__(self, 
                 cfg: DictConfig, 
                 COMM,
                 scaler: Optional[GradScaler] = None,
                 client: Optional[OnlineClient] = None
    ) -> None:
        self.cfg = cfg
        if scaler is None:
            self.scaler = None
        self.backend = self.cfg.backend
        self.client = client

        # ~~~ Get MPI info
        self.comm = COMM
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.local_rank = int(os.getenv("PALS_LOCAL_RANKID"))
        self.local_size = int(os.getenv("PALS_LOCAL_SIZE"))
        #self.host_name = MPI.Get_processor_name()

        # ~~~~ Init torch stuff 
        self.setup_torch()
        #self.import_torch_geometric()

        # ~~~~ Setup local graph 
        self.pos, graph_read_time = self.load_graph_data()
        self.comm.Barrier()
        if self.rank == 0: print(f'[Trainer] Read graph in {graph_read_time:.4f} seconds', flush=True)

        # ~~~~ Setup training data 
        self.load_trajectory()
        self.comm.Barrier()

        # ~~~ Initialize torch distributed and wrap model in DDP
        self.init_process_group(self.cfg.master_addr, self.cfg.master_port)
        #if self.size > 1:
        #    self.model = DDP(self.model, broadcast_buffers=False, gradient_as_bucket_view=True)
        
        # ~~~~ Set loss function
        #self.loss_fn = nn.MSELoss()
        #if self.with_cuda or self.with_xpu:
        #    self.loss_fn.to(self.device)

        # ~~~~ Set optimizer
        #self.optimizer = self.build_optimizer(self.model)

        #self.import_torch_geometric()

    def import_torch_geometric(self):
        _root_logger = logging.getLogger()
        _prev_level = _root_logger.level
        _root_logger.setLevel(logging.WARNING)
        import torch_geometric
        from torch_geometric.data import Data
        import torch_geometric.utils as pyg_utils
        _root_logger.setLevel(_prev_level)

    def init_process_group(self, master_addr: str, master_port: int):
        # Lazy import to avoid CXI resource contention with ADIOS2
        #import torch.distributed as dist
        
        os.environ['RANK'] = str(self.rank)
        os.environ['WORLD_SIZE'] = str(self.size)
        if master_addr=='none':
            MASTER_ADDR = socket.gethostname() if self.rank == 0 else None
            MASTER_ADDR = self.comm.bcast(MASTER_ADDR, root=0)
        else:
            MASTER_ADDR = str(master_addr)
        os.environ['MASTER_ADDR'] = MASTER_ADDR
        os.environ['MASTER_PORT'] = str(master_port)

        if torch.cuda.is_available():
            backend = 'nccl' if self.backend is None else str(self.backend)
        elif torch.xpu.is_available():
            backend = 'xccl' if self.backend is None else str(self.backend)
        else:
            backend = 'gloo' if self.backend is None else str(backend)
        dist.init_process_group(backend, rank=int(self.rank), world_size=int(self.size), init_method='env://')
    
    def cleanup(self):
        # Lazy import to avoid CXI resource contention with ADIOS2
        #import torch.distributed as dist
        dist.destroy_process_group()

    def build_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        #optimizer = optim.Adam(model.parameters(), lr=0.0)
        optimizer = optim.AdamW(model.parameters(), lr=0.0, betas=(0.9, 0.95), weight_decay=0.1)
        return optimizer

    def setup_torch(self):
        # Random seeds
        torch.manual_seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)

        # Set device
        self.with_cuda = torch.cuda.is_available()
        self.with_xpu = torch.xpu.is_available()
        if self.with_cuda:
            self.device = torch.device('cuda')
            self.n_devices = torch.cuda.device_count()
            self.device_id = self.local_rank if self.n_devices>1 else 0
        elif self.with_xpu:
            self.device = torch.device('xpu')
            self.n_devices = torch.xpu.device_count()
            self.device_id = self.local_rank if self.n_devices>1 else 0
        else:
            self.device = torch.device('cpu')
            self.device_id = 'cpu'

        # Device and intra-op threads
        if self.with_cuda:
            torch.cuda.set_device(self.device_id)
        elif self.with_xpu:
            torch.xpu.set_device(self.device_id)
        torch.set_num_threads(self.cfg.num_threads)

        # Precision
        if self.cfg.precision == 'fp32':
            self.torch_dtype = torch.float32
        elif self.cfg.precision == 'bf16':
            self.torch_dtype = torch.bfloat16
        elif self.cfg.precision == 'fp64':
            self.torch_dtype = torch.float64
        else:
            sys.exit('Only fp32, fp64 and bf16 data types are currently supported')

        # Reset peak memory stats and empty cache
        if self.with_cuda:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
        elif self.with_xpu:
            torch.xpu.reset_peak_memory_stats()
            torch.xpu.empty_cache()

    def load_data(self, file_name, 
                  dtype: Optional[type] = np.float64, 
                  extension: Optional[str] = ""
    ):
        data = self.client.get_array(file_name).astype(dtype)
        return data

    def load_graph_data(self):
        """
        Load in the local graph
        """
        if self.rank == 0: print('[Trainer] Reading the graph ...', flush=True)
        main_path = ""
        
        if self.cfg.client.backend == 'adios':
            tic = time.perf_counter()
            graph_data = self.client.get_graph_data_from_stream()
            read_time = time.perf_counter() - tic
            self.N = graph_data['N']
            pos = graph_data['pos']
        else:
            path_to_pos_full = main_path + 'pos_node_rank_%d_size_%d' %(self.rank,self.size)

            # Polynomial order
            self.Np = np.array([0], dtype=np.float32)
            if self.rank == 0:
                path_to_Np = main_path + "Np_rank_%d_size_%d" %(self.rank, self.size)
                self.Np = self.load_data(path_to_Np, dtype=np.float32)
            self.comm.Bcast(self.Np, root=0)

            # Node positions
            if self.cfg.verbose: log.info('[RANK %d]: Loading positions and global node index' %(self.rank))
            #pos = np.fromfile(self.cfg.gnn_outputs_path+'/'+path_to_pos_full + ".bin", dtype=np.float64).reshape((-1,3))
            pos = self.load_data(path_to_pos_full, extension='.bin').reshape((-1,3))

        return pos, read_time

    def load_trajectory(self):
        """Load a solution trajectory
        """
        self.comm.Barrier() # sync helps here
        if self.rank == 0: print(f'[Trainer] Reading trajectory data ...', flush=True)
        # read files
        if self.cfg.client.backend == 'smartredis':
            # Get the file list
            tic = time.time()
            output_files = self.client.get_file_list(f'outputs_rank_{self.rank}') # outputs must come first
            input_files = self.client.get_file_list(f'inputs_rank_{self.rank}')
            self.online_timers['metaData'].append(time.time()-tic)

            # Load files
            if self.cfg.verbose: log.info(f'[RANK {self.rank}]: Found {len(output_files)} trajectory files in DB')
            for i in range(len(output_files)):
                tic = time.time()
                data_x_i = self.client.get_array(input_files[i]).astype(NP_FLOAT_DTYPE).T
                toc = time.time()
                data_x_i = self.prepare_snapshot_data(data_x_i)
                
                tic = time.time()
                data_y_i = self.client.get_array(output_files[i]).astype(NP_FLOAT_DTYPE).T
                toc = time.time()
                data_y_i = self.prepare_snapshot_data(data_y_i)
                self.data_list.append(
                        {'x': data_x_i, 'y':data_y_i}
                )
        elif self.cfg.client.backend == 'adios':
            data_x_i, data_y_i, ttime = self.client.get_train_data_from_stream()
        return data_x_i, data_y_i, ttime

    def train_step(self) -> Tensor:
        time.sleep(5)
        data_x_i, data_y_i, ttime = self.load_trajectory()
        return ttime