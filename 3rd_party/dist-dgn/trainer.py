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
from torch.cuda.amp.grad_scaler import GradScaler
import torch.nn as nn
import torch.optim as optim
#from torch.utils.data import DataLoader
#torch.use_deterministic_algorithms(True)
#import torch.utils.data
#import torch.utils.data.distributed
#import torch.multiprocessing as mp
#import torch.distributions as tdist 
#from torch.profiler import profile, record_function, ProfilerActivity
#import torch.nn.functional as F
#from torchvision import datasets, transforms

import torch.distributed as dist
import torch.distributed.nn as distnn
from torch.nn.parallel import DistributedDataParallel as DDP

# PyTorch Geometric
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch_geometric.utils as pyg_utils
#import torch_geometric.nn as tgnn

# Local imports
import utils
from scheduler import ScheduledOptim
import gnn
import graph_connectivity as gcon
from client import OnlineClient
import create_halo_info_par
from step_sampler import UniformStepSampler
from diffusion_process import DiffusionProcess

log = logging.getLogger(__name__)
Tensor = torch.Tensor
NP_FLOAT_DTYPE = np.float32
SMALL = 1e-12
GB_SIZE = 1024**3

try:
    import mpi4py
    mpi4py.rc.initialize = False
    from mpi4py import MPI
    if not MPI.Is_initialized():
        MPI.Init()
    COMM = MPI.COMM_WORLD
    RANK = COMM.Get_rank()
    SIZE = COMM.Get_size()
    LOCAL_RANK = int(os.getenv("PALS_LOCAL_RANKID"))
    LOCAL_SIZE = int(os.getenv("PALS_LOCAL_SIZE"))
    WITH_DDP = True
except ModuleNotFoundError as e:
    SIZE = 1
    RANK = 0
    LOCAL_RANK = 0
    MASTER_ADDR = 'localhost'
    WITH_DDP = False
    pass

try:
    WITH_CUDA = torch.cuda.is_available()
except:
    WITH_CUDA = False
    pass

try:
    WITH_XPU = torch.xpu.is_available()
except:
    WITH_XPU = False
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


class DGNTrainer:
    def __init__(self, 
                 cfg: DictConfig, 
                 scaler: Optional[GradScaler] = None,
                 client: Optional[OnlineClient] = None
    ) -> None:
        self.cfg = cfg
        self.rank = RANK
        if scaler is None:
            self.scaler = None
        self.device = DEVICE
        self.backend = self.cfg.backend
        self.client = client

        # ~~~ Perform some checks
        if not self.cfg.consistency:
            assert self.cfg.halo_swap_mode == 'none', \
                "For inconsistent model, set halo_swap_mode=none"
        if self.cfg.online:
            log.warning('Online backends not implemented for this model yet')
            MPI.Abort(COMM, 1)
        if self.cfg.batch_size > 1 or self.cfg.val_batch_size > 1:
            log.error('Only batch size 1 is currently supported')
            MPI.Abort(COMM, 1)

        # ~~~ Initialize DDP
        if WITH_DDP:
            os.environ['RANK'] = str(RANK)
            os.environ['WORLD_SIZE'] = str(SIZE)
            if self.cfg.master_addr=='none':
                MASTER_ADDR = socket.gethostname() if RANK == 0 else None
                MASTER_ADDR = COMM.bcast(MASTER_ADDR, root=0)
            else:
                MASTER_ADDR = str(cfg.master_addr)
            os.environ['MASTER_ADDR'] = MASTER_ADDR
            os.environ['MASTER_PORT'] = str(cfg.master_port)
            utils.init_process_group(RANK, SIZE, backend=self.backend)
        
        # ~~~~ Init torch stuff 
        self.setup_torch()

        # ~~~~ Setup timers
        if self.cfg.timers:
            self.timer_step = 0
            self.timer_step_max = self.total_iterations - self.iteration
            self.timers = self.setup_timers(self.timer_step_max)
            self.timers_max = self.setup_timers(self.timer_step_max)
            self.timers_min = self.setup_timers(self.timer_step_max)
            self.timers_avg = self.setup_timers(self.timer_step_max)

        # ~~~ Setup online timers
        if self.cfg.online:
            self.online_timers = self.setup_online_timers()

        # ~~~~ Setup local graph 
        self.data_reduced, \
            self.data_full, \
            self.idx_full2reduced, \
            self.idx_reduced2full = self.setup_local_graph()
                
        # ~~~~ Setup halo nodes 
        self.neighboring_procs = []
        self.setup_halo()

        # ~~~~ Setup data 
        self.data_list = []
        self.data = {}
        self.setup_data()
        if RANK == 0: log.info('Done with setup_data')

        # ~~~~ Setup halo exchange masks
        self.mask_send, self.mask_recv = self.build_masks()
        if RANK == 0: log.info('Done with build_masks')

        self.buffer_send, self.buffer_recv, self.n_buffer_rows = self.build_buffers(self.cfg.mlp_hidden_channels)
        if RANK == 0: log.info('Done with build_buffers')

        # ~~~~ Build model and move to gpu 
        self.model = self.build_model()
        if RANK == 0: 
            log.info('Built model with %i trainable parameters' %(self.count_weights(self.model)))
        self.model.to(self.device)
        self.model.to(self.torch_dtype)
        if RANK == 0: log.info('Done with build_model')

        # ~~~~ Set the total number of training iterations 
        self.total_iterations = self.cfg.phase1_steps + self.cfg.phase2_steps + self.cfg.phase3_steps

        # ~~~~ Init training and validation loss history 
        self.loss_hist_train = np.zeros(self.total_iterations)
        self.loss_hist_val = np.zeros(self.total_iterations)

        # ~~~~ Set model and checkpoint savepaths 
        try:
            self.ckpt_path = cfg.ckpt_dir + '/' + self.model.get_save_header() + '.tar'
            self.model_path = cfg.model_dir + '/' + self.model.get_save_header() + '.tar'
        except (AttributeError) as e:
            self.ckpt_path = cfg.ckpt_dir + 'checkpoint.tar'
            self.model_path = cfg.model_dir + 'model.tar'

        # ~~~~ Load model parameters if we are restarting from checkpoint
        self.iteration = 0
        if self.cfg.restart:
            if RANK == 0: log.info(f'Loading model checkpoint from {self.ckpt_path}')
            ckpt = torch.load(self.ckpt_path, weights_only=False)
            self.model.load_state_dict(ckpt['model_state_dict'])
            self.iteration = ckpt['iteration'] + 1
            self.loss_hist_train = ckpt['loss_hist_train']
            self.loss_hist_val = ckpt['loss_hist_val']

            if len(self.loss_hist_train) < self.total_iterations:
                loss_hist_train_new = np.zeros(self.total_iterations)
                loss_hist_val_new = np.zeros(self.total_iterations)

                loss_hist_train_new[:len(self.loss_hist_train)] = self.loss_hist_train
                loss_hist_val_new[:len(self.loss_hist_val)] = self.loss_hist_val

                self.loss_hist_train = loss_hist_train_new
                self.loss_hist_val = loss_hist_val_new
        if self.cfg.model_task == 'inference':
            if RANK == 0: log.info(f'Loading model checkpoint from {self.model_path}')
            ckpt = torch.load(self.model_path, weights_only=False)
            self.model.load_state_dict(ckpt['state_dict'])

        # ~~~~ Set loss function
        self.loss_fn = nn.MSELoss()
        if WITH_CUDA or WITH_XPU:
            self.loss_fn.to(self.device)

        # ~~~~ Set optimizer
        self.optimizer = self.build_optimizer(self.model)

        # ~~~~ Load optimizer parameters if we are restarting from checkpoint
        if self.cfg.restart:
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            if RANK == 0:
                astr = 'Restarting from checkpoint -- Iteration %d/%d' %(self.iteration, self.total_iterations)
                log.info(astr)

        # ~~~~ Set scheduler:
        self.s_optimizer = ScheduledOptim(self.optimizer, 
                                          self.cfg.phase1_steps, 
                                          self.cfg.phase2_steps, 
                                          self.cfg.phase3_steps, 
                                          self.cfg.lr_phase12, 
                                          self.cfg.lr_phase23)
        self.s_optimizer.reset_n_steps(self.iteration)

        # ~~~~ Set step sampler
        self.step_sampler = UniformStepSampler(self.cfg.num_diffusion_steps, 
                                               self.device, 
                                               self.torch_dtype)

        # ~~~~ Set diffusion process
        self.diffusion_process = DiffusionProcess(self.cfg.num_diffusion_steps)

        # ~~~~ Wrap model in DDP
        if WITH_DDP and SIZE > 1:
            self.model = DDP(self.model, broadcast_buffers=False, gradient_as_bucket_view=True)

    def checkpoint(self):
        if RANK == 0:
            t_ckpt = time.time()

            if not os.path.exists(self.cfg.ckpt_dir):
                os.makedirs(self.cfg.ckpt_dir)

            if WITH_DDP and SIZE > 1:
                sd = self.model.module.state_dict()
            else:
                sd = self.model.state_dict()
            ckpt = {'iteration' : self.iteration,
                    'model_state_dict' : sd,
                    'optimizer_state_dict' : self.optimizer.state_dict(),
                    'loss_hist_train' : self.loss_hist_train,
                    'loss_hist_val' : self.loss_hist_val}
            torch.save(ckpt, self.ckpt_path + f".{self.iteration}")
            torch.save(ckpt, self.ckpt_path)
            t_ckpt = time.time() - t_ckpt

            astr = f"Checkpointing ({t_ckpt:.4g} sec)"
            sepstr = '-' * len(astr)
            log.info(sepstr)
            log.info(astr)

        dist.barrier()

    def save_model(self):
        if RANK == 0:
            astr = f"Finished training. Saving model to {self.model_path}."
            log.info(astr)
            if WITH_CUDA or WITH_XPU:
                self.model.to('cpu')
            if not os.path.exists(self.cfg.model_dir):
                os.makedirs(self.cfg.model_dir)

            if WITH_DDP and SIZE > 1:
                sd = self.model.module.state_dict()
                ind = self.model.module.input_dict()
            else:
                sd = self.model.state_dict()
                ind = self.model.input_dict()

            save_dict = {
                        'state_dict' : sd,
                        'input_dict' : ind,
                        'loss_hist_train' : self.loss_hist_train,
                        'loss_hist_val' : self.loss_hist_val,
                        'iteration' : self.iteration,
                        }
            torch.save(save_dict, self.model_path)


    def build_model(self) -> nn.Module:
        if RANK == 0:
            log.info('In build_model...')

        sample = self.data['train']['example']
        graph = self.data['graph']

        # Get the polynomial order -- for naming the model
        try:
            poly = np.cbrt(self.Np) - 1.
            poly = int(poly)
        except:
            poly = 0

        # Model architecture
        arch = {
            'input_node_features': sample['x'].shape[1],
            'input_edge_features': graph.edge_attr.shape[1],
            'mlp_hidden_channels': self.cfg.mlp_hidden_channels,
            'n_mlp_hidden_layers': self.cfg.n_mlp_hidden_layers,
            'n_messagePassing_layers': self.cfg.n_messagePassing_layers,
            'halo_swap_mode': self.cfg.halo_swap_mode,
            'layer_norm': self.cfg.layer_norm,
            'dropout_rate': self.cfg.dropout_rate,
            'emb_width': self.cfg.emb_width,
            'name': 'POLY_%d_SIZE_%d_SEED_%d' %(poly,SIZE,self.cfg.seed)
        }
        # TODO: this is where things like Re, num_pins, distance to wall, etc. would go
        arch['cond_node_features'] = 0

        model = gnn.DistributedDGN(arch)
        return model

    def count_weights(self, model) -> int:
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return n_params

    def build_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        #optimizer = optim.Adam(model.parameters(), lr=0.0)
        optimizer = optim.AdamW(model.parameters(), lr=0.0, betas=(0.9, 0.95), weight_decay=0.1)
        return optimizer

    def build_scheduler(self, optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                patience=5, threshold=0.0001, threshold_mode='rel',
                                cooldown=0, min_lr=1e-8, eps=1e-08, verbose=True)
        return scheduler

    def setup_torch(self):
        # Random seeds
        torch.manual_seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)

        # Device and intra-op threads
        if WITH_CUDA:
            torch.cuda.set_device(DEVICE_ID + self.cfg.device_skip)
        elif WITH_XPU:
            torch.xpu.set_device(DEVICE_ID + self.cfg.device_skip)
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

    def halo_swap(self, input_tensor, buff_send, buff_recv):
        """
        Performs halo swap using send/receive buffers
        """
        if SIZE > 1:
            # Fill send buffer
            for i in self.neighboring_procs:
                buff_send[i] = input_tensor[self.mask_send[i]]

            # Perform swap
            req_send_list = []
            for i in self.neighboring_procs:
                req_send = dist.isend(tensor=buff_send[i], dst=i)
                req_send_list.append(req_send)
            
            req_recv_list = []
            for i in self.neighboring_procs:
                req_recv = dist.irecv(tensor=buff_recv[i], src=i)
                req_recv_list.append(req_recv)

            for req_send in req_send_list:
                req_send.wait()

            for req_recv in req_recv_list:
                req_recv.wait()

            dist.barrier()

            # Fill halo nodes 
            for i in self.neighboring_procs:
                input_tensor[self.mask_recv[i]] = buff_recv[i]
        return input_tensor 

    def build_masks(self):
        """
        Builds index masks for facilitating halo swap of nodes 
        """
        mask_send = [torch.tensor([], dtype=self.torch_dtype)] * SIZE
        mask_recv = [torch.tensor([], dtype=self.torch_dtype)] * SIZE

        if SIZE > 1 and self.cfg.consistency: 
            #n_nodes_local = self.data.n_nodes_internal + self.data.n_nodes_halo
            #halo_info = self.data['train']['example'].halo_info
            halo_info = self.data['graph'].halo_info

            for i in self.neighboring_procs:
                idx_i = halo_info[:,3] == i
                # index of nodes to send to proc i 
                mask_send[i] = halo_info[:,0][idx_i] 
                
                # index of nodes to receive from proc i  
                mask_recv[i] = halo_info[:,1][idx_i]

                if len(mask_send[i]) != len(mask_recv[i]): 
                    log.info('For neighbor rank %d, the number of send nodes and the number of receive nodes do not match. Check to make sure graph is partitioned correctly.' %(i))
                    utils.force_abort()
        return mask_send, mask_recv 

    def build_buffers(self, n_features):
        n_max = 0
        
        if SIZE == 1:
            buff_send = [torch.tensor([], dtype=self.torch_dtype)] * SIZE
            buff_recv = [torch.tensor([], dtype=self.torch_dtype)] * SIZE 
        else: 
            # Get the maximum number of nodes that will be exchanged (required for all_to_all halo swap)
            n_nodes_to_exchange = torch.zeros(SIZE)
            for i in self.neighboring_procs:
                n_nodes_to_exchange[i] = len(self.mask_send[i])
            n_max = n_nodes_to_exchange.max()
            if WITH_CUDA or WITH_XPU: 
                n_max = n_max.to(self.device)
            dist.all_reduce(n_max, op=dist.ReduceOp.MAX)
            n_max = int(n_max)

            # fill the buffers -- make all buffer sizes the same (required for all_to_all) 
            if self.cfg.halo_swap_mode == "none":
                buff_send = [torch.empty(0, device=DEVICE, dtype=self.torch_dtype)] * SIZE
                buff_recv = [torch.empty(0, device=DEVICE, dtype=self.torch_dtype)] * SIZE
            elif self.cfg.halo_swap_mode == "all_to_all":
                buff_send = [torch.empty(0, device=DEVICE, dtype=self.torch_dtype)] * SIZE
                buff_recv = [torch.empty(0, device=DEVICE, dtype=self.torch_dtype)] * SIZE
                for i in range(SIZE): 
                    buff_send[i] = torch.empty([n_max, n_features], dtype=self.torch_dtype, device=DEVICE) 
                    buff_recv[i] = torch.empty([n_max, n_features], dtype=self.torch_dtype, device=DEVICE)
            elif self.cfg.halo_swap_mode == "all_to_all_opt":
                buff_send = [torch.empty(0, device=DEVICE, dtype=self.torch_dtype)] * SIZE
                buff_recv = [torch.empty(0, device=DEVICE, dtype=self.torch_dtype)] * SIZE
                for i in self.neighboring_procs:
                    buff_send[i] = torch.empty([int(n_nodes_to_exchange[i]), n_features], dtype=self.torch_dtype, device=DEVICE) 
                    buff_recv[i] = torch.empty([int(n_nodes_to_exchange[i]), n_features], dtype=self.torch_dtype, device=DEVICE)
            elif self.cfg.halo_swap_mode == "all_to_all_opt_intel":
                buff_send = [torch.zeros(1, device=DEVICE, dtype=self.torch_dtype)] * SIZE
                buff_recv = [torch.zeros(1, device=DEVICE, dtype=self.torch_dtype)] * SIZE
                for i in self.neighboring_procs:
                    buff_send[i] = torch.zeros([int(n_nodes_to_exchange[i]), n_features], dtype=self.torch_dtype, device=DEVICE) 
                    buff_recv[i] = torch.zeros([int(n_nodes_to_exchange[i]), n_features], dtype=self.torch_dtype, device=DEVICE)
            elif self.cfg.halo_swap_mode == "send_recv":
                buff_send = [torch.empty(0, device=DEVICE, dtype=self.torch_dtype)] * SIZE
                buff_recv = [torch.empty(0, device=DEVICE, dtype=self.torch_dtype)] * SIZE
                for i in self.neighboring_procs:
                    buff_send[i] = torch.empty([int(n_nodes_to_exchange[i]), n_features], dtype=self.torch_dtype, device=DEVICE) 
                    buff_recv[i] = torch.empty([int(n_nodes_to_exchange[i]), n_features], dtype=self.torch_dtype, device=DEVICE)

            #for i in self.neighboring_procs:
            #    buff_send[i] = torch.empty([len(self.mask_send[i]), n_features], dtype=torch.float32, device=DEVICE_ID) 
            #    buff_recv[i] = torch.empty([len(self.mask_recv[i]), n_features], dtype=torch.float32, device=DEVICE_ID)
        
            # Measure the size of the buffers
            buff_send_sz = [0] * SIZE
            buff_recv_sz = [0] * SIZE
            for i in range(SIZE): 
                buff_send_sz[i] = torch.numel(buff_send[i])*buff_send[i].element_size()/1024
                buff_recv_sz[i] = torch.numel(buff_recv[i])*buff_recv[i].element_size()/1024
        
            # Print information about the buffers
            if RANK == 0:
                log.info('[RANK %d]: Created send and receive buffers for %s halo exchange:' %(RANK,self.cfg.halo_swap_mode))
                log.info(f'[RANK {RANK}]: Send buffers of size [KB]: {buff_send_sz}')
                log.info(f'[RANK {RANK}]: Receive buffers of size [KB]: {buff_recv_sz}')
            elif self.cfg.verbose: 
                log.info('[RANK %d]: Created send and receive buffers for %s halo exchange:' %(RANK,self.cfg.halo_swap_mode))
                log.info(f'[RANK {RANK}]: Send buffers of size [KB]: {buff_send_sz}')
                log.info(f'[RANK {RANK}]: Receive buffers of size [KB]: {buff_recv_sz}')

        return buff_send, buff_recv, n_max 

    def init_send_buffer(self, n_buffer_rows, n_features, device):
        buff_send = [torch.tensor([])] * SIZE
        if SIZE > 1: 
            for i in range(SIZE): 
                buff_send[i] = torch.empty([n_buffer_rows, n_features], dtype=self.torch_dtype, device=device) 
        return buff_send 

    def gather_node_tensor(self, input_tensor, dst=0, dtype=torch.float32):
        """
        Gathers node-based tensor into root proc. Shape is [n_internal_nodes, n_features] 
        NOTE: input tensor on all ranks should correspond to INTERNAL nodes (exclude halo nodes) 
        n_internal_nodes can vary for each proc, but n_features must be the same 
        """
        # torch.distributed.gather(tensor, gather_list=None, dst=0, group=None, async_op=False)
        n_nodes = torch.tensor(input_tensor.shape[0])
        n_features = torch.tensor(input_tensor.shape[1])

        n_nodes_procs = list(torch.empty([1], dtype=torch.int64, device=DEVICE)) * SIZE
        if WITH_CUDA or WITH_XPU:
            n_nodes = n_nodes.to(self.device)
        dist.all_gather(n_nodes_procs, n_nodes)

        gather_list = None
        if RANK == 0:
            gather_list = [None] * SIZE
            for i in range(SIZE):
                gather_list[i] = torch.empty([n_nodes_procs[i], n_features], 
                                             dtype=dtype,
                                             device=DEVICE)
        dist.gather(input_tensor, gather_list, dst=0)
        return gather_list

    def load_data(self, file_name, 
                  dtype: Optional[type] = np.float64, 
                  extension: Optional[str] = ""
    ):
        if not self.cfg.online:
            # check extension anyway
            ext = file_name.split('.')[-1]
            if extension == ".bin" or ext == "bin":
                data = np.fromfile(file_name+extension, dtype=dtype)
            elif extension == ".npy" or ext == "npy":
                data = np.load(file_name+extension)
            elif extension == ".npz" or ext == "npz":
                data = np.load(file_name+extension)
            else:
                data = np.loadtxt(file_name, dtype=dtype)
        else:
            data = self.client.get_array(file_name).astype(dtype)
            if isinstance(file_name, str):
                if 'edge_index' not in file_name:
                    data = data.T
            else:
                data = data.T
        return data

    def load_graph_data(self):
        """
        Load in the local graph
        """
        if RANK == 0: log.info('Setting up the graph ...')
        if not self.cfg.online:
            main_path = self.cfg.gnn_outputs_path + '/'
        else:
            main_path = ""
        
        path_to_pos_full = main_path + 'pos_node_rank_%d_size_%d' %(RANK,SIZE)
        path_to_ei = main_path + 'edge_index_rank_%d_size_%d' %(RANK,SIZE)
        path_to_overlap = main_path + 'overlap_ids_rank_%d_size_%d' %(RANK,SIZE)
        path_to_glob_ids = main_path + 'global_ids_rank_%d_size_%d' %(RANK,SIZE)
        path_to_unique_local = main_path + 'local_unique_mask_rank_%d_size_%d' %(RANK,SIZE)
        path_to_unique_halo = main_path + 'halo_unique_mask_rank_%d_size_%d' %(RANK,SIZE)

        # Polynomial order
        self.Np = np.array([0], dtype=np.float32)
        if RANK == 0:
            path_to_Np = main_path + "Np_rank_%d_size_%d" %(RANK, SIZE)
            self.Np = self.load_data(path_to_Np, dtype=np.float32)
        COMM.Bcast(self.Np, root=0)

        # Node positions
        if self.cfg.verbose: log.info('[RANK %d]: Loading positions and global node index' %(RANK))
        #pos = np.fromfile(self.cfg.gnn_outputs_path+'/'+path_to_pos_full + ".bin", dtype=np.float64).reshape((-1,3))
        pos = self.load_data(path_to_pos_full, extension='.bin').reshape((-1,3))

        # Global node index
        #gli = np.fromfile(self.cfg.gnn_outputs_path+'/'+path_to_glob_ids + ".bin", dtype=np.int64).reshape((-1,1))
        gli = self.load_data(path_to_glob_ids,dtype=np.int64,extension='.bin').reshape((-1,1))

        # Edge index
        if self.cfg.verbose: log.info('[RANK %d]: Loading edge index' %(RANK))
        #ei = np.fromfile(self.cfg.gnn_outputs_path+'/'+path_to_ei + ".bin", dtype=np.int32).reshape((-1,2)).T
        ei = self.load_data(path_to_ei, dtype=np.int32, extension='.bin') 
        if not self.cfg.online:
            ei = ei.reshape((-1,2)).T
        ei = ei.astype(np.int64) # sb: int64 for edge_index

        # Local unique mask
        if self.cfg.verbose: log.info('[RANK %d]: Loading local unique mask' %(RANK))
        #local_unique_mask = np.fromfile(self.cfg.gnn_outputs_path+'/'+path_to_unique_local + ".bin", dtype=np.int32)
        local_unique_mask = self.load_data(path_to_unique_local,dtype=np.int32,extension='.bin')

        # Halo unique mask
        halo_unique_mask = np.array([])
        if SIZE > 1:
            #halo_unique_mask = np.fromfile(self.cfg.gnn_outputs_path+'/'+path_to_unique_halo + ".bin", dtype=np.int32)
            halo_unique_mask = self.load_data(path_to_unique_halo,dtype=np.int32,extension='.bin')

        return pos, gli, ei, local_unique_mask, halo_unique_mask

    def setup_local_graph(self):
        """
        Setup the local graph
        """
        # ~~~~ Read the graph data structures
        pos, gli, ei, local_unique_mask, halo_unique_mask = self.load_graph_data()

        # We are only periodic in z for the BFS. so we do the following:  
        pos = pos.astype(NP_FLOAT_DTYPE)
        pos_orig = np.copy(pos)
        L_z = 2. 
        # pos[:,2] = np.cos(2.*np.pi*pos[:,2]/L_z) # cosine
        pos[:,2] = np.abs((pos[:,2] % L_z) - L_z / 2) # piecewise linear 

        # ~~~~ Make the full graph: 
        if self.cfg.verbose: log.info('[RANK %d]: Making the FULL GLL-based graph with overlapping nodes' %(RANK))
        data_full = Data(x = None, 
                         edge_index = torch.tensor(ei), 
                         pos_orig = torch.tensor(pos_orig), 
                         pos = torch.tensor(pos), 
                         global_ids = torch.tensor(gli.squeeze()), 
                         local_unique_mask = torch.tensor(local_unique_mask), 
                         halo_unique_mask = torch.tensor(halo_unique_mask)
        )
        data_full.edge_index = pyg_utils.remove_self_loops(data_full.edge_index)[0]
        data_full.edge_index = pyg_utils.coalesce(data_full.edge_index)
        data_full.edge_index = pyg_utils.to_undirected(data_full.edge_index)
        data_full.local_ids = torch.tensor(range(data_full.pos.shape[0]))

        # ~~~~ Get reduced (non-overlapping) graph and indices to go from full to reduced  
        if self.cfg.verbose: log.info('[RANK %d]: Making the REDUCED GLL-based graph with non-overlapping nodes' %(RANK))
        data_reduced, idx_full2reduced = gcon.get_reduced_graph(data_full)

        # ~~~~ Get the indices to go from reduced back to full graph  
        # idx_reduced2full = None
        if self.cfg.verbose: log.info('[RANK %d]: Getting idx_reduced2full' %(RANK))
        idx_reduced2full = gcon.get_upsample_indices(data_full, data_reduced, idx_full2reduced)

        return data_reduced, data_full, idx_full2reduced, idx_reduced2full

    def setup_halo(self):
        if SIZE > 1 and self.cfg.consistency:
            if self.cfg.verbose: log.info('[RANK %d]: Assembling halo_ids_list using reduced graph' %(RANK))
            if not self.cfg.online:
                path_to_ew = self.cfg.gnn_outputs_path + '/edge_weights_rank_%d_size_%d' %(RANK,SIZE)
                path_to_node_degree = self.cfg.gnn_outputs_path + '/node_degree_rank_%d_size_%d' %(RANK,SIZE)
                path_to_halo_info = self.cfg.gnn_outputs_path + '/halo_info_rank_%d_size_%d' %(RANK,SIZE)
                edge_freq = torch.tensor(self.load_data(path_to_ew,extension='.npy'), dtype=self.torch_dtype)
                edge_weight = 1.0/edge_freq 
                node_degree = torch.tensor(self.load_data(path_to_node_degree,extension='.npy'), dtype=self.torch_dtype)
                halo_info = torch.tensor(self.load_data(path_to_halo_info,extension='.npy'))
            else:
                if self.client.file_exists(f'halo_info_rank_{RANK}_size_{SIZE}'):
                    halo_info = torch.tensor(self.client.get_array(f'halo_info_rank_{RANK}_size_{SIZE}'))
                    node_degree = torch.tensor(self.client.get_array(f'node_degree_rank_{RANK}_size_{SIZE}'))
                    edge_weight = torch.tensor(self.client.get_array(f'edge_weight_rank_{RANK}_size_{SIZE}'))
                else:
                    tic = time.time()
                    halo_ids = create_halo_info_par.get_reduced_halo_ids(self.data_reduced)
                    halo_info_glob = create_halo_info_par.get_halo_info_fast(self.data_reduced, halo_ids)
                    if RANK ==0: log.info('[RANK %d]: computed halo info in %f sec' %(RANK,time.time()-tic))
                    halo_info = halo_info_glob[RANK]
                    self.client.put_array(f'halo_info_rank_{RANK}_size_{SIZE}', halo_info.numpy())

                    tic = time.time()
                    node_degree = create_halo_info_par.get_node_degree(self.data_reduced, halo_info)
                    if RANK ==0: log.info('[RANK %d]: computed node degree in %f sec' %(RANK,time.time()-tic))
                    self.client.put_array(f'node_degree_rank_{RANK}_size_{SIZE}', node_degree.numpy())

                    tic = time.time()
                    edge_freq = create_halo_info_par.get_edge_weights(self.data_reduced, halo_info_glob)
                    edge_weight = (1.0/edge_freq).to(self.torch_dtype)
                    if RANK ==0: log.info('[RANK %d]: computed edge weights in %f sec' %(RANK,time.time()-tic))
                    self.client.put_array(f'edge_weight_rank_{RANK}_size_{SIZE}', edge_weight.to(torch.float32).numpy())

            self.neighboring_procs = np.unique(halo_info[:,3])
            n_nodes_local = self.data_reduced.pos.shape[0]
            n_nodes_halo = halo_info.shape[0]
            if self.cfg.verbose: 
                log.info(f'[RANK {RANK}]: Found {len(self.neighboring_procs)} neighboring processes: {self.neighboring_procs}')
            else:
                if RANK == 0: log.info(f'[RANK {RANK}]: Found {len(self.neighboring_procs)} neighboring processes: {self.neighboring_procs}')
        else:
            halo_info = torch.zeros(1, dtype=self.torch_dtype)
            n_nodes_local = self.data_reduced.pos.shape[0]
            n_nodes_halo = 0
            edge_weight = torch.zeros(1, dtype=self.torch_dtype)
            node_degree = torch.zeros(1, dtype=self.torch_dtype)

        self.data_reduced.n_nodes_local = torch.tensor(n_nodes_local, dtype=torch.int64)
        self.data_reduced.n_nodes_halo = torch.tensor(n_nodes_halo, dtype=torch.int64)
        self.data_reduced.halo_info = halo_info
        self.data_reduced.edge_weight = edge_weight
        self.data_reduced.node_degree = node_degree
        return 

    def prepare_snapshot_data(self, data_x: np.ndarray):
        data_x = data_x.astype(NP_FLOAT_DTYPE) # force NP_FLOAT_DTYPE
         
        # Retain only N_gll = Np*Ne elements
        N_gll = self.data_full.pos.shape[0]
        data_x = data_x[:N_gll, :]

        # get data in reduced format 
        data_x_reduced = data_x[self.idx_full2reduced, :] 
        x = torch.tensor(data_x_reduced, dtype=torch.float32)

        # Add halo nodes by appending the end of the node arrays
        if self.cfg.consistency:
            n_nodes_halo = self.data_reduced.n_nodes_halo
            n_features_x = data_x_reduced.shape[1]
            data_x_halo = torch.zeros((n_nodes_halo, n_features_x), dtype=torch.float32)
            x = torch.cat((x, data_x_halo), dim=0)
        return x

    def compute_statistics(self, data_list: list, var: str):
        device = 'cpu'
        n_features = data_list[0][var].shape[1]
        n_nodes_local = self.data_reduced.n_nodes_local
        n_snaps = len(data_list)
        x_full = torch.zeros((n_snaps, n_nodes_local, n_features), dtype=self.torch_dtype)
        for i in range(len(data_list)):
            x_full[i,:,:] = data_list[i][var][:n_nodes_local, :]
        data_mean_ = x_full.mean(axis=(0,1)).to(device)
        data_var_ = x_full.var(axis=(0,1)).to(device)
        n_scale_ = torch.tensor([n_nodes_local * n_snaps], dtype=self.torch_dtype, device=device)

        data_mean_gather = [torch.zeros(n_features, dtype=self.torch_dtype, device=device) for _ in range(SIZE)]
        data_mean_gather = utils.mpi_all_gather(data_mean_)

        data_var_gather = [torch.zeros(n_features, dtype=self.torch_dtype, device=device) for _ in range(SIZE)]
        data_var_gather = utils.mpi_all_gather(data_var_)

        n_scale_gather = [torch.zeros(1, dtype=self.torch_dtype, device=device) for _ in range(SIZE)]
        n_scale_gather = utils.mpi_all_gather(n_scale_)

        data_mean_gather = torch.stack(data_mean_gather)
        data_var_gather = torch.stack(data_var_gather)
        n_scale_gather = torch.stack(n_scale_gather)

        data_mean = torch.sum(n_scale_gather * data_mean_gather, axis=0)/torch.sum(n_scale_gather)
        data_mean = data_mean.unsqueeze(0)
            
        num_1 = torch.sum(n_scale_gather * data_var_gather, axis=0) # n_i * var_i
        num_2 = torch.sum(n_scale_gather * (data_mean_gather - data_mean)**2, axis=0)
        data_var = (num_1 + num_2)/torch.sum(n_scale_gather)
        data_std = torch.sqrt(data_var)
        data_std = data_std.unsqueeze(0)
        return data_mean, data_std

    def load_field_data(self, data_dir: str):
        if RANK == 0: log.info("Loading field data...")
        field_name = 'u' # velocity

        # read files
        if not self.cfg.online:
            file_list = os.listdir(data_dir)
            files = [item for item in file_list \
                            if (f'fld_{field_name}' in item) and (f'rank_{RANK}' in item)]
            files.sort(key=lambda x:int(x.split('.')[0].split('_')[-1]))
        else:
            log.warning('Online backends not implemented for this model yet')
            MPI.Abort(COMM, 1)

        # populate dataset
        if not self.cfg.online:
            path_prepend = data_dir + '/'
            files = [path_prepend+file for file in files]
        log.info(f'[RANK {RANK}]: Found {len(files)} field files to load')
        for i in range(len(files)):
            tic = time.time()
            data_x = self.load_data(files[i], dtype=np.float64).reshape((-1,3))
            toc = time.time()
            if self.cfg.online:
                self.online_timers['trainDataTime'].append(toc-tic)
                self.online_timers['trainDataThroughput'].append(data_x.nbytes/GB_SIZE/(toc-tic))
            data_x = self.prepare_snapshot_data(data_x)
            self.data_list.append({'x': data_x})

        # split into train/validation
        data = {'train': [], 'validation': []}
        fraction_valid = 0.0
        if fraction_valid > 0 and len(self.data_list)*fraction_valid > 1:
            # How many total snapshots to extract
            n_full = len(self.data_list)
            n_valid = int(np.floor(fraction_valid * n_full))

            # Get validation set indices
            idx_valid = np.sort(np.random.choice(n_full, n_valid, replace=False))

            # Get training set indices
            idx_train = np.array(list(set(list(range(n_full))) - set(list(idx_valid))))

            # Train/validation split
            data['train'] = [self.data_list[i] for i in idx_train]
            data['validation'] = [self.data_list[i] for i in idx_valid]
        else:
            data['train'] = self.data_list
            data['validation'] = [{}]

        if RANK == 0: log.info(f"Number of training snapshots: {len(data['train'])}")
        if RANK == 0: log.info(f"Number of validation snapshots: {0}")
        
        # Compute statistics for normalization
        stats = {'x': []}
        if 'stats' not in self.data.keys():
            if os.path.exists(data_dir + f"/data_stats.npz"):
                if RANK == 0:
                    npzfile = np.load(data_dir + f"/data_stats.npz")
                    stats_arr_x = np.stack([npzfile['x_mean'][0], npzfile['x_std'][0]])
                else:
                    n_features = self.data_list[0]['x'].shape[1]
                    stats_arr_x = np.zeros((2,n_features), dtype=np.float32)
                COMM.Bcast(stats_arr_x, root=0)
                stats['x'] = [stats_arr_x[0], stats_arr_x[1]]
                if RANK == 0: log.info(f"Read training data statistics from {data_dir}/data_stats.npz")
            else: 
                x_mean, x_std = self.compute_statistics(data['train'],'x')
                if RANK == 0 and not self.cfg.online:
                    np.savez(data_dir + f"/data_stats.npz", 
                        x_mean=x_mean, x_std=x_std,
                    )
                stats['x'] = [x_mean, x_std]
                if RANK == 0: log.info(f"Computed training data statistics for each node feature")
        return data, stats
 
    def setup_data(self):
        """
        Generate the PyTorch Geometric Dataset 
        """
        if RANK == 0:
            log.info('In setup_data...')

        device_for_loading = 'cpu'

        data_dir = self.cfg.gnn_outputs_path
        data, stats = self.load_field_data(data_dir)
        
        # Get dictionary 
        reduced_graph_dict = self.data_reduced.to_dict()

        # Create training dataset -- only 1 snapshot for demo
        data_graph = Data()
        for key in reduced_graph_dict.keys():
            data_graph[key] = reduced_graph_dict[key]
        if self.cfg.consistency:
            n_nodes_halo = self.data_reduced.n_nodes_halo 
            n_features_pos = self.data_reduced.pos.shape[1]
            pos_halo = torch.zeros((n_nodes_halo, n_features_pos), dtype=self.torch_dtype)
            data_graph.pos = torch.cat((data_graph.pos, pos_halo), dim=0)
        else:
            data_graph.pos = data_graph.pos
        #data_temp.node_degree = torch.cat((data_temp.node_degree, node_degree_halo), dim=0)
        #data_temp.edge_index = torch.cat((data_temp.edge_index, edge_index_halo), dim=1)
        #data_temp.edge_weight = torch.cat((data_temp.edge_weight, edge_weight_halo), dim=0)
        #data_temp.edge_weight_temp = data_temp.edge_weight

        # Populate edge_attrs
        cart = torch_geometric.transforms.Cartesian(norm=False, max_value = None, cat = False)
        dist = torch_geometric.transforms.Distance(norm = False, max_value = None, cat = True)
        data_graph = cart(data_graph) # adds cartesian/component-wise distance
        data_graph = dist(data_graph) # adds euclidean distance
        data_graph = data_graph.to(device_for_loading)

        # Normalize edge_attrs by length of the longest edge 
        distance = data_graph.edge_attr[:,-1]
        distance_max_ = distance.max().to(self.device)
        distance_max = distnn.all_reduce(distance_max_, op=distnn.ReduceOp.MAX).to(device_for_loading)
        data_graph.edge_attr = (data_graph.edge_attr/distance_max).to(self.torch_dtype)

        # ~~~~ Populate the data loader
        # No need for distributed sampler -- create standard dataset loader  
        # We can use the standard pytorch dataloader on (x,y) 
        if (RANK == 0):
            log.info(f"{data_graph}")
            log.info(f"shape of x: {data['train'][0]['x'].shape}")
        train_data_scaled = []
        for item in  data['train']:
            #tdict = {}
            #data = ((item['x'] - stats['x'][0])/(stats['x'][1] + SMALL)).to(self.torch_dtype)
            train_data_scaled.append(Data(x=((item['x'] - stats['x'][0])/(stats['x'][1] + SMALL)).to(self.torch_dtype)))
        train_loader = DataLoader(train_data_scaled,
                                  batch_size=self.cfg.batch_size,
                                  shuffle=True)

        val_data_scaled = data['validation'].copy()
        if val_data_scaled[0]:
            for item in  val_data_scaled:
                #tdict = {}
                #tdict['x'] = ((item['x'] - stats['x'][0])/(stats['x'][1] + SMALL)).to(self.torch_dtype)
                val_data_scaled.append(Data(x=((item['x'] - stats['x'][0])/(stats['x'][1] + SMALL)).to(self.torch_dtype)))
        valid_loader = DataLoader(val_data_scaled,
                                  batch_size=self.cfg.val_batch_size,
                                  shuffle=False)

        self.data =  {
            'train': {
                'loader': train_loader,
                'example': data['train'][0],
            },
            'validation': {
                'loader': valid_loader,
                'example': data['validation'][0],
            },
            'stats': {
                'x_mean': stats['x'][0],
                'x_std': stats['x'][1],
            },
            'graph': data_graph
        }

    def setup_timers(self, n_record: int) -> dict:
        timers = {}
        timers['forwardPass'] = np.zeros(n_record)
        timers['backwardPass'] = np.zeros(n_record)
        timers['loss'] = np.zeros(n_record)
        timers['optimizerStep'] = np.zeros(n_record)
        timers['dataTransfer'] = np.zeros(n_record)
        timers['bufferInit'] = np.zeros(n_record)
        timers['collectives'] = np.zeros(n_record)
        timers['dataTransfer'] = np.zeros(n_record)
        return timers
    
    def setup_online_timers(self) -> dict:
        timers = {}
        timers['metaData'] = []
        timers['trainDataTime'] = []
        timers['trainDataSize'] = []
        timers['trainDataThroughput'] = []
        return timers

    def update_timer(self, key: str, tstep: int, time: float):
        self.timers[key][tstep] = time
        self.synchronize()

    def update_timer_stats(self):
        keys = self.timers.keys()
        i = self.timer_step
        for key in keys:
            t_data = np.array(self.timers[key][i], dtype=np.float32)
            if SIZE > 1:
                t_avg = np.empty_like(t_data)
                t_min = np.empty_like(t_data)
                t_max = np.empty_like(t_data)
                COMM.Allreduce(t_data, t_avg, op=MPI.SUM)
                t_avg = t_avg/SIZE
                COMM.Allreduce(t_data, t_min, op=MPI.MIN)
                COMM.Allreduce(t_data, t_max, op=MPI.MAX)
            else:
                t_avg = t_data
                t_min = t_data
                t_max = t_data
            self.timers_avg[key][i] = t_avg #metric_average(torch.tensor( self.timers[key][i] )).item()
            self.timers_min[key][i] = t_min #metric_min(torch.tensor( self.timers[key][i] )).item()
            self.timers_max[key][i] = t_max #metric_max(torch.tensor( self.timers[key][i] )).item()
            #if RANK == 0:
            #    log.info(f"t_{key} [min,max,avg] = [{self.timers_min[key][i]},{self.timers_max[key][i]},{self.timers_avg[key][i]}]") 
        return

    def collect_timer_stats(self) -> None:
        self.timer_stats = {}
        for key, val in self.timers.items():
            times = np.delete(val,[0,1])
            times = times[times != 0]
            collected_arr = np.zeros((times.size*SIZE))
            COMM.Gather(times,collected_arr,root=0)
            avg = np.mean(collected_arr)
            std = np.std(collected_arr)
            minn = np.amin(collected_arr); min_loc = [minn, 0]
            maxx = np.amax(collected_arr); max_loc = [maxx, 0]
            summ = np.sum(collected_arr)
            stats = {
                "avg": avg,
                "std": std,
                "sum": summ,
                "min": [min_loc[0],min_loc[1]],
                "max": [max_loc[0],max_loc[1]]
            }
            self.timer_stats[key] = stats

    def print_timer_stats(self) -> None:
        for key, val in self.timer_stats.items():
            stats_string = f": min = {val['min'][0]:>6e} , " + \
                           f"max = {val['max'][0]:>6e} , " + \
                           f"avg = {val['avg']:>6e} , " + \
                           f"std = {val['std']:>6e} "
            log.info(f"{key} [s] " + stats_string)

    def synchronize(self):
        if WITH_CUDA:
            torch.cuda.synchronize()
        if WITH_XPU:
            torch.xpu.synchronize()

    def train_step(self, data: Data) -> Tensor:
        loss = torch.tensor([0.0])
        graph = self.data['graph']
        tic = time.time()
        if WITH_CUDA or WITH_XPU:
            data = data.to(self.device)
            graph.edge_index = graph.edge_index.to(self.device)
            graph.edge_attr = graph.edge_attr.to(self.device)
            graph.batch = graph.batch.to(self.device) if graph.batch is not None else None
            graph.halo_info = graph.halo_info.to(self.device)
            graph.edge_weight = graph.edge_weight.to(self.device)
            graph.node_degree = graph.node_degree.to(self.device)
            loss = loss.to(self.device)
        if self.cfg.timers: self.update_timer('dataTransfer', self.timer_step, time.time() - tic)

        self.s_optimizer.zero_grad()

        # re-allocate send buffer 
        tic = time.time()
        if self.cfg.halo_swap_mode != 'none':
            for i in range(SIZE):
                if self.cfg.halo_swap_mode == "all_to_all_opt_intel":
                    self.buffer_send[i] = torch.zeros_like(self.buffer_send[i])
                    self.buffer_recv[i] = torch.zeros_like(self.buffer_recv[i])
                else:
                    self.buffer_send[i] = torch.empty_like(self.buffer_send[i])
                    self.buffer_recv[i] = torch.empty_like(self.buffer_recv[i])
        else:
            self.buffer_send = None
            self.buffer_recv = None
        if self.cfg.timers: self.update_timer('bufferInit', self.timer_step, time.time() - tic)
        
        # Sample a batch of random diffusion steps
        r, weights = self.step_sampler.sample(batch_size=self.cfg.batch_size)
        if self.cfg.verbose:
            if RANK == 0: log.info(f"Sampled diffusion steps: {r.cpu().numpy().tolist()} for batch size {self.cfg.batch_size}")

        # Diffuse the solution/target field
        BC_mask = None # no BCs for now
        field_r, noise = self.diffusion_process.forward(data['x'], r, BC_mask)
        if self.cfg.verbose:
            if RANK == 0: log.info(f"Performed forward diffusion process on {self.cfg.batch_size} solution fields")
            if RANK == 0: 
                log.info(f"Shape of field_r: {field_r.shape}, shape of x: {data['x'].shape}, shape of r: {r.shape}")
                log.info(f"graph.batch: {graph.batch}")
        
        # Prediction
        tic = time.time()
        model_noise, model_var = self.model(
                             field_r = field_r,
                             r = r,
                             edge_index = graph.edge_index,
                             edge_attr = graph.edge_attr,
                             edge_weight = graph.edge_weight,
                             halo_info = graph.halo_info,
                             mask_send = self.mask_send,
                             mask_recv = self.mask_recv,
                             buffer_send = self.buffer_send,
                             buffer_recv = self.buffer_recv,
                             neighboring_procs = self.neighboring_procs,
                             SIZE = SIZE,
                             cond_node_features = None, # TODO: Add conditional node features
                             batch = graph.batch)
        if self.cfg.timers: self.update_timer('forwardPass', self.timer_step, time.time() - tic)

        # Accumulate loss
        tic = time.time()
        n_nodes_local = graph.n_nodes_local
        if SIZE == 1 or not self.cfg.consistency:
            loss = self.loss_fn(model_noise[:n_nodes_local], noise[:n_nodes_local])
            # TODO: Look at DGN4cfd hybrid loss for full loss function
            effective_nodes = n_nodes_local 
        else: # custom 
            if RANK == 0: log.error('Custom loss function not implemented for SIZE > 1')
            MPI.Abort(COMM, 1)
            """
            n_output_features = pred.shape[1]
            squared_errors_local = torch.pow(pred[:n_nodes_local] - target[:n_nodes_local], 2)
            squared_errors_local = squared_errors_local/graph.node_degree[:n_nodes_local].unsqueeze(-1)

            sum_squared_errors_local = squared_errors_local.sum()
            effective_nodes_local = torch.sum(1.0/graph.node_degree[:n_nodes_local])

            effective_nodes = distnn.all_reduce(effective_nodes_local)
            sum_squared_errors = distnn.all_reduce(sum_squared_errors_local)
            loss = (1.0/(effective_nodes*n_output_features)) * sum_squared_errors
            """
        if self.cfg.timers: self.update_timer('loss', self.timer_step, time.time() - tic)

        tic = time.time()
        loss.backward()
        if self.cfg.timers: self.update_timer('backwardPass', self.timer_step, time.time() - tic)

        tic = time.time()
        self.s_optimizer.step_and_update_lr()
        if self.cfg.timers: self.update_timer('optimizerStep', self.timer_step, time.time() - tic)

        # Update timers
        self.synchronize()
        if self.cfg.timers:
            if self.timer_step < self.timer_step_max - 1:
                self.update_timer_stats()
                self.timer_step += 1
        return loss
    
    def inference_step(self, x) -> Tensor:
        graph = self.data['graph']
        stats = self.data['stats']
        tic = time.time()
        if WITH_CUDA or WITH_XPU:
            x = x.to(self.device)
            graph.edge_index = graph.edge_index.to(self.device)
            graph.edge_weight = graph.edge_weight.to(self.device)
            graph.edge_attr = graph.edge_attr.to(self.device)
            graph.batch = graph.batch.to(self.device) if graph.batch is not None else None
            graph.halo_info = graph.halo_info.to(self.device)
            graph.node_degree = graph.node_degree.to(self.device)
        if self.cfg.timers: self.update_timer('dataTransfer', self.timer_step, time.time() - tic)
                
        # re-allocate send buffer 
        tic = time.time()
        if self.cfg.halo_swap_mode != 'none':
            for i in range(SIZE):
                if self.cfg.halo_swap_mode == "all_to_all_opt_intel":
                    self.buffer_send[i] = torch.zeros_like(self.buffer_send[i])
                    self.buffer_recv[i] = torch.zeros_like(self.buffer_recv[i])
                else:
                    self.buffer_send[i] = torch.empty_like(self.buffer_send[i])
                    self.buffer_recv[i] = torch.empty_like(self.buffer_recv[i])
        else:
            self.buffer_send = None
            self.buffer_recv = None
        if self.cfg.timers: self.update_timer('bufferInit', self.timer_step, time.time() - tic)
        
        # Prediction
        tic = time.time()
        #x_scaled = (x[0] - stats['mean'])/(stats['std'] + SMALL)
        x_scaled = x[0] if len(x.shape) > 2 else x
        out_gnn = self.model(x = x_scaled,
                             edge_index = graph.edge_index,
                             edge_attr = graph.edge_attr,
                             edge_weight = graph.edge_weight,
                             halo_info = graph.halo_info,
                             mask_send = self.mask_send,
                             mask_recv = self.mask_recv,
                             buffer_send = self.buffer_send,
                             buffer_recv = self.buffer_recv,
                             neighboring_procs = self.neighboring_procs,
                             SIZE = SIZE,
                             batch = graph.batch)
        if self.cfg.timers: self.update_timer('forwardPass', self.timer_step, time.time() - tic)

        if self.cfg.use_residual: 
            pred = out_gnn + x_scaled
        else:
            pred = out_gnn

        # Update timers
        self.synchronize()
        if self.cfg.timers:
            if self.timer_step < self.timer_step_max - 1:
                self.update_timer_stats()
                self.timer_step += 1

        return pred

    def test(self) -> dict:
        running_loss = torch.tensor(0.)
        count = torch.tensor(0.)
        if WITH_CUDA or WITH_XPU:
            running_loss = running_loss.to(self.device)
            count = count.to(self.device)
        self.model.eval()
        test_loader = self.data['test']['loader']

        with torch.no_grad():
            for data in test_loader:
                loss = torch.tensor([0.0])
                graph = self.data['graph']
        
                if WITH_CUDA or WITH_XPU:
                    data['x'] = data['x'].to(self.device)
                    data['y'] = data['y'].to(self.device)
                    graph.edge_index = graph.edge_index.to(self.device)
                    graph.edge_attr = graph.edge_attr.to(self.device)
                    graph.batch = graph.batch.to(self.device) if graph.batch is not None else None
                    graph.halo_info = graph.halo_info.to(self.device)
                    graph.edge_weight = graph.edge_weight.to(self.device)
                    graph.node_degree = graph.node_degree.to(self.device)
                    loss = loss.to(self.device)


                # re-allocate send buffer
                if self.cfg.halo_swap_mode != 'none':
                    for i in range(SIZE):
                        if self.cfg.halo_swap_mode == "all_to_all_opt_intel":
                            self.buffer_send[i] = torch.zeros_like(self.buffer_send[i])
                            self.buffer_recv[i] = torch.zeros_like(self.buffer_recv[i])
                        else:
                            self.buffer_send[i] = torch.empty_like(self.buffer_send[i])
                            self.buffer_recv[i] = torch.empty_like(self.buffer_recv[i])
                else:
                    self.buffer_send = None
                    self.buffer_recv = None

                out_gnn = self.model(x = data['x'][0],
                             edge_index = graph.edge_index,
                             edge_attr = graph.edge_attr,
                             edge_weight = graph.edge_weight,
                             halo_info = graph.halo_info,
                             mask_send = self.mask_send,
                             mask_recv = self.mask_recv,
                             buffer_send = self.buffer_send,
                             buffer_recv = self.buffer_recv,
                             neighboring_procs = self.neighboring_procs,
                             SIZE = SIZE,
                             batch = graph.batch)   
       
                # Accumulate loss
                target = data['y'][0]
                n_nodes_local = graph.n_nodes_local
                if SIZE == 1 or not self.cfg.consistency:
                    loss = self.loss_fn(out_gnn[:n_nodes_local], target[:n_nodes_local])
                    effective_nodes = n_nodes_local 
                else: # custom 
                    n_output_features = out_gnn.shape[1]
                    squared_errors_local = torch.pow(out_gnn[:n_nodes_local] - target[:n_nodes_local], 2)
                    squared_errors_local = squared_errors_local/graph.node_degree[:n_nodes_local].unsqueeze(-1)

                    sum_squared_errors_local = squared_errors_local.sum()
                    effective_nodes_local = torch.sum(1.0/graph.node_degree[:n_nodes_local])

                    effective_nodes = distnn.all_reduce(effective_nodes_local)
                    sum_squared_errors = distnn.all_reduce(sum_squared_errors_local)
                    loss = (1.0/(effective_nodes*n_output_features)) * sum_squared_errors

                running_loss += loss.item()
                count += 1

            running_loss = running_loss / count
            loss_avg = utils.metric_average(running_loss)

        return {'loss': loss_avg}

    def writeGraphStatistics(self):
        if RANK == 0: log.info(f"In writeGraphStatistics")
        # Write the number of nodes, halo nodes, and edges in each rank of the sub-graph 
        
        if SIZE == 1:
            model = self.model
        else:
            model = self.model.module

        # if path doesnt exist, make it 
        savepath = self.cfg.work_dir + "/outputs/GraphStatistics/weak_scaling" 
        if RANK == 0:
            if not os.path.exists(savepath):
                os.makedirs(savepath)
                log.info("Directory created by root processor.")
            else:
                log.info("Directory already exists.")
        COMM.Barrier()

        # Number of local nodes 
        n_nodes_local = self.data_reduced.n_nodes_local
        n_nodes_halo = self.data_reduced.n_nodes_halo if self.cfg.consistency else 0
        n_edges = self.data_reduced.edge_index.shape[1]

        if self.cfg.verbose:
            log.info(f"[RANK {RANK}] -- number of local nodes: {n_nodes_local}, number of halo nodes: {n_nodes_halo}, number of edges: {n_edges}")
        else:
            if RANK == 0: log.info(f"[RANK {RANK}] -- number of local nodes: {n_nodes_local}, number of halo nodes: {n_nodes_halo}, number of edges: {n_edges}")

        a = {} 
        a['n_nodes_local'] = n_nodes_local
        a['n_nodes_halo'] = n_nodes_halo
        a['n_edges'] = n_edges
        torch.save(a, savepath + '/%s.tar' %(model.get_save_header())) 

    def postprocess(self):
        """ Do some postprocessing.""" 
        # Get gradient norm 
        grads = [
            param.grad.detach().flatten()
            for param in self.model.parameters()
            if param.grad is not None
        ]
        gradnorm = torch.cat(grads).norm()
        dist.barrier()
        return [gradnorm]
