"""
PyTorch DDP integrated with PyGeom for multi-node training
"""
from __future__ import absolute_import, division, print_function, annotations
import os
import sys
import socket
import logging

from typing import Optional, Union, Callable

import numpy as np

import hydra
import time
import torch
#torch.use_deterministic_algorithms(True)
import torch.utils.data
import torch.utils.data.distributed
from torch.cuda.amp.grad_scaler import GradScaler
import torch.multiprocessing as mp
import torch.distributions as tdist 
from torch.profiler import profile, record_function, ProfilerActivity

import torch.distributed as dist
import torch.distributed.nn_mod as distnn
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from omegaconf import DictConfig, OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
Tensor = torch.Tensor

# PyTorch Geometric
import torch_geometric
from torch_geometric.data import Data
import torch_geometric.utils as utils
import torch_geometric.nn as tgnn

# Models
import models.gnn as gnn

# Graph connectivity/plotting 
import graph_connectivity as gcon
import graph_plotting as gplot

log = logging.getLogger(__name__)

TORCH_FLOAT_DTYPE = torch.float32
NP_FLOAT_DTYPE = np.float32

# Get MPI:
try:
    from mpi4py import MPI
    WITH_DDP = True
    LOCAL_RANK = os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK', '0')
    # LOCAL_RANK = os.environ['OMPI_COMM_WORLD_LOCAL_RANK']
    SIZE = MPI.COMM_WORLD.Get_size()
    RANK = MPI.COMM_WORLD.Get_rank()
    COMM = MPI.COMM_WORLD

    WITH_CUDA = torch.cuda.is_available()

    # Override gpu utilization
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
    WITH_DDP = False
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

def force_abort():
    time.sleep(2)
    if WITH_DDP:
        COMM.Abort()
    else:
        sys.exit("Exiting...")

def metric_average(val: Tensor):
    if (WITH_DDP):
        dist.all_reduce(val, op=dist.ReduceOp.SUM)
        return val / SIZE
    return val

def metric_min(val: Tensor):
    if (WITH_DDP):
        dist.all_reduce(val, op=dist.ReduceOp.MIN)
    return val

def metric_max(val: Tensor):
    if (WITH_DDP):
        dist.all_reduce(val, op=dist.ReduceOp.MAX)
    return val

def trace_handler(p):
    output = p.key_averages().table(sort_by="cpu_time_total", row_limit=20)
    #print(output)
    print(temp_test)
    #p.export_chrome_trace("/tmp/trace_" + str(p.step_num) + ".json")

class Trainer:
    def __init__(self, cfg: DictConfig, scaler: Optional[GradScaler] = None):
        self.cfg = cfg
        self.rank = RANK
        if scaler is None:
            self.scaler = None
        #self.device = 'gpu' if torch.cuda.is_available() else 'cpu'
        self.device = 'gpu' if WITH_CUDA else 'cpu'
        self.backend = self.cfg.backend
        if WITH_DDP:
            init_process_group(RANK, SIZE, backend=self.backend)
        
        # ~~~~ Init torch stuff 
        self.setup_torch()

        # ~~~~ Setup local graph 
        self.data_reduced, self.data_full, self.idx_full2reduced, self.idx_reduced2full = self.setup_local_graph()

        # ~~~~ Setup halo nodes 
        self.neighboring_procs = []
        self.setup_halo()

        # ~~~~ Setup data 
        self.data = self.setup_data()
        if RANK == 0: log.info('Done with setup_data')


        # ~~~~ Setup halo exchange masks
        self.mask_send, self.mask_recv = self.build_masks()
        if RANK == 0: log.info('Done with build_masks')
        
        # ~~~~ Initialize send/recv buffers on device (if applicable)
        self.buffer_send, self.buffer_recv, self.n_buffer_rows = self.build_buffers(self.cfg.hidden_channels)
        if RANK == 0: log.info('Done with build_buffers')

        # ~~~~ Build model and move to gpu 
        self.model = self.build_model()
        if self.device == 'gpu':
            self.model.cuda()
        self.model.to(TORCH_FLOAT_DTYPE)

        if RANK == 0: log.info('Done with build_model')

        # ~~~~ Init training and testing loss history 
        self.loss_hist_train = np.zeros(self.cfg.epochs)
        self.loss_hist_test = np.zeros(self.cfg.epochs)
        self.loss_hist_train_iter = np.zeros(100000)
        self.loss_hist_test_iter = np.zeros(100000)

        # ~~~~ Set model and checkpoint savepaths 
        try:
            self.ckpt_path = cfg.ckpt_dir + self.model.get_save_header() + '.tar'
            self.model_path = cfg.model_dir + self.model.get_save_header() + '.tar'
        except (AttributeError) as e:
            self.ckpt_path = cfg.ckpt_dir + 'checkpoint.tar'
            self.model_path = cfg.model_dir + 'model.tar'

        # ~~~~ Load model parameters if we are restarting from checkpoint
        self.epoch = 0
        self.epoch_start = 1
        self.training_iter = 0
        if self.cfg.restart:
            ckpt = torch.load(self.ckpt_path)
            self.model.load_state_dict(ckpt['model_state_dict'])
            self.epoch_start = ckpt['epoch'] + 1
            self.epoch = self.epoch_start
            self.training_iter = ckpt['training_iter']

            self.loss_hist_train = ckpt['loss_hist_train']
            self.loss_hist_test = ckpt['loss_hist_test']

            if len(self.loss_hist_train) < self.cfg.epochs:
                loss_hist_train_new = np.zeros(self.cfg.epochs)
                loss_hist_test_new = np.zeros(self.cfg.epochs)

                loss_hist_train_new[:len(self.loss_hist_train)] = self.loss_hist_train
                loss_hist_test_new[:len(self.loss_hist_test)] = self.loss_hist_test

                self.loss_hist_train = loss_hist_train_new
                self.loss_hist_test = loss_hist_test_new

        # ~~~~ Wrap model in DDP
        if WITH_DDP and SIZE > 1:
            self.model = DDP(self.model)

        # ~~~~ Set loss function
        self.loss_fn = nn.MSELoss()

        # ~~~~ Set optimizer 
        self.optimizer = self.build_optimizer(self.model)

        # ~~~~ Set scheduler 
        self.scheduler = self.build_scheduler(self.optimizer)

        # ~~~~ Load optimizer+scheduler parameters if we are restarting from checkpoint
        if self.cfg.restart:
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            if RANK == 0:
                astr = 'RESTARTING FROM CHECKPOINT -- STATE AT EPOCH %d/%d' %(self.epoch_start-1, self.cfg.epochs)
                sepstr = '-' * len(astr)
                log.info(sepstr)
                log.info(astr)
                log.info(sepstr)

        # ~~~~ Setup train_step timers 
        self.timer_step = 0
        self.timer_step_max = self.cfg.epochs
        self.timers = self.setup_timers(self.timer_step_max)
        self.timers_max = self.setup_timers(self.timer_step_max)
        self.timers_min = self.setup_timers(self.timer_step_max)
        self.timers_avg = self.setup_timers(self.timer_step_max)

    def build_model(self) -> nn.Module:
        if RANK == 0:
            log.info('In build_model...')

        sample = self.data['train']['example'] 
        graph = self.data['graph']

        # Get the polynomial order -- for naming the model  
        try:
            main_path = self.cfg.gnn_outputs_path
            poly = 0
            if RANK == 0:
                Np = np.loadtxt(main_path + "Np_rank_%d_size_%d" %(RANK, SIZE), dtype=np.float32)
                poly = np.cbrt(Np) - 1.
                poly = int(poly)
            poly = COMM.bcast(poly, root=0)
        except FileNotFoundError:
            poly = 0

        # Full model 
        input_node_channels = sample['x'].shape[1]
        input_edge_channels = graph.edge_attr.shape[1]
        hidden_channels = self.cfg.hidden_channels
        output_node_channels = sample['y'].shape[1]
        n_mlp_hidden_layers = self.cfg.n_mlp_hidden_layers
        n_messagePassing_layers = self.cfg.n_messagePassing_layers
        halo_swap_mode = self.cfg.halo_swap_mode
        name = 'POLY_%d_RANK_%d_SIZE_%d_SEED_%d' %(poly,RANK,SIZE,self.cfg.seed) 

        model = gnn.DistributedGNN(input_node_channels,
                           input_edge_channels,
                           hidden_channels,
                           output_node_channels,
                           n_mlp_hidden_layers,
                           n_messagePassing_layers,
                           halo_swap_mode,
                           name)

        return model

    def build_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        """
        DDP: scale learning rate by the number of GPUs
        """
        # optimizer = optim.Adam(model.parameters(),
        #                        lr=SIZE * self.cfg.lr_init)
        optimizer = optim.Adam(model.parameters(),
                               lr=self.cfg.lr_init)

        return optimizer

    def build_scheduler(self, optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                patience=5, threshold=0.0001, threshold_mode='rel',
                                cooldown=0, min_lr=1e-8, eps=1e-08, verbose=True)
        return scheduler

    def setup_torch(self):
        torch.manual_seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)

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
        mask_send = [torch.tensor([])] * SIZE
        mask_recv = [torch.tensor([])] * SIZE

        #mask_send = [None] * SIZE
        #mask_recv = [None] * SIZE

        if SIZE > 1: 
            #n_nodes_local = self.data.n_nodes_internal + self.data.n_nodes_halo
            # sb: test 

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
                    force_abort()
        return mask_send, mask_recv 

    def build_buffers(self, n_features):
        buff_send = [torch.tensor([], device=DEVICE_ID)] * SIZE
        buff_recv = [torch.tensor([], device=DEVICE_ID)] * SIZE
        n_max = 0
        
        if SIZE > 1: 

            # Get the maximum number of nodes that will be exchanged (required for all_to_all based halo swap)
            n_nodes_to_exchange = torch.zeros(SIZE)
            for i in self.neighboring_procs:
                n_nodes_to_exchange[i] = len(self.mask_send[i])
            n_max = n_nodes_to_exchange.max()
            if WITH_CUDA: 
                n_max = n_max.cuda()
            dist.all_reduce(n_max, op=dist.ReduceOp.MAX)
            n_max = int(n_max)

            # fill the buffers -- make all buffer sizes the same (required for all_to_all) 
            if self.cfg.halo_swap_mode == "all_to_all":
                for i in range(SIZE): 
                    buff_send[i] = torch.empty([n_max, n_features], dtype=TORCH_FLOAT_DTYPE, device=DEVICE_ID) 
                    buff_recv[i] = torch.empty([n_max, n_features], dtype=TORCH_FLOAT_DTYPE, device=DEVICE_ID)
            elif self.cfg.halo_swap_mode == "send_recv":
                for i in self.neighboring_procs:
                    buff_send[i] = torch.empty([int(n_nodes_to_exchange[i]), n_features], dtype=TORCH_FLOAT_DTYPE, device=DEVICE_ID) 
                    buff_recv[i] = torch.empty([int(n_nodes_to_exchange[i]), n_features], dtype=TORCH_FLOAT_DTYPE, device=DEVICE_ID)

            #for i in self.neighboring_procs:
            #    buff_send[i] = torch.empty([len(self.mask_send[i]), n_features], dtype=torch.float32, device=DEVICE_ID) 
            #    buff_recv[i] = torch.empty([len(self.mask_recv[i]), n_features], dtype=torch.float32, device=DEVICE_ID)

        return buff_send, buff_recv, n_max 

    def init_send_buffer(self, n_buffer_rows, n_features, device):
        buff_send = [torch.tensor([])] * SIZE
        if SIZE > 1: 
            for i in range(SIZE): 
                buff_send[i] = torch.empty([n_buffer_rows, n_features], dtype=TORCH_FLOAT_DTYPE, device=device) 
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

        n_nodes_procs = list(torch.empty([1], dtype=torch.int64, device=DEVICE_ID)) * SIZE
        if WITH_CUDA:
            n_nodes = n_nodes.cuda()
        dist.all_gather(n_nodes_procs, n_nodes)

        gather_list = None
        if RANK == 0:
            gather_list = [None] * SIZE
            for i in range(SIZE):
                gather_list[i] = torch.empty([n_nodes_procs[i], n_features], 
                                             dtype=dtype,
                                             device=DEVICE_ID)
        dist.gather(input_tensor, gather_list, dst=0)
        return gather_list

    # def gather_node_tensor(self, input_tensor, dst=0, dtype=torch.float32):
    #     """
    #     Gathers node-based tensor into root proc. Shape is [n_internal_nodes, n_features] 
    #     NOTE: input tensor on all ranks should correspond to INTERNAL nodes (exclude halo nodes) 
    #     n_internal_nodes can vary for each proc, but n_features must be the same 
    #     """
    #     # torch.distributed.gather(tensor, gather_list=None, dst=0, group=None, async_op=False)
    #     n_features = input_tensor.shape[1]
    #     gather_list = None
    #     if RANK == 0:
    #         gather_list = [None] * SIZE
    #         for i in range(SIZE):
    #             gather_list[i] = torch.empty([self.n_nodes_internal_procs[i], n_features],
    #                                          dtype=dtype,
    #                                          device=DEVICE_ID)
    #     dist.gather(input_tensor, gather_list, dst=0)
    #     return gather_list 
        
    def setup_local_graph(self):
        """
        Load in the local graph
        """
        main_path = self.cfg.gnn_outputs_path 

        path_to_pos_full = main_path + 'pos_node_rank_%d_size_%d' %(RANK,SIZE)
        path_to_ei = main_path + 'edge_index_rank_%d_size_%d' %(RANK,SIZE)
        path_to_overlap = main_path + 'overlap_ids_rank_%d_size_%d' %(RANK,SIZE)
        path_to_glob_ids = main_path + 'global_ids_rank_%d_size_%d' %(RANK,SIZE)
        path_to_unique_local = main_path + 'local_unique_mask_rank_%d_size_%d' %(RANK,SIZE)
        path_to_unique_halo = main_path + 'halo_unique_mask_rank_%d_size_%d' %(RANK,SIZE)
        
        # ~~~~ Get positions and global node index
        if self.cfg.verbose: log.info('[RANK %d]: Loading positions and global node index' %(RANK))
        pos = np.fromfile(path_to_pos_full + ".bin", dtype=np.float64).reshape((-1,3))
        pos = pos.astype(NP_FLOAT_DTYPE)
        pos = np.cos(pos) # SB: positional encoding for periodic case 

        gli = np.fromfile(path_to_glob_ids + ".bin", dtype=np.int64).reshape((-1,1))

        # ~~~~ Get edge index
        if self.cfg.verbose: log.info('[RANK %d]: Loading edge index' %(RANK))
        ei = np.fromfile(path_to_ei + ".bin", dtype=np.int32).reshape((-1,2)).T
        ei = ei.astype(np.int64) # sb: int64 for edge_index 
        
        # ~~~~ Get local unique mask
        if self.cfg.verbose: log.info('[RANK %d]: Loading local unique mask' %(RANK))
        local_unique_mask = np.fromfile(path_to_unique_local + ".bin", dtype=np.int32)

        # ~~~~ Get halo unique mask
        halo_unique_mask = np.array([])
        if SIZE > 1:
            halo_unique_mask = np.fromfile(path_to_unique_halo + ".bin", dtype=np.int32)

        # ~~~~ Make the full graph: 
        if self.cfg.verbose: log.info('[RANK %d]: Making the FULL GLL-based graph with overlapping nodes' %(RANK))
        data_full = Data(x = None, edge_index = torch.tensor(ei), pos = torch.tensor(pos), global_ids = torch.tensor(gli.squeeze()), local_unique_mask = torch.tensor(local_unique_mask), halo_unique_mask = torch.tensor(halo_unique_mask))
        data_full.edge_index = utils.remove_self_loops(data_full.edge_index)[0]
        data_full.edge_index = utils.coalesce(data_full.edge_index)
        data_full.edge_index = utils.to_undirected(data_full.edge_index)
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
        if self.cfg.verbose: log.info('[RANK %d]: Assembling halo_ids_list using reduced graph' %(RANK))
        main_path = self.cfg.gnn_outputs_path

        halo_info = None
        if SIZE > 1:
            halo_info = torch.tensor(np.load(main_path + '/halo_info_rank_%d_size_%d.npy' %(RANK,SIZE)))
            # Get list of neighboring processors for each processor
            self.neighboring_procs = np.unique(halo_info[:,3])
            n_nodes_local = self.data_reduced.pos.shape[0]
            n_nodes_halo = halo_info.shape[0]
        else:
            #print('[RANK %d] neighboring procs: ' %(RANK), self.neighboring_procs)
            halo_info = torch.Tensor([])
            n_nodes_local = self.data_reduced.pos.shape[0]
            n_nodes_halo = 0

        #if self.cfg.verbose: log.info('[RANK %d] neighboring procs: ' %(RANK), self.neighboring_procs)

        self.data_reduced.n_nodes_local = torch.tensor(n_nodes_local, dtype=torch.int64)
        self.data_reduced.n_nodes_halo = torch.tensor(n_nodes_halo, dtype=torch.int64)
        self.data_reduced.halo_info = halo_info

        return 


    def prepare_snapshot_data(self, path_to_snap: str):
        data_x = np.fromfile(path_to_snap, dtype=np.float64).reshape((-1,3)) 
        data_x = data_x.astype(NP_FLOAT_DTYPE) # force NP_FLOAT_DTYPE
        
        # Retain only N_gll = Np*Ne elements
        N_gll = self.data_full.pos.shape[0]
        data_x = data_x[:N_gll, :]

        # get data in reduced format 
        data_x_reduced = data_x[self.idx_full2reduced, :] 

        # Add halo nodes by appending the end of the node arrays
        n_nodes_halo = self.data_reduced.n_nodes_halo
        n_features_x = data_x_reduced.shape[1]
        data_x_halo = torch.zeros((n_nodes_halo, n_features_x), dtype=TORCH_FLOAT_DTYPE)
        x = torch.tensor(data_x_reduced)
        x = torch.cat((x, data_x_halo), dim=0)
        return x

    def setup_data(self):
        """
        Generate the PyTorch Geometric Dataset 
        """
        if RANK == 0:
            log.info('In setup_data...')

        device_for_loading = 'cpu'

        # data directory
        dtfac = 100
        data_dir = self.cfg.traj_data_path + f"/tinit_75.000000_dtfactor_{dtfac}/data_rank_{RANK}_size_{SIZE}"

        # read files and remove pressure
        files_temp = os.listdir(data_dir)
        files = [item for item in files_temp if 'p_step' not in item] 
        files.sort(key=lambda x:int(x.split('_')[-1].split('.')[0]))

        # populate dataset for single-step predictions 
        idx = list(range(len(files)))
        idx_x = idx[:-1]
        idx_y = idx[1:]
        data_traj = []
        if RANK == 0: log.info("Loading trajectory data...")
        for i in range(len(idx_x)):
            step_x_i = idx_x[i]
            step_y_i = idx_y[i]
            path_x_i = data_dir + "/" + files[idx_x[i]]
            path_y_i = data_dir + "/" + files[idx_y[i]]
            data_x_i = self.prepare_snapshot_data(path_x_i)
            data_y_i = self.prepare_snapshot_data(path_y_i)
            data_traj.append(
                    {'x': data_x_i, 'y':data_y_i, 'step_x':step_x_i, 'step_y':step_y_i} 
                    )

        # split into train/test 
        fraction_valid = 0.1
        if fraction_valid > 0:
            # How many total snapshots to extract 
            n_full = len(idx_x)
            n_valid = int(np.floor(fraction_valid * n_full))

            # Get validation set indices 
            idx_valid = np.sort(np.random.choice(n_full, n_valid, replace=False))

            # Get training set indices 
            idx_train = np.array(list(set(list(range(n_full))) - set(list(idx_valid))))

            # Train/test split 
            data_traj_train = [data_traj[i] for i in idx_train]
            data_traj_valid = [data_traj[i] for i in idx_valid]
        else:
            data_traj_train = data_traj
            data_traj_valid = [{}]

        if RANK == 0: log.info(f"Number of training snapshots: {len(idx_train)}")
        if RANK == 0: log.info(f"Number of validation snapshots: {len(idx_valid)}")

        # Get data in reduced format (non-overlapping)
        pos_reduced = self.data_reduced.pos

        # Read in edge weights 
        path_to_ew = self.cfg.gnn_outputs_path + 'edge_weights_rank_%d_size_%d.npy' %(RANK,SIZE)
        edge_freq = torch.tensor(np.load(path_to_ew), dtype=TORCH_FLOAT_DTYPE)
        self.data_reduced.edge_weight = 1.0/edge_freq

        # Read in node degree
        path_to_node_degree = self.cfg.gnn_outputs_path + 'node_degree_rank_%d_size_%d.npy' %(RANK,SIZE)
        node_degree = torch.tensor(np.load(path_to_node_degree), dtype=TORCH_FLOAT_DTYPE)
        self.data_reduced.node_degree = node_degree

        # Add halo nodes by appending the end of the node arrays  
        n_nodes_halo = self.data_reduced.n_nodes_halo 
        n_features_pos = pos_reduced.shape[1]
        pos_halo = torch.zeros((n_nodes_halo, n_features_pos), dtype=TORCH_FLOAT_DTYPE)

        #node_degree_halo = torch.zeros((n_nodes_halo), dtype=TORCH_FLOAT_DTYPE)

        # # Add self-edges for halo nodes (unused) 
        # n_nodes_local = self.data_reduced.n_nodes_local
        # edge_index_halo = torch.arange(n_nodes_local, n_nodes_local + n_nodes_halo, dtype=torch.int64)
        # edge_index_halo = torch.stack((edge_index_halo,edge_index_halo))

        # Add filler edge weights for these self-edges 
        edge_weight_halo = torch.zeros(n_nodes_halo)

        # Populate data object 
        data_x_reduced = data_traj[0]['x']
        data_y_reduced = data_traj[0]['y']
        n_features_in = data_x_reduced.shape[1]
        n_features_out = data_y_reduced.shape[1]
        n_nodes = self.data_reduced.pos.shape[0]
        
        # Get dictionary 
        reduced_graph_dict = self.data_reduced.to_dict()

        # Create training dataset -- only 1 snapshot for demo
        data_graph = Data()
        for key in reduced_graph_dict.keys():
            data_graph[key] = reduced_graph_dict[key]
        data_graph.pos = torch.cat((data_graph.pos, pos_halo), dim=0)
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

        # No need for distributed sampler -- create standard dataset loader  
        # We can use the standard pytorch dataloader on (x,y) 
        #train_loader = torch_geometric.loader.DataLoader(train_dataset, batch_size=self.cfg.batch_size, shuffle=False)
        #test_loader = torch_geometric.loader.DataLoader(test_dataset, batch_size=self.cfg.test_batch_size, shuffle=False)  
        if (RANK == 0):
            log.info(f"{data_graph}")
            log.info(f"shape of x: {data_traj[0]['x'].shape}")
            log.info(f"shape of y: {data_traj[0]['y'].shape}")
        
        # ~~~~ Populate the data sampler. No need to use torch_geometric sampler -- we assume we have fixed connectivity, and a "GRAPH" batch size of 1. We need a sampler only over the [x,y] pairs (i.e., the elements in data_traj)
        # train_loader = torch_geometric.loader.DataLoader(train_dataset, batch_size=self.cfg.batch_size, shuffle=False)
        assert self.cfg.batch_size == 1, f"batch_size {self.cfg.batch_size} must be set to 1!"
        assert self.cfg.test_batch_size == 1, f"test_batch_size {self.cfg.batch_size} must be set to 1!"

        train_loader = torch.utils.data.DataLoader(dataset=data_traj_train, 
                                     batch_size=self.cfg.batch_size,
                                     shuffle=True)

        test_loader = torch.utils.data.DataLoader(dataset=data_traj_valid,
                                            batch_size=self.cfg.test_batch_size,
                                            shuffle=False)

        return {
            'train': {
                'loader': train_loader,
                'example': data_traj_train[0],
            },
            'test': {
                'loader': test_loader,
                'example': data_traj_valid[0],
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
        return timers

    def update_timers(self):
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
            if RANK == 0:
                log.info(f"t_{key} [min,max,avg] = [{self.timers_min[key][i]},{self.timers_max[key][i]},{self.timers_avg[key][i]}]") 
        return

    def train_step(self, data: DataBatch) -> Tensor:
        loss = torch.tensor([0.0])
        graph = self.data['graph']
        self.timers['dataTransfer'][self.timer_step] = time.time()
        if WITH_CUDA:
            data['x'] = data['x'].cuda() 
            data['y'] = data['y'].cuda()
            graph.edge_index = graph.edge_index.cuda()
            graph.edge_weight = graph.edge_weight.cuda()
            graph.edge_attr = graph.edge_attr.cuda()
            graph.batch = graph.batch.cuda() if graph.batch is not None else None
            graph.halo_info = graph.halo_info.cuda()
            graph.node_degree = graph.node_degree.cuda()
            loss = loss.cuda()
        self.timers['dataTransfer'][self.timer_step] = time.time() - self.timers['dataTransfer'][self.timer_step]

        self.optimizer.zero_grad()

        # re-allocate send buffer 
        self.timers['bufferInit'][self.timer_step] = time.time()
        if self.cfg.halo_swap_mode != 'none':
            for i in range(SIZE):
                self.buffer_send[i] = torch.zeros_like(self.buffer_send[i])
            for i in range(SIZE):
                self.buffer_recv[i] = torch.zeros_like(self.buffer_recv[i])
        else:
            buffer_send = None
            buffer_recv = None
        self.timers['bufferInit'][self.timer_step] = time.time() - self.timers['bufferInit'][self.timer_step]
        
        # Prediction
        self.timers['forwardPass'][self.timer_step] = time.time()
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
        self.timers['forwardPass'][self.timer_step] = time.time() - self.timers['forwardPass'][self.timer_step]

        # Accumulate loss
        self.timers['loss'][self.timer_step] = time.time()
        target = data['y'][0]
        n_nodes_local = graph.n_nodes_local
        if SIZE == 1:
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
        self.timers['loss'][self.timer_step] = time.time() - self.timers['loss'][self.timer_step]

        self.timers['backwardPass'][self.timer_step] = time.time()
        loss.backward()
        self.timers['backwardPass'][self.timer_step] = time.time() - self.timers['backwardPass'][self.timer_step]

        if SIZE == 1:
            model = self.model 
        else:
            model = self.model.module 
        #for p in model.parameters():
        #    p.grad = 0.1 * SIZE * torch.ones_like(p.grad)  # or whatever other operation

        self.timers['optimizerStep'][self.timer_step] = time.time()
        self.optimizer.step()
        self.timers['optimizerStep'][self.timer_step] = time.time() - self.timers['optimizerStep'][self.timer_step]

        # Update timers 
        if self.timer_step < self.timer_step_max - 1:
            self.update_timers()
            self.timer_step += 1
        else: # write timers 
            savepath = self.cfg.profile_dir
            if RANK == 0:
                if not os.path.exists(savepath):
                    os.makedirs(savepath)
            COMM.Barrier()
            torch.save(self.timers, savepath + '/timers_%s.tar' %(model.get_save_header()))
            torch.save(self.timers_max, savepath + '/timers_max_%s.tar' %(model.get_save_header()))
            torch.save(self.timers_min, savepath + '/timers_min_%s.tar' %(model.get_save_header()))
            torch.save(self.timers_avg, savepath + '/timers_avg_%s.tar' %(model.get_save_header()))

        return loss 

    def train_step_verification(self, data: DataBatch) -> Tensor:
        loss = torch.tensor([0.0])
        
        if WITH_CUDA:
            data.x = data.x.cuda() 
            data.y = data.y.cuda()
            data.edge_index = data.edge_index.cuda()
            data.edge_weight = data.edge_weight.cuda()
            data.edge_attr = data.edge_attr.cuda()
            data.batch = data.batch.cuda() if data.batch is not None else None
            data.halo_info = data.halo_info.cuda()
            data.node_degree = data.node_degree.cuda()
            data.pos = data.pos.cuda()
            loss = loss.cuda()
                    
        self.optimizer.zero_grad()
        
        out_gnn = self.model(x = data.x,
                             edge_index = data.edge_index,
                             edge_attr = data.edge_attr,
                             edge_weight = data.edge_weight,
                             halo_info = data.halo_info,
                             mask_send = self.mask_send,
                             mask_recv = self.mask_recv,
                             buffer_send = self.buffer_send,
                             buffer_recv = self.buffer_recv,
                             neighboring_procs = self.neighboring_procs,
                             SIZE = SIZE,
                             batch = data.batch)

        # Accumulate loss
        target = data.x

        # Toy loss: evaluate at all of the nodes 
        n_nodes_local = data.n_nodes_local
        if WITH_CUDA:
            n_nodes_local = n_nodes_local.cuda()

        if SIZE == 1:
            loss = self.loss_fn(out_gnn[:n_nodes_local], target[:n_nodes_local])
            effective_nodes = n_nodes_local 
        else: # custom 
            n_output_features = out_gnn.shape[1]
            squared_errors_local = torch.pow(out_gnn[:n_nodes_local] - target[:n_nodes_local], 2)
            squared_errors_local = squared_errors_local/data.node_degree[:n_nodes_local].unsqueeze(-1)

            sum_squared_errors_local = squared_errors_local.sum()
            effective_nodes_local = torch.sum(1.0/data.node_degree[:n_nodes_local])

            effective_nodes = distnn.all_reduce(effective_nodes_local)
            sum_squared_errors = distnn.all_reduce(sum_squared_errors_local)
            loss = (1.0/(effective_nodes*n_output_features)) * sum_squared_errors

        
        #loss.backward()
        #self.optimizer.step()

        # Scaled sum of input node features 
        x_scaled = data.x[:n_nodes_local, :]/data.node_degree[:n_nodes_local].unsqueeze(-1)
        sum_x_scaled = x_scaled.sum(axis=0)
        total_sum_x_scaled = distnn.all_reduce(sum_x_scaled)

        # Scaled sum of output node features 
        y_scaled = out_gnn[:n_nodes_local, :].detach()/data.node_degree[:n_nodes_local].unsqueeze(-1) 
        sum_y_scaled = y_scaled.sum(axis=0)
        total_sum_y_scaled = distnn.all_reduce(sum_y_scaled)

        # Scaled sum of positions 
        pos_scaled = data.pos[:n_nodes_local, :]/data.node_degree[:n_nodes_local].unsqueeze(-1)
        sum_pos_scaled = pos_scaled.sum(axis=0)
        total_sum_pos_scaled = distnn.all_reduce(sum_pos_scaled)

        # Sum of n_nodes_local 
        n_nodes = distnn.all_reduce(n_nodes_local)

        # Edge weights 
        n_edges_local = torch.tensor(data.edge_index.shape[1])
        if WITH_CUDA:
            n_edges_local = n_edges_local.cuda()
        n_edges = distnn.all_reduce(n_edges_local)
        effective_edges_local = torch.tensor(data.edge_weight.sum())
        effective_edges = distnn.all_reduce(effective_edges_local)

        log.info('[RANK %d] Loss: %g' %(RANK, loss.item()))
        #log.info('[RANK %d] QoI in scaled: %g' %(RANK, total_sum_x_scaled.item()))
        log.info(f'[RANK {RANK}] : Input dtype : {data.x.dtype}')
        log.info(f'[RANK {RANK}] : Output dtype : {out_gnn.dtype}')
        log.info(f'[RANK {RANK}] : QoI in scaled : {total_sum_x_scaled}')
        log.info(f'[RANK {RANK}] : QoI out scaled : {total_sum_y_scaled}')
        log.info('[RANK %d] n_nodes total: %g \t effective_nodes: %g' %(RANK, n_nodes.item(), effective_nodes.item()))
        log.info('[RANK %d] ei_edges_local: %g \t ei_edges total: %g' %(RANK, n_edges_local.item(), n_edges.item()))
        log.info('[RANK %d] effective_edges_local: %g \t effective_edges total: %g' %(RANK, effective_edges_local.item(), effective_edges.item()))
        #log.info('[RANK %d] Effective nodes: %g' %(RANK, effective_nodes.item()))
        #log.info('[RANK %d] QoI out: %g' %(RANK, qoi_out.item()))

        # Print the backward gradient   
        if SIZE == 1:
            model = self.model 
        else:
            model = self.model.module 

        # loop through model parameters 
        grad_dict = {name: param.grad for name, param in model.named_parameters()}
        grad_dict["loss"] = loss.item()
        grad_dict["total_sum_x_scaled"] = total_sum_x_scaled
        grad_dict["total_sum_y_scaled"] = total_sum_y_scaled
        grad_dict["total_sum_pos_scaled"] = total_sum_pos_scaled
        grad_dict["effective_nodes"] = effective_nodes
        grad_dict["effective_edges"] = effective_edges

        if (TORCH_FLOAT_DTYPE == torch.float64):
            path_desc = 'float64'
        else:
            path_desc = 'float32'
        
        savepath = self.cfg.work_dir + '/outputs/postproc/real_gnn_test_4/periodic_after_fix_edges_2/gradient_data_gpu_nondeterministic_POLARIS/tgv_poly_1/%s' %(path_desc)
        savepath = self.cfg.work_dir + '/outputs/postproc/gnn_verification_for_paper/tgv_poly_1/%s' %(path_desc)

        # if path doesnt exist, make it 
        if RANK == 0:
            if not os.path.exists(savepath):
                os.makedirs(savepath)
                print("Directory created by root processor.")
            else:
                print("Directory already exists.")

        # Synchronize all processors
        COMM.Barrier()
        
        torch.save(grad_dict, savepath + '/%s.tar' %(model.get_save_header()))
       
        force_abort()
        return loss 

    def train_step_profile(self):
        self.model.train()
        wait = 5
        warmup = 50
        active = 200

        # with torch.no_grad(): 
        with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(
                    wait=wait,
                    warmup=warmup,
                    active=active)
                #on_trace_ready=trace_handler
            ) as prof:

            data = self.data['train']['example']
            for idx in range(wait+warmup+active): 
                # log.info(f"\t[RANK {RANK}] -- step {idx}")

                loss = torch.tensor([0.0])
                if WITH_CUDA:
                    data.x = data.x.cuda() 
                    data.y = data.y.cuda()
                    data.edge_index = data.edge_index.cuda()
                    data.edge_weight = data.edge_weight.cuda()
                    data.edge_attr = data.edge_attr.cuda()
                    data.batch = data.batch.cuda() if data.batch is not None else None
                    data.halo_info = data.halo_info.cuda()
                    data.node_degree = data.node_degree.cuda()
                    loss = loss.cuda()

                self.optimizer.zero_grad()

                # re-allocate send buffer 
                if self.cfg.halo_swap_mode != 'none':
                    for i in range(SIZE):
                        self.buffer_send[i] = torch.zeros_like(self.buffer_send[i])
                    for i in range(SIZE):
                        self.buffer_recv[i] = torch.zeros_like(self.buffer_recv[i])

                else:
                    buffer_send = None
                    buffer_recv = None
                
                with record_function(f"[RANK {RANK}] FORWARD PASS"):
                    out_gnn = self.model(x = data.x,
                                         edge_index = data.edge_index,
                                         edge_attr = data.edge_attr,
                                         edge_weight = data.edge_weight,
                                         halo_info = data.halo_info,
                                         mask_send = self.mask_send,
                                         mask_recv = self.mask_recv,
                                         buffer_send = self.buffer_send,
                                         buffer_recv = self.buffer_recv,
                                         neighboring_procs = self.neighboring_procs,
                                         SIZE = SIZE,
                                         batch = data.batch)

                # target = data.x

                # # Accumulate loss
                # with record_function(f"[RANK {RANK}] LOSS"):
                #     n_nodes_local = data.n_nodes_local
                #     if SIZE == 1:
                #         loss = self.loss_fn(out_gnn[:n_nodes_local], target[:n_nodes_local])
                #         effective_nodes = n_nodes_local 
                #     else:
                #         n_output_features = out_gnn.shape[1]
                #         squared_errors_local = torch.pow(out_gnn[:n_nodes_local] - target[:n_nodes_local], 2)
                #         squared_errors_local = squared_errors_local/data.node_degree[:n_nodes_local].unsqueeze(-1)

                #         sum_squared_errors_local = squared_errors_local.sum()
                #         effective_nodes_local = torch.sum(1.0/data.node_degree[:n_nodes_local])

                #         effective_nodes = distnn.all_reduce(effective_nodes_local)
                #         sum_squared_errors = distnn.all_reduce(sum_squared_errors_local)
                #         loss = (1.0/(effective_nodes*n_output_features)) * sum_squared_errors
                # 
                # with record_function(f"[RANK {RANK}] BACKWARD PASS"):
                #     loss.backward()

                # self.optimizer.step()

                # Step the profiler 
                prof.step()

        return prof

    def train_epoch(self, epoch: int) -> dict:
        self.model.train()
        start = time.time()
        running_loss = torch.tensor(0.)
        count = torch.tensor(0.)

        if WITH_CUDA:
            running_loss = running_loss.cuda()
            count = count.cuda()

        #train_sampler = self.data['train']['sampler']
        #train_sampler.set_epoch(epoch)

        train_loader = self.data['train']['loader']

        for bidx, data in enumerate(train_loader):
            batch_size = len(data)
            #loss = self.train_step_verification(data)
            loss = self.train_step(data)
            self.loss_hist_train_iter[self.training_iter] = loss.item()

            running_loss += loss.item()
            count += 1 # accumulate current batch count
            self.training_iter += 1 # accumulate total training iteration

            # Log on Rank 0:
            if bidx % self.cfg.logfreq == 0 and RANK == 0:
                metrics = {
                    'epoch': epoch,
                    'dt': time.time() - start,
                    'batch_loss': loss.item(),
                    'running_loss': running_loss,
                }
                pre = [
                    f'[{RANK}]',
                    (   # looks like: [num_processed/total (% complete)]
                        f'[{epoch}/{self.cfg.epochs}:'
                        f' Batch {bidx+1}'
                        f' ({100. * (bidx+1) / len(train_loader):.0f}%)]'
                    ),
                ]
                log.info(' '.join([
                    *pre, *[f'{k}={v:.4f}' for k, v in metrics.items()]
                ]))

        # divide running loss by number of batches
        running_loss = running_loss / count
        loss_avg = metric_average(running_loss)

        return {'loss': loss_avg}

    def test(self) -> dict:
        running_loss = torch.tensor(0.)
        count = torch.tensor(0.)
        if WITH_CUDA:
            running_loss = running_loss.cuda()
            count = count.cuda()
        self.model.eval()
        test_loader = self.data['test']['loader']

        with torch.no_grad():
            for data in test_loader:
                loss = torch.tensor([0.0])
                
                running_loss += loss.item()
                count += 1

            running_loss = running_loss / count
            loss_avg = metric_average(running_loss)

        return {'loss': loss_avg}

    def writeGraphStatistics(self):
        if RANK == 0: log.info(f"In writeGraphStatistics")
        # Write the number of nodes, halo nodes, and edges in each rank of the sub-graph 
        
        if SIZE == 1:
            model = self.model
        else:
            model = self.model.module

        log.info(f"[RANK {RANK}] -- model save header : {model.get_save_header()}")

        # if path doesnt exist, make it 
        savepath = self.cfg.work_dir + "/outputs/GraphStatistics/weak_scaling" 
        if RANK == 0:
            if not os.path.exists(savepath):
                os.makedirs(savepath)
                print("Directory created by root processor.")
            else:
                print("Directory already exists.")
        COMM.Barrier()

        # Number of local nodes 
        n_nodes_local = self.data_reduced.n_nodes_local
        n_nodes_halo = self.data_reduced.n_nodes_halo
        n_edges = self.data_reduced.edge_index.shape[1]

        log.info(f"[RANK {RANK}] -- number of local nodes: {n_nodes_local}, number of halo nodes: {n_nodes_halo}, number of edges: {n_edges}")

        a = {} 
        a['n_nodes_local'] = n_nodes_local
        a['n_nodes_halo'] = n_nodes_halo
        a['n_edges'] = n_edges
        torch.save(a, savepath + '/%s.tar' %(model.get_save_header())) 
        
        return 



def train(cfg: DictConfig) -> None:
    start = time.time()
    trainer = Trainer(cfg)
    trainer.writeGraphStatistics()
    epoch_times = []

    for epoch in range(trainer.epoch_start, cfg.epochs+1):
        # ~~~~ Training step 
        t0 = time.time()
        trainer.epoch = epoch
        train_metrics = trainer.train_epoch(epoch)
        trainer.loss_hist_train[epoch-1] = train_metrics["loss"]

        epoch_time = time.time() - t0
        epoch_times.append(epoch_time)

        # ~~~~ Validation step
        test_metrics = trainer.test()
        trainer.loss_hist_test[epoch-1] = test_metrics["loss"]
        
        # ~~~~ Printing
        if RANK == 0:
            astr = f'[TEST] loss={test_metrics["loss"]:.4e}'
            sepstr = '-' * len(astr)
            log.info(sepstr)
            log.info(astr)
            log.info(sepstr)
            summary = '  '.join([
                '[TRAIN]',
                f'loss={train_metrics["loss"]:.4e}',
                f'epoch_time={epoch_time:.4g} sec'
            ])
            log.info((sep := '-' * len(summary)))
            log.info(summary)
            log.info(sep)

        # ~~~~ Step scheduler based on validation loss
        # trainer.scheduler.step(test_metrics["loss"]) # SB: toggle scheduler

        # ~~~~ Checkpointing step 
        if epoch % cfg.ckptfreq == 0 and RANK == 0:
            astr = 'Checkpointing on root processor, epoch = %d' %(epoch)
            sepstr = '-' * len(astr)
            log.info(sepstr)
            log.info(astr)
            log.info(sepstr)

            if not os.path.exists(cfg.ckpt_dir):
                os.makedirs(cfg.ckpt_dir)

            if WITH_DDP and SIZE > 1:
                sd = trainer.model.module.state_dict()
            else:
                sd = trainer.model.state_dict()

            ckpt = {'epoch' : epoch,
                    'training_iter' : trainer.training_iter,
                    'model_state_dict' : sd,
                    'optimizer_state_dict' : trainer.optimizer.state_dict(),
                    'scheduler_state_dict' : trainer.scheduler.state_dict(),
                    'loss_hist_train' : trainer.loss_hist_train,
                    'loss_hist_test' : trainer.loss_hist_test}
            
            torch.save(ckpt, trainer.ckpt_path)
        dist.barrier()

    rstr = f'[{RANK}] ::'
    log.info(' '.join([
        rstr,
        f'Total training time: {time.time() - start} seconds'
    ]))
    
    if RANK == 0:
        if WITH_CUDA:
            trainer.model.to('cpu')
        if not os.path.exists(cfg.model_dir):
            os.makedirs(cfg.model_dir)

        if WITH_DDP and SIZE > 1:
            sd = trainer.model.module.state_dict()
            ind = trainer.model.module.input_dict()
        else:
            sd = trainer.model.state_dict()
            ind = trainer.model.input_dict()

        save_dict = {
                    'state_dict' : sd,
                    'input_dict' : ind,
                    'loss_hist_train' : trainer.loss_hist_train,
                    'loss_hist_test' : trainer.loss_hist_test,
                    'training_iter' : trainer.training_iter,
                    'loss_hist_train_iter' : trainer.loss_hist_train_iter
                    }
        
        torch.save(save_dict, trainer.model_path)

    # Plot connectivity
    if (cfg.plot_connectivity):
        gplot.plot_graph(trainer.data['train']['example'], RANK, cfg.work_dir)

    return 


def train_profile(cfg: DictConfig) -> None:
    start = time.time()
    trainer = Trainer(cfg)
    # epoch_times = []

    # Run a bunch of train steps 
    t_prof = time.time()
    prof = trainer.train_step_profile()
    t_prof = time.time() - t_prof
    log.info(f"[RANK {RANK}] -- t_prof = {t_prof} s")

    if RANK == 0:
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

    # save profiler data 
    if SIZE == 1:
        model = trainer.model
    else:
        model = trainer.model.module

    # if path doesnt exist, make it 
    savepath = cfg.profile_dir
    if RANK == 0:
        if not os.path.exists(savepath):
            os.makedirs(savepath)
            print("Directory created by root processor.")
        else:
            print("Directory already exists.")
    COMM.Barrier()

    torch.save(prof.key_averages(), savepath + '/%s.tar' %(model.get_save_header()))

    return 

@hydra.main(version_base=None, config_path='./conf', config_name='config')
def main(cfg: DictConfig) -> None:
    print('Rank %d, local rank %d, which has device %s. Sees %d devices.' %(RANK,int(LOCAL_RANK),DEVICE,torch.cuda.device_count()))
    if RANK == 0:
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('INPUTS:')
        print(OmegaConf.to_yaml(cfg)) 
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

    if cfg.profile: 
        train_profile(cfg)
    else: 
        train(cfg)

    #halo_test(cfg)
    
    cleanup()

if __name__ == '__main__':
    main()
