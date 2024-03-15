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
import torch.utils.data
import torch.utils.data.distributed
import torch.multiprocessing as mp

import torch.distributed as dist
import torch.distributed.nn as distnn
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

# Models
import models.gnn as gnn

# Graph connectivity
import graph_connectivity as gcon

# Clean printing
from prettytable import PrettyTable 

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
    WITH_CUDA = False

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

class Trainer:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.rank = RANK
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

        # ~~~~ Wrap model in DDP
        if WITH_DDP and SIZE > 1:
            self.model = DDP(self.model)
        
        # ~~~~ Set loss function
        self.loss_fn = nn.MSELoss()

        # ~~~~ Set optimizer
        self.optimizer = self.build_optimizer(self.model)
        

    def build_model(self) -> nn.Module:
        sample = self.data['train']['example']

        # Get the polynomial order -- for naming the model
        try:
            main_path = self.cfg.gnn_outputs_path
            Np = np.loadtxt(main_path + "Np_rank_%d_size_%d" %(RANK, SIZE), dtype=np.float32)
            poly = np.cbrt(Np) - 1.
            poly = int(poly)
        except FileNotFoundError:
            poly = 0

        input_node_channels = sample.x.shape[1]
        input_edge_channels = sample.edge_attr.shape[1]
        hidden_channels = self.cfg.hidden_channels
        output_node_channels = sample.y.shape[1]
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

    def train_step(self):
        data = self.data['train']['example']
        loss = torch.tensor([0.0])
        if WITH_CUDA:
            data.x = data.x.cuda()
            data.y = data.y.cuda()
            data.edge_index = data.edge_index.cuda()
            data.edge_weight = data.edge_weight.cuda()
            data.edge_attr = data.edge_attr.cuda()
            data.batch = data.batch.cuda() if data.batch else None
            data.halo_info = data.halo_info.cuda()
            data.node_degree = data.node_degree.cuda()
            loss = loss.cuda()
        
        self.optimizer.zero_grad()

        # re-allocate buffers
        if self.cfg.halo_swap_mode != 'none':
            for i in range(SIZE):
                self.buffer_send[i] = torch.zeros_like(self.buffer_send[i])
            for i in range(SIZE):
                self.buffer_recv[i] = torch.zeros_like(self.buffer_recv[i])
        else:
            buffer_send = None
            buffer_recv = None

        # Prediction
        if RANK == 0: log.info("GNN prediction...")
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


        # Loss
        if RANK == 0: log.info("GNN loss...")
        target = data.x
        n_nodes_local = data.n_nodes_local
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

        if RANK == 0: log.info("loss.backward()...")
        loss.backward()

        if RANK == 0: log.info("optimizer.step()...")
        self.optimizer.step()
        return 

    def setup_torch(self):
        torch.manual_seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)

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
            halo_info = self.data['train']['example'].halo_info

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
        buff_send = [torch.tensor([])] * SIZE
        buff_recv = [torch.tensor([])] * SIZE
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

        return buff_send, buff_recv, n_max 

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
        if RANK == 0: log.info("In setup_halo...")
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

        self.data_reduced.n_nodes_local = torch.tensor(n_nodes_local, dtype=torch.int64)
        self.data_reduced.n_nodes_halo = torch.tensor(n_nodes_halo, dtype=torch.int64)
        self.data_reduced.halo_info = halo_info

        return 

    def setup_data(self):
        """
        Generate the PyTorch Geometric Dataset 
        """

        if RANK == 0:
            log.info('In setup_data...')

        # Load data
        main_path = self.cfg.gnn_outputs_path
        path_to_x = main_path + 'fld_u_time_10.0_rank_%d_size_%d' %(RANK,SIZE)
        path_to_y = main_path + 'fld_u_time_10.0_rank_%d_size_%d' %(RANK,SIZE)
        data_x = np.fromfile(path_to_x + ".bin", dtype=np.float64).reshape((-1,3))
        data_x = data_x.astype(NP_FLOAT_DTYPE)
        data_y = np.fromfile(path_to_y + ".bin", dtype=np.float64).reshape((-1,3))
        data_y = data_y.astype(NP_FLOAT_DTYPE)

        # Retain only N_gll = Np*Ne elements
        N_gll = self.data_full.pos.shape[0]
        data_x = data_x[:N_gll, :]
        data_y = data_y[:N_gll, :]

        # Get data in reduced format (non-overlapping)
        data_x_reduced = data_x[self.idx_full2reduced, :]
        data_y_reduced = data_y[self.idx_full2reduced, :]
        pos_reduced = self.data_reduced.pos

        # Read in edge weights
        path_to_ew = main_path + 'edge_weights_rank_%d_size_%d.npy' %(RANK,SIZE)
        edge_freq = torch.tensor(np.load(path_to_ew), dtype=TORCH_FLOAT_DTYPE)
        self.data_reduced.edge_weight = 1.0/edge_freq

        # Read in node degree
        path_to_node_degree = main_path + 'node_degree_rank_%d_size_%d.npy' %(RANK,SIZE)
        node_degree = torch.tensor(np.load(path_to_node_degree), dtype=TORCH_FLOAT_DTYPE)
        self.data_reduced.node_degree = node_degree

        # Add halo nodes by appending the end of the node arrays
        n_nodes_halo = self.data_reduced.n_nodes_halo
        n_features_x = data_x_reduced.shape[1]
        data_x_halo = torch.zeros((n_nodes_halo, n_features_x), dtype=TORCH_FLOAT_DTYPE)

        n_features_y = data_y_reduced.shape[1]
        data_y_halo = torch.zeros((n_nodes_halo, n_features_y), dtype=TORCH_FLOAT_DTYPE)

        n_features_pos = pos_reduced.shape[1]
        pos_halo = torch.zeros((n_nodes_halo, n_features_pos), dtype=TORCH_FLOAT_DTYPE)

        node_degree_halo = torch.zeros((n_nodes_halo), dtype=TORCH_FLOAT_DTYPE)

        # Add self-edges for halo nodes (unused)
        n_nodes_local = self.data_reduced.n_nodes_local
        edge_index_halo = torch.arange(n_nodes_local, n_nodes_local + n_nodes_halo, dtype=torch.int64)
        edge_index_halo = torch.stack((edge_index_halo,edge_index_halo))

        # Add filler edge weights for these self-edges
        edge_weight_halo = torch.zeros(n_nodes_halo)

        # Populate data object 
        n_features_in = data_x_reduced.shape[1]
        n_features_out = data_y_reduced.shape[1]
        n_nodes = self.data_reduced.pos.shape[0]
        device_for_loading = 'cpu'

        # Get dictionary 
        reduced_graph_dict = self.data_reduced.to_dict()

        # Create training dataset -- only 1 snapshot for demo
        data_train_list = []
        data_temp = Data(
                            x = torch.tensor(data_x_reduced),
                            y = torch.tensor(data_y_reduced)
                        )
        for key in reduced_graph_dict.keys():
            data_temp[key] = reduced_graph_dict[key]
        data_temp.x = torch.cat((data_temp.x, data_x_halo), dim=0)
        data_temp.y = torch.cat((data_temp.y, data_y_halo), dim=0)
        data_temp.pos = torch.cat((data_temp.pos, pos_halo), dim=0)
        data_temp.node_degree = torch.cat((data_temp.node_degree, node_degree_halo), dim=0)
        data_temp.edge_index = torch.cat((data_temp.edge_index, edge_index_halo), dim=1)
        data_temp.edge_weight = torch.cat((data_temp.edge_weight, edge_weight_halo), dim=0)
        data_temp.edge_weight_temp = data_temp.edge_weight

        # Populate edge_attrs
        cart = torch_geometric.transforms.Cartesian(norm=False, max_value = None, cat = False)
        dist = torch_geometric.transforms.Distance(norm = False, max_value = None, cat = True)
        cart(data_temp) # adds cartesian/component-wise distance
        dist(data_temp) # adds euclidean distance

        data_temp = data_temp.to(device_for_loading)
        data_train_list.append(data_temp)
        n_train = len(data_train_list) # should be 1

        train_dataset = data_train_list

        if (RANK == 0):
            print(data_train_list[0])

        return {
            'train': {
                'example': train_dataset[0],
            }
        }

def gnn_test(cfg: DictConfig) -> None:
    t_start = time.time()
    trainer = Trainer(cfg)
    trainer.train_step()
    t_end = time.time()
    return

def halo_test(cfg: DictConfig) -> None:
    t_start = time.time()
    trainer = Trainer(cfg)
    log.info(f"[RANK {RANK}] -- num neighboring procs: {len(trainer.neighboring_procs)}")
    t_end = time.time()

    mode = trainer.cfg.halo_swap_mode 
    data = trainer.data['train']['example']
    n_nodes_local = data.n_nodes_local
    n_nodes_halo = data.n_nodes_halo
    input_tensor = data.x

    if WITH_CUDA:
        input_tensor = input_tensor.cuda()

    # get the buffers 
    mask_send, mask_recv = trainer.mask_send, trainer.mask_recv 
    buff_send, buff_recv, n_buffer_rows = trainer.build_buffers(input_tensor.shape[1])

    # re-initialize the buffers (this goes before forward pass, doing it here for completeness)
    for i in range(SIZE):
        buff_send[i] = torch.zeros_like(buff_send[i])
    for i in range(SIZE):
        buff_recv[i] = torch.zeros_like(buff_recv[i])

    # step 1: populate the send buffers 
    for i in trainer.neighboring_procs:
        n_send = len(mask_send[i])
        buff_send[i][:n_send,:] = input_tensor[mask_send[i]]
        if RANK == 0: log.info(f"buff_send shape for nei {i}: {buff_send[i].shape}")

    # step 2: swap  
    if mode == "all_to_all": 
        distnn.all_to_all(buff_recv, buff_send)
        dist.barrier()
    elif mode == "send_recv": # asynchronous 
        send_req = []
        for dst in trainer.neighboring_procs:
            tmp = dist.isend(buff_send[dst], dst)
            send_req.append(tmp)
        recv_req = []
        for src in trainer.neighboring_procs:
            tmp = dist.irecv(buff_recv[src], src)
            recv_req.append(tmp)

        for req in send_req:
            req.wait()
        for req in recv_req:
            req.wait()
        dist.barrier()
    elif mode == "sendrecv_sync":

        # # SB: this hangs 
        # log.info(f"[RANK {RANK}] send")
        # for dst in trainer.neighboring_procs:
        #     log.info(f"[RANK {RANK}] \t dst={dst}")
        #     dist.send(buff_send[dst], dst)
        # log.info(f"[RANK {RANK}] recv")
        # for src in trainer.neighboring_procs:
        #     log.info(f"[RANK {RANK}] \t src={src}")
        #     dist.recv(buff_recv[src], src)

        # SB: doing it manually like this does not hang 
        # 0 to 1 
        if RANK == 0:
            # send to rank 1 
            dst = 1
            dist.send(buff_send[dst], dst) 
        if RANK == 1:
            # recv from rank 0 
            src = 0 
            dist.recv(buff_recv[src], src)
        
        # 1 to 0 
        if RANK == 1:
            # send to rank 0 
            dst = 0
            dist.send(buff_send[dst], dst)
        if RANK == 0:
            # recv from rank 1 
            src = 1
            dist.recv(buff_recv[src], src)

    else:
        pass

    # step 3: copy receive buffers back in to data  
    if RANK == 0: log.info(f"halo nodes before: {input_tensor[n_nodes_local:]}")
    for i in trainer.neighboring_procs:
        n_recv = len(mask_recv[i])
        input_tensor[mask_recv[i]] = buff_recv[i][:n_recv,:]
    if RANK == 0: log.info(f"halo nodes after: {input_tensor[n_nodes_local:]}")
    return


@hydra.main(version_base=None, config_path='./conf', config_name='config')
def main(cfg: DictConfig) -> None:
    if RANK == 0:
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('INPUTS:')
        print(OmegaConf.to_yaml(cfg)) 
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

    #gnn_test(cfg)
    halo_test(cfg)
    
    cleanup()

if __name__ == '__main__':
    main()
