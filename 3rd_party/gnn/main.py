"""
PyTorch DDP integrated with PyGeom for multi-node training
"""
from __future__ import absolute_import, division, print_function, annotations
import os
import socket
import logging

from typing import Optional, Union, Callable

import numpy as np

import hydra
import time
import torch
import torch.utils.data
import torch.utils.data.distributed
from torch.cuda.amp.grad_scaler import GradScaler
import torch.multiprocessing as mp
import torch.distributions as tdist 

import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from omegaconf import DictConfig
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

def metric_average(val: Tensor):
    if (WITH_DDP):
        dist.all_reduce(val, op=dist.ReduceOp.SUM)
        return val / SIZE
    return val

class Trainer:
    def __init__(self, cfg: DictConfig, scaler: Optional[GradScaler] = None):
        self.cfg = cfg
        self.rank = RANK
        if scaler is None:
            self.scaler = None
        self.device = 'gpu' if torch.cuda.is_available() else 'cpu'
        self.backend = self.cfg.backend
        if WITH_DDP:
            init_process_group(RANK, SIZE, backend=self.backend)
        
        # ~~~~ Init torch stuff 
        self.setup_torch()


        # ~~~~ Setup local graph 
        self.data_reduced, self.data_full, self.idx_full2reduced, self.idx_reduced2full = self.setup_local_graph()

        # ~~~~ Setup halo nodes 
        self.neighboring_procs = {}
        self.setup_halo()

        # ~~~~ Setup data 
        self.data = self.setup_data()

        # ~~~~ # self.n_nodes_internal_procs = list(torch.empty([1], dtype=torch.int64, device=DEVICE_ID)) * SIZE
        # ~~~~ # if WITH_CUDA:
        # ~~~~ #     self.data.n_nodes_internal = self.data.n_nodes_internal.cuda()
        # ~~~~ # dist.all_gather(self.n_nodes_internal_procs, self.data.n_nodes_internal)
        # ~~~~ # print('[RANK %d] -- data: ' %(RANK), self.data)

        # ~~~~ # # ~~~~ Setup halo exchange masks
        # ~~~~ # self.mask_send, self.mask_recv = self.build_masks()

        # ~~~~ # # ~~~~ Initialize send/recv buffers on device (if applicable)
        # ~~~~ # n_features_pos = self.data.pos.shape[1]
        # ~~~~ # self.pos_buffer_send, self.pos_buffer_recv = self.build_buffers(n_features_pos)

        # ~~~~ # # ~~~~ Do a halo swap on position matrices 
        # ~~~~ # #if RANK == 1: 
        # ~~~~ # #    print('[RANK %d] -- pos before: ' %(RANK), self.data.pos)
        # ~~~~ # if WITH_CUDA:
        # ~~~~ #     self.data.pos = self.data.pos.cuda()
        # ~~~~ # self.data.pos = self.halo_swap(self.data.pos, self.pos_buffer_send, self.pos_buffer_recv)
        # ~~~~ # #if RANK == 0: 
        # ~~~~ # #    print('[RANK %d] -- pos after: ' %(RANK), self.data.pos)

        # ~~~~ Build model and move to gpu 
        self.model = self.build_model()
        if self.device == 'gpu':
            self.model.cuda()

        # ~~~~ Init training and testing loss history 
        self.loss_hist_train = np.zeros(self.cfg.epochs)
        self.loss_hist_test = np.zeros(self.cfg.epochs)

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

    def build_model(self) -> nn.Module:
        sample = self.data['train']['example']
        input_channels = sample.x.shape[1] 
        output_channels = sample.y.shape[1]
        hidden_channels = 16
        n_mlp_layers = [2, 3, 3] #[encoder/decoder layers, edge update layers, node update layers] 
        activation = F.elu

        model = gnn.mp_gnn(input_channels, 
                           hidden_channels, 
                           output_channels, 
                           n_mlp_layers, 
                           activation)

        return model

    def build_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        """
        DDP: scale learning rate by the number of GPUs
        """
        optimizer = optim.Adam(model.parameters(),
                               lr=SIZE * self.cfg.lr_init)
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
        mask_send = [None] * SIZE
        mask_recv = [None] * SIZE
        if SIZE > 1: 
            n_nodes_local = self.data.n_nodes_internal + self.data.n_nodes_halo
            halo_info = self.data.halo_info

            for i in self.neighboring_procs:
                idx_i = self.data.halo_info[:,2] == i
                # index of nodes to send to proc i 
                mask_send[i] = self.data.halo_info[:,0][idx_i] 
                #mask_send[i] = torch.unique(mask_send[i])
                
                # index of nodes to receive from proc i  
                mask_recv[i] = self.data.halo_info[:,1][idx_i]
                #mask_recv[i] = torch.unique(mask_recv[i])


            print('[RANK %d] mask_send: ' %(RANK), mask_send)
            print('[RANK %d] mask_recv: ' %(RANK), mask_recv)

        return mask_send, mask_recv 

    def build_buffers(self, n_features):
        buff_send = [None] * SIZE
        buff_recv = [None] * SIZE
        if SIZE > 1: 
            for i in self.neighboring_procs:
                buff_send[i] = torch.empty([len(self.mask_send[i]), n_features], dtype=torch.float32, device=DEVICE_ID) 
                buff_recv[i] = torch.empty([len(self.mask_recv[i]), n_features], dtype=torch.float32, device=DEVICE_ID)
        return buff_send, buff_recv 

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
        if self.cfg.verbose: print('[RANK %d]: Loading positions and global node index' %(RANK))
        pos = np.loadtxt(path_to_pos_full, dtype=np.float32)
        gli = np.loadtxt(path_to_glob_ids, dtype=np.int64).reshape((-1,1))

        # ~~~~ Get edge index
        if self.cfg.verbose: print('[RANK %d]: Loading edge index' %(RANK))
        ei = np.loadtxt(path_to_ei, dtype=np.int64).T
        
        # ~~~~ Get local unique mask
        if self.cfg.verbose: print('[RANK %d]: Loading local unique mask' %(RANK))
        local_unique_mask = np.loadtxt(path_to_unique_local, dtype=np.int64)

        # ~~~~ Get halo unique mask
        halo_unique_mask = np.array([])
        if SIZE > 1:
            halo_unique_mask = np.loadtxt(path_to_unique_halo, dtype=np.int64)

        # ~~~~ Make the full graph: 
        if self.cfg.verbose: print('[RANK %d]: Making the FULL GLL-based graph with overlapping nodes' %(RANK))
        data_full = Data(x = None, edge_index = torch.tensor(ei), pos = torch.tensor(pos), global_ids = torch.tensor(gli.squeeze()), local_unique_mask = torch.tensor(local_unique_mask), halo_unique_mask = torch.tensor(halo_unique_mask))
        data_full.edge_index = utils.remove_self_loops(data_full.edge_index)[0]
        data_full.edge_index = utils.coalesce(data_full.edge_index)
        data_full.edge_index = utils.to_undirected(data_full.edge_index)
        data_full.local_ids = torch.tensor(range(data_full.pos.shape[0]))

        # ~~~~ Get reduced (non-overlapping) graph and indices to go from full to reduced  
        if self.cfg.verbose: print('[RANK %d]: Making the REDUCED GLL-based graph with non-overlapping nodes' %(RANK))
        data_reduced, idx_full2reduced = gcon.get_reduced_graph(data_full)

        # ~~~~ Get the indices to go from reduced back to full graph  
        # idx_reduced2full = None
        if self.cfg.verbose: print('[RANK %d]: Getting idx_reduced2full' %(RANK))
        idx_reduced2full = gcon.get_upsample_indices(data_full, data_reduced, idx_full2reduced)

        return data_reduced, data_full, idx_full2reduced, idx_reduced2full


    def setup_halo(self):
        if self.cfg.verbose: print('[RANK %d]: Assembling halo_ids_list using reduced graph' %(RANK))
        main_path = self.cfg.gnn_outputs_path

        halo_info = None
        if SIZE > 1:
            halo_info = torch.tensor(np.load(main_path + '/halo_info_rank_%d_size_%d.npy' %(RANK,SIZE)))
            # Get list of neighboring processors for each processor
            self.neighboring_procs = np.unique(halo_info[:,3])
            n_nodes_local = self.data_reduced.pos.shape[0]
            n_nodes_halo = halo_info.shape[0]
        else:
            print('[RANK %d] neighboring procs: ' %(RANK), self.neighboring_procs)
            halo_info = torch.Tensor([])
            n_nodes_local = self.data_reduced.pos.shape[0]
            n_nodes_halo = 0

        if self.cfg.verbose: print('[RANK %d] neighboring procs: ' %(RANK), self.neighboring_procs)

        self.data_reduced.n_nodes_local = torch.tensor(n_nodes_local, dtype=torch.int64)
        self.data_reduced.n_nodes_halo = torch.tensor(n_nodes_halo, dtype=torch.int64)
        self.data_reduced.halo_info = halo_info

        return 

    def setup_data(self):
        """
        Generate the PyTorch Geometric Dataset 
        """
        # Load data 
        main_path = self.cfg.gnn_outputs_path
        path_to_x = main_path + 'x_rank_%d_size_%d' %(RANK,SIZE)
        path_to_y = main_path + 'y_rank_%d_size_%d' %(RANK,SIZE)
        data_x = np.loadtxt(path_to_x, ndmin=2, dtype=np.float32)
        data_y = np.loadtxt(path_to_y, ndmin=2, dtype=np.float32)

        # Get data in reduced format (non-overlapping)
        data_x_reduced = data_x[self.idx_full2reduced, :]
        data_y_reduced = data_y[self.idx_full2reduced, :]

        # Populate data object 
        n_features_in = data_x_reduced.shape[1]
        n_features_out = data_y_reduced.shape[1]
        n_nodes = self.data_reduced.pos.shape[0]
        device_for_loading = 'cpu'

        # Populate edge_attrs
        data_ref = self.data_reduced
        cart = torch_geometric.transforms.Cartesian(norm=False, max_value = None, cat = False)
        dist = torch_geometric.transforms.Distance(norm = False, max_value = None, cat = True)
        cart(data_ref) # adds cartesian/component-wise distance
        dist(data_ref) # adds euclidean distance

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

        data_temp = data_temp.to(device_for_loading)
        data_train_list.append(data_temp)
        n_train = len(data_train_list) # should be 1
       
        # Create validation dataset -- same data for demo  
        data_valid_list = []
        data_temp = Data(   
                            x = torch.tensor(data_x_reduced), 
                            y = torch.tensor(data_y_reduced)
                        )
        for key in reduced_graph_dict.keys():
            data_temp[key] = reduced_graph_dict[key]

        data_temp = data_temp.to(device_for_loading)
        data_valid_list.append(data_temp)
        n_valid = len(data_valid_list) # should be 1

        train_dataset = data_train_list
        test_dataset = data_valid_list 

        # No need for distributed sampler -- create standard dataset loader  
        train_loader = torch_geometric.loader.DataLoader(train_dataset, batch_size=self.cfg.batch_size, shuffle=False)
        test_loader = torch_geometric.loader.DataLoader(test_dataset, batch_size=self.cfg.test_batch_size, shuffle=False)  


        if (RANK == 0):
            print(data_train_list[0])

        return {
            'train': {
                'loader': train_loader,
                'example': train_dataset[0],
            },
            'test': {
                'loader': test_loader,
                'example': test_dataset[0],
            }
        }


    
    def setup_data_random(self):
        """
        Generate the PyTorch Geometric Dataset using dummy random data. 
        """
        kwargs = {}
        n_features_in = 3 
        n_features_out = 1 
        n_nodes = self.data_reduced.pos.shape[0]
        device_for_loading = 'cpu'

        reduced_graph_dict = self.data_reduced.to_dict()

        n_train = 16
        data_train_list = []
        for i in range(n_train): 
            data_temp = Data(   
                                x = torch.rand((n_nodes, n_features_in)), 
                                y = torch.rand((n_nodes, n_features_out))
                            )
            for key in reduced_graph_dict.keys():
                data_temp[key] = reduced_graph_dict[key]

            data_temp = data_temp.to(device_for_loading)
            data_train_list.append(data_temp)

        n_valid = 16
        data_valid_list = [] 
        for i in range(n_valid):
            data_temp = Data(   
                                x = torch.rand((n_nodes, n_features_in)), 
                                y = torch.rand((n_nodes, n_features_out))
                            )
            for key in reduced_graph_dict.keys():
                data_temp[key] = reduced_graph_dict[key]

            data_temp = data_temp.to(device_for_loading)
            data_valid_list.append(data_temp)

        train_dataset = data_train_list
        test_dataset = data_valid_list 

        # DDP: use DistributedSampler to partition training data
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=SIZE, rank=RANK,
        )
        train_loader = torch_geometric.loader.DataLoader(
            train_dataset,
            batch_size=self.cfg.batch_size,
            sampler=train_sampler,
            **kwargs
        )

        # DDP: use DistributedSampler to partition the test data
        test_sampler = torch.utils.data.distributed.DistributedSampler(
            test_dataset, num_replicas=SIZE, rank=RANK
        )
        test_loader = torch_geometric.loader.DataLoader(
            test_dataset, batch_size=self.cfg.test_batch_size
        )

        return {
            'train': {
                'sampler': train_sampler,
                'loader': train_loader,
                'example': train_dataset[0],
            },
            'test': {
                'sampler': test_sampler,
                'loader': test_loader,
                'example': test_dataset[0],
            }
        }

    def train_step(self, data: DataBatch) -> Tensor:
        loss = torch.tensor([0.0])
        
        if WITH_CUDA:
            data.x = data.x.cuda() 
            data.edge_index = data.edge_index.cuda()
            data.edge_attr = data.edge_attr.cuda()
            data.pos = data.pos.cuda()
            data.batch = data.batch.cuda()
            loss = loss.cuda()
        
        self.optimizer.zero_grad()

        # Prediction 
        out_gnn = self.model(data.x, data.edge_index, data.edge_attr, data.pos, data.batch)

        # Accumulate loss
        target = data.y
        if WITH_CUDA:
            target = target.cuda()

        loss = self.loss_fn(out_gnn, target)

        if self.scaler is not None and isinstance(self.scaler, GradScaler):
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()

        return loss 

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
            loss = self.train_step(data)
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
                
                if WITH_CUDA:
                    data.x = data.x.cuda()
                    data.edge_index = data.edge_index.cuda()
                    data.edge_attr = data.edge_attr.cuda()
                    data.pos = data.pos.cuda()
                    data.batch = data.batch.cuda()
                    loss = loss.cuda()
                
                # Prediction 
                out_gnn = self.model(data.x, data.edge_index, data.edge_attr, data.pos, data.batch)

                # Accumulate loss
                target = data.y
                if WITH_CUDA:
                    target = target.cuda()
                loss = self.loss_fn(out_gnn, target)

                running_loss += loss.item()
                count += 1

            running_loss = running_loss / count
            loss_avg = metric_average(running_loss)

        return {'loss': loss_avg}


def train(cfg: DictConfig) -> None:
    start = time.time()
    trainer = Trainer(cfg)
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
        trainer.scheduler.step(test_metrics["loss"])

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
                    'training_iter' : trainer.training_iter
                    }
        
        torch.save(save_dict, trainer.model_path)

    # Plot connectivity
    if (cfg.plot_connectivity):
        gplot.plot_graph(trainer.data['train']['example'], RANK, cfg.work_dir)


@hydra.main(version_base=None, config_path='./conf', config_name='config')
def main(cfg: DictConfig) -> None:
    print('Rank %d, local rank %d, which has device %s. Sees %d devices.' %(RANK,int(LOCAL_RANK),DEVICE,torch.cuda.device_count()))
    train(cfg)
    

    cleanup()

if __name__ == '__main__':
    main()
