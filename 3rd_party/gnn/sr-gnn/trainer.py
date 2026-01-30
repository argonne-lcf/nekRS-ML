import os
import sys
from omegaconf import DictConfig
from typing import Optional
import socket
import numpy as np
from pickle import UnpicklingError
import time

import torch
from torch.cuda.amp.grad_scaler import GradScaler
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP

# PyTorch Geometric
import torch_geometric
import torch_geometric.nn as tgnn
from torch_geometric.data import Data

import mpi4py.rc
mpi4py.rc.initialize = False
from mpi4py import MPI

from gnn import GNN_Element_Neighbor_Lo_Hi
import dataprep.nekrs_graph_setup  # needed to load the .pt data

import logging
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self, 
        cfg: DictConfig, 
        comm: MPI.COMM_WORLD, 
        scaler: Optional[GradScaler] = None
    ) -> None:
        self.cfg = cfg
        self.comm = comm
        self.scaler = scaler

        # ~~~ Get MPI info
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.local_rank = int(os.getenv("PALS_LOCAL_RANKID"))
        self.local_size = int(os.getenv("PALS_LOCAL_SIZE"))
        self.host_name = MPI.Get_processor_name()

        # ~~~ Initialize torch distributed
        self.init_process_group(self.cfg.master_addr, self.cfg.master_port)

        # ~~~~ Init torch stuff
        self.setup_torch()

        # ~~~~ Init training and testing loss history
        self.loss_hist_train = np.zeros(self.cfg.epochs)
        self.loss_hist_test = np.zeros(self.cfg.epochs)
        self.lr_hist = np.zeros(self.cfg.epochs)

        # ~~~~ Init datasets
        self.data = self.setup_data()

        # ~~~~ Init model and move to gpu
        self.model = self.build_model()
        self.model.to(self.device)
        self.model.to(self.torch_dtype)

        # ~~~~ Set model and checkpoint savepaths:
        if cfg.model_dir[-1] != "/":
            cfg.model_dir += "/"
        if cfg.ckpt_dir[-1] != "/":
            cfg.ckpt_dir += "/"
        try:
            self.ckpt_path = (
                cfg.ckpt_dir + self.model.get_save_header() + ".tar"
            )
            self.model_path = (
                cfg.model_dir + self.model.get_save_header() + ".tar"
            )
        except AttributeError as e:
            self.ckpt_path = cfg.ckpt_dir + "checkpoint.tar"
            self.model_path = cfg.model_dir + "model.tar"

        # ~~~~ Load model parameters if we are restarting from checkpoint
        self.comm.Barrier()
        self.epoch = 0
        self.epoch_start = 1
        self.training_iter = 0
        if self.cfg.restart:
            ckpt = torch.load(self.ckpt_path)
            self.model.load_state_dict(ckpt["model_state_dict"])
            self.epoch_start = ckpt["epoch"] + 1
            self.epoch = self.epoch_start
            self.training_iter = ckpt["training_iter"]
            self.loss_hist_train = ckpt["loss_hist_train"]
            self.loss_hist_test = ckpt["loss_hist_test"]
            self.lr_hist = ckpt["lr_hist"]

            if len(self.loss_hist_train) < self.cfg.epochs:
                loss_hist_train_new = np.zeros(self.cfg.epochs)
                loss_hist_test_new = np.zeros(self.cfg.epochs)
                lr_hist_new = np.zeros(self.cfg.epochs)
                loss_hist_train_new[: len(self.loss_hist_train)] = (
                    self.loss_hist_train
                )
                loss_hist_test_new[: len(self.loss_hist_test)] = (
                    self.loss_hist_test
                )
                lr_hist_new[: len(self.lr_hist)] = self.lr_hist
                self.loss_hist_train = loss_hist_train_new
                self.loss_hist_test = loss_hist_test_new
                self.lr_hist = lr_hist_new
        self.comm.Barrier()

        # ~~~~ Set loss function
        self.loss_fn = nn.MSELoss()

        # ~~~~ Set optimizer
        self.optimizer = self.build_optimizer(self.model)

        # ~~~~ Set scheduler
        self.scheduler = self.build_scheduler(self.optimizer)

        # ~~~~ Load optimizer+scheduler parameters if we are restarting from checkpoint
        if self.cfg.restart:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            if self.rank == 0:
                astr = "RESTARTING FROM CHECKPOINT -- STATE AT EPOCH %d/%d" % (
                    self.epoch_start - 1,
                    self.cfg.epochs,
                )
                sepstr = "-" * len(astr)
                logger.info(sepstr)
                logger.info(astr)
                logger.info(sepstr)

        # ~~~~ Wrap model in DDP
        if self.size > 1:
            self.model = DDP(
                self.model,
                broadcast_buffers=False,
                gradient_as_bucket_view=True,
            )

    def init_process_group(self, master_addr: str, master_port: int):
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
            backend = 'nccl' if self.cfg.backend is None else str(self.cfg.backend)
        elif torch.xpu.is_available():
            backend = 'xccl' if self.cfg.backend is None else str(self.cfg.backend)
        else:
            backend = 'gloo' if self.cfg.backend is None else str(self.cfg.backend)
        dist.init_process_group(backend, rank=int(self.rank), world_size=int(self.size), init_method='env://')

    def cleanup():
        dist.destroy_process_group()

    def build_model(self) -> nn.Module:

        sample = self.data["train"]["example"]

        input_node_channels = sample.x.shape[1]
        input_edge_channels_coarse = (
            sample.pos_norm_lo.shape[1] + sample.x.shape[1] + 1
        )
        hidden_channels = self.cfg.hidden_channels
        input_edge_channels_fine = (
            sample.pos_norm_hi.shape[1] + hidden_channels + 1
        )
        output_node_channels = sample.y.shape[1]
        n_mlp_hidden_layers = self.cfg.n_mlp_hidden_layers
        n_messagePassing_layers = self.cfg.n_messagePassing_layers
        use_fine_messagePassing = self.cfg.use_fine_messagePassing
        name = self.cfg.model_name
        model = GNN_Element_Neighbor_Lo_Hi(
            input_node_channels=input_node_channels,
            input_edge_channels_coarse=input_edge_channels_coarse,
            input_edge_channels_fine=input_edge_channels_fine,
            hidden_channels=hidden_channels,
            output_node_channels=output_node_channels,
            n_mlp_hidden_layers=n_mlp_hidden_layers,
            n_messagePassing_layers=n_messagePassing_layers,
            use_fine_messagePassing=use_fine_messagePassing,
            device=self.device,
            name=name,
        )
        return model

    def build_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        optimizer = optim.Adam(model.parameters(), lr=self.size * self.cfg.lr_init)
        return optimizer

    def build_scheduler(
        self, optimizer: torch.optim.Optimizer
    ) -> torch.optim.lr_scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=5,
            threshold=0.0001,
            threshold_mode="rel",
            cooldown=0,
            min_lr=1e-8,
            eps=1e-08,
        )  # verbose=True)
        return scheduler

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
        if self.cfg.precision == "fp32":
            self.torch_dtype = torch.float32
        else:
            sys.exit("Only fp32 data type is currently supported")

    def setup_data(self):
        kwargs = {}

        # multi snapshot - oneshot
        n_element_neighbors = self.cfg.n_element_neighbors
        try:
            train_dataset = torch.load(
                self.cfg.data_dir + f"/train_dataset.pt", weights_only=False
            )
            test_dataset = torch.load(
                self.cfg.data_dir + f"/valid_dataset.pt", weights_only=False
            )
        except UnpicklingError as e:  # for backward compatibility
            if self.rank == 0:
                logger.warning(f"{e}")
            torch.serialization.add_safe_globals([
                dataprep.nekrs_graph_setup.DataLoHi
            ])
            torch.serialization.add_safe_globals([
                torch_geometric.data.data.DataEdgeAttr
            ])
            torch.serialization.add_safe_globals([
                torch_geometric.data.data.DataTensorAttr
            ])
            torch.serialization.add_safe_globals([
                torch_geometric.data.storage.GlobalStorage
            ])
            train_dataset = torch.load(
                self.cfg.data_dir + f"/train_dataset.pt", weights_only=True
            )
            test_dataset = torch.load(
                self.cfg.data_dir + f"/valid_dataset.pt", weights_only=True
            )
        except Exception:
            raise

        if self.rank == 0:
            logger.info("train dataset: %d elements" % (len(train_dataset)))
            logger.info("valid dataset: %d elements" % (len(test_dataset)))

        # DDP: use DistributedSampler to partition training data
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=self.size,
            rank=self.rank,
            shuffle=True,
        )
        train_loader = torch_geometric.loader.DataLoader(
            train_dataset,
            batch_size=self.cfg.batch_size,
            follow_batch=["x", "y"],
            sampler=train_sampler,
            **kwargs,
        )

        # DDP: use DistributedSampler to partition the test data
        test_sampler = torch.utils.data.distributed.DistributedSampler(
            test_dataset,
            num_replicas=self.size,
            rank=self.rank,
            shuffle=False,
        )
        test_loader = torch_geometric.loader.DataLoader(
            test_dataset,
            batch_size=self.cfg.test_batch_size,
            follow_batch=["x", "y"],
            sampler=test_sampler,
        )

        return {
            "train": {
                "sampler": train_sampler,
                "loader": train_loader,
                "example": train_dataset[0],
                # 'stats': [data_mean, data_std]
            },
            "test": {
                "sampler": test_sampler,
                "loader": test_loader,
            },
        }

    def metric_average(self, val: torch.Tensor):
        if self.size > 1:
            dist.all_reduce(val, op=dist.ReduceOp.SUM)
            return val / self.size
        return val

    def train_step(self, data: Data) -> torch.Tensor:
        # t_total = time.time()
        self.comm.Barrier()
        try:
            _ = data.node_weight
        except AttributeError:
            data.node_weight = data.x.new_ones(data.x.shape[0], 1)

        # coincident edge index and node degree -- only used when we have element neighbors
        edge_index_coin = (
            data.edge_index_coin if self.cfg.n_element_neighbors > 0 else None
        )
        degree = data.degree if self.cfg.n_element_neighbors > 0 else None

        if self.with_cuda or self.with_xpu:
            data.x = data.x.to(self.device)
            data.x_mean_lo = data.x_mean_lo.to(self.device)
            data.x_mean_hi = data.x_mean_hi.to(self.device)
            data.x_std_lo = data.x_std_lo.to(self.device)
            data.x_std_hi = data.x_std_hi.to(self.device)
            data.node_weight = data.node_weight.to(self.device)
            data.y = data.y.to(self.device)
            data.edge_index_lo = data.edge_index_lo.to(self.device)
            data.edge_index_hi = data.edge_index_hi.to(self.device)
            data.pos_norm_lo = data.pos_norm_lo.to(self.device)
            data.pos_norm_hi = data.pos_norm_hi.to(self.device)
            data.x_batch = data.x_batch.to(self.device)
            data.y_batch = data.y_batch.to(self.device)
            data.central_element_mask = data.central_element_mask.to(
                self.device
            )
            if self.cfg.n_element_neighbors > 0:
                edge_index_coin = edge_index_coin.to(self.device)
                degree = degree.to(self.device)

        self.optimizer.zero_grad()

        # 1) Preprocessing: scale input
        eps = 1e-10
        x_scaled = (data.x - data.x_mean_lo) / (data.x_std_lo + eps)

        # 2) evaluate model
        # t_2 = time.time()
        out_gnn = self.model(
            x=x_scaled,
            mask=data.central_element_mask,
            edge_index_lo=data.edge_index_lo,
            edge_index_hi=data.edge_index_hi,
            pos_lo=data.pos_norm_lo,
            pos_hi=data.pos_norm_hi,
            batch_lo=data.x_batch,
            batch_hi=data.y_batch,
            edge_index_coin=edge_index_coin,
            degree=degree,
        )
        # t_2 = time.time() - t_2

        # 3) set the target
        if self.cfg.use_residual:
            mask = data.central_element_mask
            if data.x_batch is None:
                data.x_batch = data.edge_index_lo.new_zeros(
                    data.pos_norm_lo.size(0)
                )
            if data.y_batch is None:
                data.y_batch = data.edge_index_hi.new_zeros(
                    data.pos_norm_hi.size(0)
                )
            if self.device.type == "xpu":
                x_interp = tgnn.unpool.knn_interpolate(
                    x=data.x[mask, :].cpu(),
                    pos_x=data.pos_norm_lo[mask, :].cpu(),
                    pos_y=data.pos_norm_hi.cpu(),
                    batch_x=data.x_batch[mask].cpu(),
                    batch_y=data.y_batch.cpu(),
                    k=8,
                )
                x_interp = x_interp.to(self.device)
            else:
                x_interp = tgnn.unpool.knn_interpolate(
                    x=data.x[mask, :],
                    pos_x=data.pos_norm_lo[mask, :],
                    pos_y=data.pos_norm_hi,
                    batch_x=data.x_batch[mask],
                    batch_y=data.y_batch,
                    k=8,
                )
            target = (data.y - x_interp) / (data.x_std_hi + eps)
        else:
            target = (data.y - data.x_mean_hi) / (data.x_std_hi + eps)

        # 4) evaluate loss
        self.comm.Barrier()
        # loss = self.loss_fn(out_gnn, target) # vanilla mse
        loss = torch.mean(data.node_weight * (out_gnn - target) ** 2)

        if self.scaler is not None and isinstance(self.scaler, GradScaler):
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()

        # t_total = time.time() - t_total

        # if self.rank == 0:
        #    if self.training_iter < 500:
        #        logger.info(f"t_1: {t_1}s \t t_2: {t_2}s \t t_total: {t_total}s")
        self.comm.Barrier()

        return loss

    def train_epoch(
        self,
        epoch: int,
    ) -> dict:
        self.model.train()
        start = time.time()
        running_loss_avg = torch.tensor(0.0)

        count = torch.tensor(0.0)
        if self.with_cuda or self.with_xpu:
            running_loss_avg = running_loss_avg.to(self.device)
            count = count.to(self.device)

        train_sampler = self.data["train"]["sampler"]
        train_loader = self.data["train"]["loader"]
        # DDP: set epoch to sampler for shuffling
        train_sampler.set_epoch(epoch)
        for bidx, data in enumerate(train_loader):
            # print('Rank %d, bid %d, data:' %(self.rank, bidx), data.y[1].shape)
            loss = self.train_step(data)
            count += 1  # accumulate current batch count
            running_loss_avg = (
                running_loss_avg * (count - 1) + loss.item()
            ) / count

            self.training_iter += 1  # accumulate total training iteration

            # Log on Rank 0:
            if bidx % self.cfg.logfreq == 0 and self.rank == 0:
                # DDP: use train_sampler to determine the number of
                # examples in this workers partition
                metrics = {
                    "epoch": epoch,
                    "dt": time.time() - start,
                    "batch_loss": loss.item(),
                    "running_loss_avg": running_loss_avg,
                }
                pre = [
                    f"[{self.rank}]",
                    (  # looks like: [num_processed/total (% complete)]
                        f"[{epoch}/{self.cfg.epochs}:"
                        # f' {bidx+1}/{len(train_sampler)}'
                        f" Batch {bidx + 1}"
                        f" ({100.0 * (bidx + 1) / len(train_loader):.0f}%)]"
                    ),
                ]
                logger.info(
                    " ".join([
                        *pre,
                        *[f"{k}={v:.4e}" for k, v in metrics.items()],
                    ])
                )

        # Allreduce, mean
        loss_avg = self.metric_average(running_loss_avg)
        return {"loss": loss_avg}

    def test(self) -> dict:
        running_loss_avg = torch.tensor(0.0)
        count = torch.tensor(0.0)
        if self.with_cuda or self.with_xpu:
            running_loss_avg = running_loss_avg.to(self.device)
            count = count.to(self.device)
        self.model.eval()
        test_loader = self.data["test"]["loader"]
        with torch.no_grad():
            for data in test_loader:
                try:
                    _ = data.node_weight
                except AttributeError:
                    data.node_weight = data.x.new_ones(data.x.shape[0], 1)

                # coincident edge index and node degree -- only used when we have element neighbors
                edge_index_coin = (
                    data.edge_index_coin
                    if self.cfg.n_element_neighbors > 0
                    else None
                )
                degree = (
                    data.degree if self.cfg.n_element_neighbors > 0 else None
                )

                if self.with_cuda or self.with_xpu:
                    data.x = data.x.to(self.device)
                    data.x_mean_lo = data.x_mean_lo.to(self.device)
                    data.x_mean_hi = data.x_mean_hi.to(self.device)
                    data.x_std_lo = data.x_std_lo.to(self.device)
                    data.x_std_hi = data.x_std_hi.to(self.device)
                    data.node_weight = data.node_weight.to(self.device)
                    data.y = data.y.to(self.device)
                    data.edge_index_lo = data.edge_index_lo.to(self.device)
                    data.edge_index_hi = data.edge_index_hi.to(self.device)
                    data.pos_norm_lo = data.pos_norm_lo.to(self.device)
                    data.pos_norm_hi = data.pos_norm_hi.to(self.device)
                    data.x_batch = data.x_batch.to(self.device)
                    data.y_batch = data.y_batch.to(self.device)
                    data.central_element_mask = data.central_element_mask.to(
                        self.device
                    )
                    if self.cfg.n_element_neighbors > 0:
                        edge_index_coin = edge_index_coin.to(self.device)
                        degree = degree.to(self.device)

                # 1) Preprocessing: scale input
                eps = 1e-10
                x_scaled = (data.x - data.x_mean_lo) / (data.x_std_lo + eps)

                # 2) evaluate model
                # t_2 = time.time()
                out_gnn = self.model(
                    x=x_scaled,
                    mask=data.central_element_mask,
                    edge_index_lo=data.edge_index_lo,
                    edge_index_hi=data.edge_index_hi,
                    pos_lo=data.pos_norm_lo,
                    pos_hi=data.pos_norm_hi,
                    batch_lo=data.x_batch,
                    batch_hi=data.y_batch,
                    edge_index_coin=edge_index_coin,
                    degree=degree,
                )
                # t_2 = time.time() - t_2

                # 3) set the target -- target = data.x + GNN(x_scaled)
                if self.cfg.use_residual:
                    mask = data.central_element_mask
                    if data.x_batch is None:
                        data.x_batch = data.edge_index_lo.new_zeros(
                            data.pos_norm_lo.size(0)
                        )
                    if data.y_batch is None:
                        data.y_batch = data.edge_index_hi.new_zeros(
                            data.pos_norm_hi.size(0)
                        )
                    if self.device.type == "xpu":
                        x_interp = tgnn.unpool.knn_interpolate(
                            x=data.x[mask, :].cpu(),
                            pos_x=data.pos_norm_lo[mask, :].cpu(),
                            pos_y=data.pos_norm_hi.cpu(),
                            batch_x=data.x_batch[mask].cpu(),
                            batch_y=data.y_batch.cpu(),
                            k=8,
                        )
                        x_interp = x_interp.to(self.device)
                    else:
                        x_interp = tgnn.unpool.knn_interpolate(
                            x=data.x[mask, :],
                            pos_x=data.pos_norm_lo[mask, :],
                            pos_y=data.pos_norm_hi,
                            batch_x=data.x_batch[mask],
                            batch_y=data.y_batch,
                            k=8,
                        )
                    target = (data.y - x_interp) / (data.x_std_hi + eps)
                else:
                    target = (data.y - data.x_mean_hi) / (data.x_std_hi + eps)

                # 4) evaluate loss
                # loss = self.loss_fn(out_gnn, target) # vanilla mse
                loss = torch.mean(data.node_weight * (out_gnn - target) ** 2)

                count += 1
                running_loss_avg = (
                    running_loss_avg * (count - 1) + loss.item()
                ) / count

            loss_avg = self.metric_average(running_loss_avg)

        return {"loss": loss_avg}

    def checkpoint(self, epoch: int):
        if self.rank == 0:
            astr = "Checkpointing on root processor, epoch = %d" % (epoch)
            sepstr = "-" * len(astr)
            logger.info(sepstr)
            logger.info(astr)
            logger.info(sepstr)

            if not os.path.exists(self.cfg.ckpt_dir):
                os.makedirs(self.cfg.ckpt_dir)

            if self.size > 1:
                state_dict = self.model.module.state_dict()
            else:
                state_dict = self.model.state_dict()
            ckpt = {
                "epoch": epoch,
                "training_iter": self.training_iter,
                "model_state_dict": state_dict,
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "loss_hist_train": self.loss_hist_train,
                "loss_hist_test": self.loss_hist_test,
                "lr_hist": self.lr_hist,
            }
            torch.save(ckpt, self.ckpt_path)
        self.comm.Barrier()

    def save_model(self):
        if self.rank == 0:
            if self.with_cuda or self.with_xpu:
                self.model.to("cpu")
            if not os.path.exists(self.cfg.model_dir):
                os.makedirs(self.cfg.model_dir)
            if self.size > 1:
                state_dict = self.model.module.state_dict()
                input_dict = self.model.module.input_dict()
            else:
                state_dict = self.model.state_dict()
                input_dict = self.model.input_dict()
            save_dict = {
                "state_dict": state_dict,
                "input_dict": input_dict,
                "loss_hist_train": self.loss_hist_train,
                "loss_hist_test": self.loss_hist_test,
                "lr_hist": self.lr_hist,
                "training_iter": self.training_iter,
            }
            torch.save(save_dict, self.model_path)
        self.comm.Barrier()