import os
import sys
import socket
from typing import Optional, Union, Callable
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
import time
from pickle import UnpicklingError

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

# GNN model
from gnn import GNN_Element_Neighbor_Lo_Hi
import dataprep.nekrs_graph_setup  # needed to load the .pt data

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)

Tensor = torch.Tensor

# Get MPI:
try:
    from mpi4py import MPI

    WITH_DDP = True
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
        if RANK == 0:
            logger.warning("Found no CUDA devices")
        pass

    try:
        WITH_XPU = torch.xpu.is_available()
    except:
        WITH_XPU = False
        if RANK == 0:
            logger.warning("Found no XPU devices")
        pass

    if WITH_CUDA:
        DEVICE = torch.device("cuda")
        N_DEVICES = torch.cuda.device_count()
        DEVICE_ID = LOCAL_RANK if N_DEVICES > 1 else 0
    elif WITH_XPU:
        DEVICE = torch.device("xpu")
        N_DEVICES = torch.xpu.device_count()
        DEVICE_ID = LOCAL_RANK if N_DEVICES > 1 else 0
    else:
        DEVICE = torch.device("cpu")
        DEVICE_ID = 0
except (ImportError, ModuleNotFoundError) as e:
    WITH_DDP = False
    SIZE = 1
    RANK = 0
    LOCAL_RANK = 0
    LOCAL_SIZE = 1
    MASTER_ADDR = "localhost"
    logger.warning("MPI Initialization failed!")
    logger.warning(e)


def init_process_group(
    rank: Union[int, str],
    world_size: Union[int, str],
    backend: Optional[str] = None,
) -> None:
    if WITH_CUDA:
        backend = "nccl" if backend is None else str(backend)
    elif WITH_XPU:
        backend = "xccl" if backend is None else str(backend)
    else:
        backend = "gloo" if backend is None else str(backend)

    dist.init_process_group(
        backend,
        rank=int(rank),
        world_size=int(world_size),
        init_method="env://",
        device_id=DEVICE_ID,
    )


def force_abort():
    time.sleep(2)
    if WITH_DDP:
        if RANK == 0:
            logger.info("Aborting...")
        COMM.Abort()
    else:
        logger.info("Exiting...")
        sys.exit(1)


def cleanup():
    dist.destroy_process_group()


def metric_average(val: Tensor):
    if WITH_DDP:
        dist.all_reduce(val, op=dist.ReduceOp.SUM)
        return val / SIZE
    return val


class Trainer:
    def __init__(
        self, cfg: DictConfig, scaler: Optional[GradScaler] = None
    ) -> None:
        self.cfg = cfg
        self.rank = RANK
        if scaler is None:
            self.scaler = None
        self.device = DEVICE
        self.backend = self.cfg.backend

        # ~~~~ Init torch stuff
        self.setup_torch()

        # ~~~~ Init DDP
        if WITH_DDP:
            os.environ["RANK"] = str(RANK)
            os.environ["WORLD_SIZE"] = str(SIZE)
            MASTER_ADDR = socket.gethostname() if RANK == 0 else None
            MASTER_ADDR = COMM.bcast(MASTER_ADDR, root=0)
            os.environ["MASTER_ADDR"] = MASTER_ADDR
            os.environ["MASTER_PORT"] = str(2345)
            init_process_group(RANK, SIZE, backend=self.backend)

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
        COMM.Barrier()
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
        COMM.Barrier()

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
            if RANK == 0:
                astr = "RESTARTING FROM CHECKPOINT -- STATE AT EPOCH %d/%d" % (
                    self.epoch_start - 1,
                    self.cfg.epochs,
                )
                sepstr = "-" * len(astr)
                logger.info(sepstr)
                logger.info(astr)
                logger.info(sepstr)

        # ~~~~ Wrap model in DDP
        if WITH_DDP and SIZE > 1:
            self.model = DDP(
                self.model,
                broadcast_buffers=False,
                gradient_as_bucket_view=True,
            )

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
        optimizer = optim.Adam(model.parameters(), lr=SIZE * self.cfg.lr_init)
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

        # Device and intra-op threads
        if WITH_CUDA:
            torch.cuda.set_device(DEVICE_ID)
        elif WITH_XPU:
            torch.xpu.set_device(DEVICE_ID)
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
            if RANK == 0:
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

        if RANK == 0:
            logger.info("train dataset: %d elements" % (len(train_dataset)))
            logger.info("valid dataset: %d elements" % (len(test_dataset)))

        # DDP: use DistributedSampler to partition training data
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=SIZE,
            rank=RANK,
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
            num_replicas=SIZE,
            rank=RANK,
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

    def train_step(self, data: Data) -> Tensor:
        # t_total = time.time()
        COMM.Barrier()
        try:
            _ = data.node_weight
        except AttributeError:
            data.node_weight = data.x.new_ones(data.x.shape[0], 1)

        # coincident edge index and node degree -- only used when we have element neighbors
        edge_index_coin = (
            data.edge_index_coin if self.cfg.n_element_neighbors > 0 else None
        )
        degree = data.degree if self.cfg.n_element_neighbors > 0 else None

        if WITH_CUDA or WITH_XPU:
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
        dist.barrier()
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

        # if RANK == 0:
        #    if self.training_iter < 500:
        #        logger.info(f"t_1: {t_1}s \t t_2: {t_2}s \t t_total: {t_total}s")
        dist.barrier()

        return loss

    def train_epoch(
        self,
        epoch: int,
    ) -> dict:
        self.model.train()
        start = time.time()
        running_loss_avg = torch.tensor(0.0)

        count = torch.tensor(0.0)
        if WITH_CUDA or WITH_XPU:
            running_loss_avg = running_loss_avg.to(self.device)
            count = count.to(self.device)

        train_sampler = self.data["train"]["sampler"]
        train_loader = self.data["train"]["loader"]
        # DDP: set epoch to sampler for shuffling
        train_sampler.set_epoch(epoch)
        for bidx, data in enumerate(train_loader):
            # print('Rank %d, bid %d, data:' %(RANK, bidx), data.y[1].shape)
            loss = self.train_step(data)
            count += 1  # accumulate current batch count
            running_loss_avg = (
                running_loss_avg * (count - 1) + loss.item()
            ) / count

            self.training_iter += 1  # accumulate total training iteration

            # Log on Rank 0:
            if bidx % self.cfg.logfreq == 0 and RANK == 0:
                # DDP: use train_sampler to determine the number of
                # examples in this workers partition
                metrics = {
                    "epoch": epoch,
                    "dt": time.time() - start,
                    "batch_loss": loss.item(),
                    "running_loss_avg": running_loss_avg,
                }
                pre = [
                    f"[{RANK}]",
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
        loss_avg = metric_average(running_loss_avg)
        return {"loss": loss_avg}

    def test(self) -> dict:
        running_loss_avg = torch.tensor(0.0)
        count = torch.tensor(0.0)
        if WITH_CUDA or WITH_XPU:
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

                if WITH_CUDA or WITH_XPU:
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

            loss_avg = metric_average(running_loss_avg)

        return {"loss": loss_avg}


def train(cfg: DictConfig):
    start = time.time()
    trainer = Trainer(cfg)
    epoch_times = []
    valid_times = []

    for epoch in range(trainer.epoch_start, cfg.epochs + 1):
        # ~~~~ Training step
        t0 = time.time()
        trainer.epoch = epoch
        train_metrics = trainer.train_epoch(epoch)
        trainer.loss_hist_train[epoch - 1] = train_metrics["loss"]
        epoch_time = time.time() - t0
        epoch_times.append(epoch_time)

        # ~~~~ Validation step
        t0 = time.time()
        test_metrics = trainer.test()
        trainer.loss_hist_test[epoch - 1] = test_metrics["loss"]
        valid_time = time.time() - t0
        valid_times.append(valid_time)

        # ~~~~ Learning rate
        lr = trainer.optimizer.param_groups[0]["lr"]
        trainer.lr_hist[epoch - 1] = lr

        if RANK == 0:
            summary = "  ".join([
                "[TRAIN]",
                f"loss={train_metrics['loss']:.4e}",
                f"epoch_time={epoch_time:.4g} sec",
                f" valid_time={valid_time:.4g} sec",
                f" learning_rate={lr:.6g}",
            ])
            logger.info((sep := "-" * len(summary)))
            logger.info(summary)
            logger.info(sep)
            astr = f"[VALIDATION] loss={test_metrics['loss']:.4e}"
            sepstr = "-" * len(astr)
            logger.info(sepstr)
            logger.info(astr)
            logger.info(sepstr)

        # ~~~~ Step scheduler based on validation loss
        trainer.scheduler.step(test_metrics["loss"])

        # ~~~~ Checkpointing step
        if epoch % cfg.ckptfreq == 0 and RANK == 0:
            astr = "Checkpointing on root processor, epoch = %d" % (epoch)
            sepstr = "-" * len(astr)
            logger.info(sepstr)
            logger.info(astr)
            logger.info(sepstr)

            if not os.path.exists(cfg.ckpt_dir):
                os.makedirs(cfg.ckpt_dir)

            if WITH_DDP and SIZE > 1:
                ckpt = {
                    "epoch": epoch,
                    "training_iter": trainer.training_iter,
                    "model_state_dict": trainer.model.module.state_dict(),
                    "optimizer_state_dict": trainer.optimizer.state_dict(),
                    "scheduler_state_dict": trainer.scheduler.state_dict(),
                    "loss_hist_train": trainer.loss_hist_train,
                    "loss_hist_test": trainer.loss_hist_test,
                    "lr_hist": trainer.lr_hist,
                }
            else:
                ckpt = {
                    "epoch": epoch,
                    "training_iter": trainer.training_iter,
                    "model_state_dict": trainer.model.state_dict(),
                    "optimizer_state_dict": trainer.optimizer.state_dict(),
                    "scheduler_state_dict": trainer.scheduler.state_dict(),
                    "loss_hist_train": trainer.loss_hist_train,
                    "loss_hist_test": trainer.loss_hist_test,
                    "lr_hist": trainer.lr_hist,
                }

            torch.save(ckpt, trainer.ckpt_path)
        dist.barrier()

    rstr = f"[{RANK}] ::"
    if RANK == 0:
        logger.info(
            " ".join([
                rstr,
                f"Total training time: {time.time() - start} seconds",
            ])
        )

    if RANK == 0:
        if WITH_CUDA or WITH_XPU:
            trainer.model.to("cpu")
        if not os.path.exists(cfg.model_dir):
            os.makedirs(cfg.model_dir)
        if WITH_DDP and SIZE > 1:
            save_dict = {
                "state_dict": trainer.model.module.state_dict(),
                "input_dict": trainer.model.module.input_dict(),
                "loss_hist_train": trainer.loss_hist_train,
                "loss_hist_test": trainer.loss_hist_test,
                "lr_hist": trainer.lr_hist,
                "training_iter": trainer.training_iter,
            }
        else:
            save_dict = {
                "state_dict": trainer.model.state_dict(),
                "input_dict": trainer.model.input_dict(),
                "loss_hist_train": trainer.loss_hist_train,
                "loss_hist_test": trainer.loss_hist_test,
                "lr_hist": trainer.lr_hist,
                "training_iter": trainer.training_iter,
            }

        torch.save(save_dict, trainer.model_path)


@hydra.main(version_base=None, config_path="./conf", config_name="config")
def main(cfg: DictConfig) -> None:
    if RANK == 0:
        logger.info("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        logger.info("INPUTS:")
        logger.info(OmegaConf.to_yaml(cfg))
        logger.info("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    train(cfg)

    cleanup()


if __name__ == "__main__":
    main()
