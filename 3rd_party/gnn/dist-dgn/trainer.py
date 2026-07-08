"""
Trainer for distributed, consistent graph neural network
"""

import sys
import os
import socket
from typing import Optional, Union, Tuple, Dict, Any
import logging
import numpy as np
import time
from omegaconf import DictConfig, OmegaConf

import torch
from torch.cuda.amp.grad_scaler import GradScaler
import torch.nn as nn
import torch.optim as optim

import torch.distributed as dist
import torch.distributed.nn as distnn
from torch.nn.parallel import DistributedDataParallel as DDP

# PyTorch Geometric
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch_geometric.utils as pyg_utils

# Local imports
import utils
from scheduler import ScheduledOptim
import gnn
import graph_transformer as gtr
import graph_connectivity as gcon
from client import OnlineClient
import create_halo_info_par
import step_sampler
from diffusion_process import DiffusionProcess
from losses import batch_wise_mean, vlb_loss
import postprocess

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
    MASTER_ADDR = "localhost"
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
    DEVICE = torch.device("cuda")
    N_DEVICES = torch.cuda.device_count()
    DEVICE_ID = LOCAL_RANK if N_DEVICES > 1 else 0
elif WITH_XPU:
    DEVICE = torch.device("xpu")
    N_DEVICES = torch.xpu.device_count()
    DEVICE_ID = LOCAL_RANK if N_DEVICES > 1 else 0
else:
    DEVICE = torch.device("cpu")
    DEVICE_ID = "cpu"


class DGNTrainer:
    def __init__(
        self,
        cfg: DictConfig,
        scaler: Optional[GradScaler] = None,
        client: Optional[OnlineClient] = None,
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
            assert self.cfg.halo_swap_mode == "none", (
                "For inconsistent model, set halo_swap_mode=none"
            )
        assert self.cfg.prediction_type in ["epsilon", "x0", "v"], (
            f"Invalid prediction_type '{self.cfg.prediction_type}'. Must be 'epsilon', 'x0', or 'v'."
        )
        assert self.cfg.loss_weighting in ["uniform", "min_snr"], (
            f"Invalid loss_weighting '{self.cfg.loss_weighting}'. Must be 'uniform' or 'min_snr'."
        )
        if self.cfg.online:
            log.warning("Online backends not implemented for this model yet")
            COMM.Abort(1)

        # ~~~ Initialize DDP
        if WITH_DDP:
            os.environ["RANK"] = str(RANK)
            os.environ["WORLD_SIZE"] = str(SIZE)
            if self.cfg.master_addr == "none":
                MASTER_ADDR = socket.gethostname() if RANK == 0 else None
                MASTER_ADDR = COMM.bcast(MASTER_ADDR, root=0)
            else:
                MASTER_ADDR = str(cfg.master_addr)
            os.environ["MASTER_ADDR"] = MASTER_ADDR
            os.environ["MASTER_PORT"] = str(cfg.master_port)
            utils.init_process_group(RANK, SIZE)

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
        (
            self.data_reduced,
            self.data_full,
            self.idx_full2reduced,
            self.idx_reduced2full,
        ) = self.setup_local_graph()

        # ~~~~ Setup halo nodes
        self.neighboring_procs = []
        self.setup_halo()

        # ~~~~ Setup data
        self.data = {}
        self.data_list = []
        self.setup_graph_data()
        if RANK == 0:
            log.info("Done with setup_graph_data")
        if self.cfg.model_task == "train":
            self.setup_train_data()
            if RANK == 0:
                log.info("Done with setup_train_data")
        elif self.cfg.model_task == "inference":
            self.load_stats()

        # ~~~~ Setup halo exchange masks
        self.mask_send, self.mask_recv = self.build_masks()
        if RANK == 0:
            log.info("Done with build_masks")

        self.buffer_send, self.buffer_recv = self.build_buffers(
            self.cfg.mlp_hidden_channels
        )
        if RANK == 0:
            log.info("Done with build_buffers")

        # ~~~~ Build model and move to gpu
        self.model = self.build_model()
        if RANK == 0:
            log.info(
                "Built model with %i trainable parameters"
                % (self.count_weights(self.model))
            )
        self.model.to(self.device)
        self.model.to(self.torch_dtype)
        if RANK == 0:
            log.info("Done with build_model")

        # ~~~~ Set the total number of training iterations
        self.total_iterations = (
            self.cfg.phase1_steps
            + self.cfg.phase2_steps
            + self.cfg.phase3_steps
        )

        # ~~~~ Init training and validation loss history
        self.loss_hist_train = np.zeros(self.total_iterations)
        self.loss_hist_val = np.zeros(self.total_iterations)

        # ~~~~ Set model and checkpoint savepaths
        try:
            self.ckpt_path = (
                cfg.ckpt_dir + "/" + self.model.get_save_header() + ".tar"
            )
            self.model_path = (
                cfg.model_dir + "/" + self.model.get_save_header() + ".tar"
            )
        except AttributeError as e:
            self.ckpt_path = cfg.ckpt_dir + "checkpoint.tar"
            self.model_path = cfg.model_dir + "model.tar"

        # ~~~~ Load model parameters if we are restarting from checkpoint
        self.iteration = 0
        self.sample_counter = 0
        if self.cfg.restart:
            if RANK == 0:
                log.info(f"Loading model checkpoint from {self.ckpt_path}")
            ckpt = torch.load(self.ckpt_path, weights_only=False, map_location="cpu")
            self.model.load_state_dict(ckpt["model_state_dict"])
            self.iteration = ckpt["iteration"] + 1
            self.loss_hist_train = ckpt["loss_hist_train"]
            self.loss_hist_val = ckpt["loss_hist_val"]

            if len(self.loss_hist_train) < self.total_iterations:
                loss_hist_train_new = np.zeros(self.total_iterations)
                loss_hist_val_new = np.zeros(self.total_iterations)

                loss_hist_train_new[: len(self.loss_hist_train)] = (
                    self.loss_hist_train
                )
                loss_hist_val_new[: len(self.loss_hist_val)] = (
                    self.loss_hist_val
                )

                self.loss_hist_train = loss_hist_train_new
                self.loss_hist_val = loss_hist_val_new
        if self.cfg.model_task == "inference":
            if RANK == 0:
                log.info(f"Loading model checkpoint from {self.model_path}")
            ckpt = torch.load(self.model_path, weights_only=False, map_location="cpu")
            self.model.load_state_dict(ckpt["model_state_dict"])

        # ~~~~ Set optimizer
        self.optimizer = self.build_optimizer(self.model)

        # ~~~~ Load optimizer parameters if we are restarting from checkpoint
        if self.cfg.restart:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if RANK == 0:
                astr = "Restarting from checkpoint -- Iteration %d/%d" % (
                    self.iteration,
                    self.total_iterations,
                )
                log.info(astr)

        # ~~~~ Set scheduler:
        self.s_optimizer = ScheduledOptim(
            self.optimizer,
            self.cfg.phase1_steps,
            self.cfg.phase2_steps,
            self.cfg.phase3_steps,
            self.cfg.lr_phase12,
            self.cfg.lr_phase23,
        )
        self.s_optimizer.reset_n_steps(self.iteration)

        # ~~~~ Set step sampler
        if self.cfg.diffusion_step_sampler == "uniform":
            self.step_sampler = step_sampler.UniformSampler(
                num_diffusion_steps=self.cfg.num_diffusion_steps,
                device=self.device,
                dtype=self.torch_dtype,
            )
        elif self.cfg.diffusion_step_sampler == "adaptive_exponential":
            self.step_sampler = step_sampler.AdaptiveExponentialSampler(
                num_diffusion_steps=self.cfg.num_diffusion_steps,
                device=self.device,
                dtype=self.torch_dtype,
            )
        else:
            sys.exit("Invalid diffusion step sampler")

        # ~~~~ Set diffusion process
        self.diffusion_process = DiffusionProcess(
            self.cfg.num_diffusion_steps,
            self.cfg.diffusion_process_schedule,
            dtype=self.torch_dtype,
        )

        # ~~~~ Wrap model in DDP
        if WITH_DDP and SIZE > 1:
            self.model = DDP(
                self.model,
                broadcast_buffers=False,
                gradient_as_bucket_view=True,
                device_ids=[DEVICE_ID + self.cfg.device_skip],
            )

    def checkpoint(self):
        if RANK == 0:
            t_ckpt = time.time()

            if not os.path.exists(self.cfg.ckpt_dir):
                os.makedirs(self.cfg.ckpt_dir)

            if WITH_DDP and SIZE > 1:
                sd = self.model.module.state_dict()
            else:
                sd = self.model.state_dict()
            ckpt = {
                "iteration": self.iteration,
                "model_state_dict": sd,
                "optimizer_state_dict": self.optimizer.state_dict(),
                "loss_hist_train": self.loss_hist_train,
                "loss_hist_val": self.loss_hist_val,
            }
            torch.save(ckpt, self.ckpt_path + f".{self.iteration}")
            torch.save(ckpt, self.ckpt_path)
            t_ckpt = time.time() - t_ckpt

            astr = f"Checkpointing ({t_ckpt:.4g} sec)"
            sepstr = "-" * len(astr)
            log.info(sepstr)
            log.info(astr)
        COMM.Barrier()

    def save_model(self):
        if RANK == 0:
            astr = f"Finished training. Saving model to {self.model_path}."
            log.info(astr)
            if WITH_CUDA or WITH_XPU:
                self.model.to("cpu")
            if not os.path.exists(self.cfg.model_dir):
                os.makedirs(self.cfg.model_dir)

            if WITH_DDP and SIZE > 1:
                sd = self.model.module.state_dict()
                arch = self.model.module.get_arch()
            else:
                sd = self.model.state_dict()
                arch = self.model.get_arch()

            save_dict = {
                "iteration": self.iteration,
                "model_state_dict": sd,
                "arch_dict": arch,
                "loss_hist_train": self.loss_hist_train,
                "loss_hist_val": self.loss_hist_val,
            }
            torch.save(save_dict, self.model_path)
        COMM.Barrier()

    def build_model(self) -> nn.Module:
        if RANK == 0:
            log.info("In build_model...")

        # sample = self.data['train']['example']
        graph = self.data["graph"]

        # Get the polynomial order -- for naming the model
        try:
            poly = np.cbrt(self.Np) - 1.0
            poly = int(poly)
        except:
            poly = 0

        cond_node_features = (
            0
            if not self.cfg.cond_node_features
            else graph.cond_node_features.shape[1]
        )

        if self.cfg.model_name == "gnn":
            arch = {
                "input_node_features": self.cfg.input_node_features,
                "cond_node_features": cond_node_features,
                "input_edge_features": graph.edge_attr.shape[1],
                "mlp_hidden_channels": self.cfg.mlp_hidden_channels,
                "n_mlp_hidden_layers": self.cfg.n_mlp_hidden_layers,
                "n_messagePassing_layers": self.cfg.n_messagePassing_layers,
                "halo_swap_mode": self.cfg.halo_swap_mode,
                "layer_norm": self.cfg.layer_norm,
                "dropout_rate": self.cfg.dropout_rate,
                "emb_width": self.cfg.emb_width,
                "learnable_variance": self.cfg.learnable_variance,
                "activation_checkpointing": self.cfg.activation_checkpointing,
                "name": "DGN_POLY_%d_SIZE_%d_SEED_%d"
                % (poly, SIZE, self.cfg.seed),
            }
            model = gnn.DistributedDGN(arch)
        elif self.cfg.model_name == "graph_transformer":
            arch = {
                "input_node_features": self.cfg.input_node_features,
                "cond_node_features": cond_node_features,
                "hidden_channels": self.cfg.mlp_hidden_channels,
                "n_transformer_layers": self.cfg.n_transformer_layers,
                "num_heads": self.cfg.num_heads,
                "poly_order": poly,
                "halo_swap_mode": self.cfg.halo_swap_mode,
                "emb_width": self.cfg.emb_width,
                "learnable_variance": self.cfg.learnable_variance,
                "mlp_ratio": 1.0,
                "hierarchical_attention": self.cfg.hierarchical_attention,
                "hierarchical_interleve_freq": self.cfg.hierarchical_interleve_freq,
                "k_summary": self.cfg.k_summary,
                "readout_chunk_size": self.cfg.readout_chunk_size,
                "activation_checkpointing": self.cfg.activation_checkpointing,
                "name": "DGT_POLY_%d_SIZE_%d_SEED_%d"
                % (poly, SIZE, self.cfg.seed),
            }
            model = gtr.DistributedDGT(arch)
        else:
            raise ValueError("Unknown model name: %s" % self.cfg.model_name)
        return model

    def count_weights(self, model) -> int:
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return n_params

    def build_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        # optimizer = optim.Adam(model.parameters(), lr=0.0)
        optimizer = optim.AdamW(
            model.parameters(), lr=0.0, betas=(0.9, 0.95), weight_decay=0.1
        )
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
            verbose=True,
        )
        return scheduler

    def setup_torch(self):
        # Random seeds — unified across ranks so any RNG draws agree on every rank.
        seed = self.cfg.seed
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Device and intra-op threads
        if WITH_CUDA:
            torch.cuda.set_device(DEVICE_ID + self.cfg.device_skip)
        elif WITH_XPU:
            torch.xpu.set_device(DEVICE_ID + self.cfg.device_skip)
        torch.set_num_threads(self.cfg.num_threads)

        # Precision
        if self.cfg.precision == "fp32":
            self.torch_dtype = torch.float32
        elif self.cfg.precision == "bf16":
            self.torch_dtype = torch.bfloat16
        elif self.cfg.precision == "fp64":
            self.torch_dtype = torch.float64
        else:
            sys.exit(
                "Only fp32, fp64 and bf16 data types are currently supported"
            )

        # Dedicated generator for rank-consistent noise draws. Kept separate
        # from the global RNG so noise generation never perturbs DataLoader
        # shuffle order or weight init.
        self.noise_generator = torch.Generator(device=self.device)

    def _consistent_noise(
        self,
        batch_size: int,
        n_features: int,
        salt: int = 0,
        extra_counter: int = 0,
    ) -> Tensor:
        """Draw standard-normal noise that is identical across ranks on
        physically-coincident nodes, without any collective and without
        materialising a full-graph tensor.

        Each node is keyed by the byte representation of its untransformed
        mesh position (pos_orig_full, float64 -> 3 x uint64). Two ranks that
        both hold the same physical node see the same key bytes and therefore
        the same noise. The C++-side "global_ids" turned out to be
        partition-local indices (interior nodes share gids across ranks
        without referring to the same physical node), so we cannot rely on
        them; the float64 positions read straight from the C++ binary dump
        are byte-stable across processes and serve as a robust per-node key.

        Per-rank work scales with n_local + n_halo only -- no tensor sized by
        the global mesh is ever allocated.

        Implementation: SplitMix64 over a per-element 64-bit counter built
        from (seed, iteration, batch, salt) keyed material plus per-node
        mixing of the three position u64 lanes; paired through Box-Muller to
        produce standard normals. uint64 wraparound is exactly what
        SplitMix64 expects, so numpy's overflow warning is silenced.
        """
        graph = self.data["graph"]
        # Untransformed mesh positions for all local slots (tier-1 + tier-2
        # + tier-3 halo padding), filled by setup_halo via a one-time
        # neighbor exchange so tier-3 entries carry their owner's true pos.
        pos = graph.pos_orig_full.cpu().numpy().astype(np.float64)
        n_local = pos.shape[0]
        n_pairs = (n_features + 1) // 2  # Box-Muller yields 2 normals per call

        # View float64 (x,y,z) as three uint64s, lossless and byte-stable.
        pos_u64 = pos.view(np.uint64)  # shape (n_local, 3)
        px = pos_u64[:, 0]
        py = pos_u64[:, 1]
        pz = pos_u64[:, 2]

        feats = np.arange(n_pairs, dtype=np.uint64)[None, :]
        mix_xor = np.uint64(0xDEADBEEFCAFEBABE)
        denom = float(1 << 53)
        two_pi = 2.0 * float(np.pi)

        def _splitmix64(x: np.ndarray) -> np.ndarray:
            x = x + np.uint64(0x9E3779B97F4A7C15)
            x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
            x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
            x = x ^ (x >> np.uint64(31))
            return x

        # uint64 arithmetic wraps modulo 2^64 — that wraparound is the entire
        # point of SplitMix64, so silence numpy's overflow warnings for the
        # mixing math.
        parts = []
        with np.errstate(over="ignore"):
            seed_u = np.uint64(int(self.cfg.seed) & 0xFFFFFFFFFFFFFFFF)
            iter_u = np.uint64(int(self.iteration) & 0xFFFFFFFFFFFFFFFF)
            salt_u = np.uint64(int(salt) & 0xFFFFFFFFFFFFFFFF)
            # extra_counter mixes in linearly (default 0 -> no perturbation)
            # so training behaviour stays bit-identical; sample() advances it
            # once per call to give independent draws.
            extra_u = np.uint64(int(extra_counter) & 0xFFFFFFFFFFFFFFFF)
            key = (
                seed_u * np.uint64(0x9E3779B97F4A7C15)
                + iter_u * np.uint64(0xBF58476D1CE4E5B9)
                + salt_u * np.uint64(0x94D049BB133111EB)
                + extra_u * np.uint64(0xD1342543DE82EF95)
            )
            # Per-node static mix of all three position lanes (independent of
            # batch / feature). Run it through SplitMix64 once so small
            # position changes propagate to all output bits.
            pos_mix = _splitmix64(
                px * np.uint64(0xD1342543DE82EF95)
                ^ py * np.uint64(0x9E6C63D0676A9A99)
                ^ pz * np.uint64(0x6A5D39EAE12657AA)
            )
            pos_mix_2d = np.broadcast_to(pos_mix[:, None], (n_local, n_pairs))
            feat_mix = feats * np.uint64(0xCBF29CE484222325)

            for b in range(batch_size):
                b_u = np.uint64(int(b) & 0xFFFFFFFFFFFFFFFF)
                counter = (
                    pos_mix_2d
                    + feat_mix
                    + b_u * np.uint64(0x6A5D39EAE12657AA)
                    + key
                )
                h1 = _splitmix64(counter)
                h2 = _splitmix64(counter ^ mix_xor)

                # Map to uniforms. Take top 53 bits (mantissa width of f64)
                # and divide by 2^53. Bump u1 off zero so log is finite.
                u1 = (h1 >> np.uint64(11)).astype(np.float64) / denom
                u2 = (h2 >> np.uint64(11)).astype(np.float64) / denom
                np.clip(u1, 1e-300, None, out=u1)

                r = np.sqrt(-2.0 * np.log(u1))
                theta = two_pi * u2
                z0 = r * np.cos(theta)
                z1 = r * np.sin(theta)

                pair = np.stack((z0, z1), axis=-1).reshape(n_local, n_pairs * 2)
                parts.append(pair[:, :n_features])

        arr = np.concatenate(parts, axis=0)
        return torch.from_numpy(arr).to(self.device).to(self.torch_dtype)

    def halo_swap(self, input_tensor, buff_send, buff_recv):
        """
        Performs halo swap of a per-node tensor via the all_to_all collective.

        buff_send / buff_recv must be a length-SIZE list where buff_send[i]
        has shape (n_nodes_to_exchange[i], n_features) — i.e. one slot per
        neighbour rank, sized per-neighbour. Use the same buffer layout as
        the all_to_all_opt branch in DistributedMessagePassingLayer.halo_swap.
        """
        if SIZE > 1:
            for i in self.neighboring_procs:
                n_send = len(self.mask_send[i])
                buff_send[i][:n_send, :] = input_tensor[self.mask_send[i]]

            dist.all_to_all(buff_recv, buff_send)

            for i in self.neighboring_procs:
                n_recv = len(self.mask_recv[i])
                input_tensor[self.mask_recv[i]] = buff_recv[i][:n_recv, :]
        return input_tensor

    def build_masks(self):
        """
        Builds index masks for facilitating halo swap of nodes
        """
        mask_send = [torch.tensor([], dtype=self.torch_dtype)] * SIZE
        mask_recv = [torch.tensor([], dtype=self.torch_dtype)] * SIZE

        if SIZE > 1 and self.cfg.consistency:
            # n_nodes_local = self.data.n_nodes_internal + self.data.n_nodes_halo
            # halo_info = self.data['train']['example'].halo_info
            halo_info = self.data["graph"].halo_info

            for i in self.neighboring_procs:
                idx_i = halo_info[:, 3] == i
                # index of nodes to send to proc i
                mask_send[i] = halo_info[:, 0][idx_i]

                # index of nodes to receive from proc i
                mask_recv[i] = halo_info[:, 1][idx_i]

                if len(mask_send[i]) != len(mask_recv[i]):
                    log.info(
                        "For neighbor rank %d, the number of send nodes and the number of receive nodes do not match. Check to make sure graph is partitioned correctly."
                        % (i)
                    )
                    utils.force_abort()
        return mask_send, mask_recv

    def build_buffers(self, n_features):
        self.n_max = 0

        if SIZE == 1:
            buff_send = [torch.tensor([], dtype=self.torch_dtype)] * SIZE
            buff_recv = [torch.tensor([], dtype=self.torch_dtype)] * SIZE
        else:
            # Get the maximum number of nodes that will be exchanged (required for all_to_all halo swap)
            self.n_nodes_to_exchange = torch.zeros(SIZE)
            for i in self.neighboring_procs:
                self.n_nodes_to_exchange[i] = len(self.mask_send[i])
            self.n_max = self.n_nodes_to_exchange.max()
            if WITH_CUDA or WITH_XPU:
                self.n_max = self.n_max.to(self.device)
            dist.all_reduce(self.n_max, op=dist.ReduceOp.MAX)
            self.n_max = int(self.n_max)

            # fill the buffers -- make all buffer sizes the same (required for all_to_all)
            if self.cfg.halo_swap_mode == "none":
                buff_send = [
                    torch.empty(0, device=DEVICE, dtype=self.torch_dtype)
                ] * SIZE
                buff_recv = [
                    torch.empty(0, device=DEVICE, dtype=self.torch_dtype)
                ] * SIZE
            elif self.cfg.halo_swap_mode == "all_to_all":
                buff_send = [
                    torch.empty(0, device=DEVICE, dtype=self.torch_dtype)
                ] * SIZE
                buff_recv = [
                    torch.empty(0, device=DEVICE, dtype=self.torch_dtype)
                ] * SIZE
                for i in range(SIZE):
                    buff_send[i] = torch.empty(
                        [self.n_max, n_features],
                        dtype=self.torch_dtype,
                        device=DEVICE,
                    )
                    buff_recv[i] = torch.empty(
                        [self.n_max, n_features],
                        dtype=self.torch_dtype,
                        device=DEVICE,
                    )
            elif self.cfg.halo_swap_mode == "all_to_all_opt":
                buff_send = [
                    torch.empty(0, device=DEVICE, dtype=self.torch_dtype)
                ] * SIZE
                buff_recv = [
                    torch.empty(0, device=DEVICE, dtype=self.torch_dtype)
                ] * SIZE
                for i in self.neighboring_procs:
                    buff_send[i] = torch.empty(
                        [int(self.n_nodes_to_exchange[i]), n_features],
                        dtype=self.torch_dtype,
                        device=DEVICE,
                    )
                    buff_recv[i] = torch.empty(
                        [int(self.n_nodes_to_exchange[i]), n_features],
                        dtype=self.torch_dtype,
                        device=DEVICE,
                    )
            elif self.cfg.halo_swap_mode == "all_to_all_opt_intel":
                buff_send = [
                    torch.zeros(1, device=DEVICE, dtype=self.torch_dtype)
                ] * SIZE
                buff_recv = [
                    torch.zeros(1, device=DEVICE, dtype=self.torch_dtype)
                ] * SIZE
                for i in self.neighboring_procs:
                    buff_send[i] = torch.zeros(
                        [int(self.n_nodes_to_exchange[i]), n_features],
                        dtype=self.torch_dtype,
                        device=DEVICE,
                    )
                    buff_recv[i] = torch.zeros(
                        [int(self.n_nodes_to_exchange[i]), n_features],
                        dtype=self.torch_dtype,
                        device=DEVICE,
                    )
            elif self.cfg.halo_swap_mode == "send_recv":
                buff_send = [
                    torch.empty(0, device=DEVICE, dtype=self.torch_dtype)
                ] * SIZE
                buff_recv = [
                    torch.empty(0, device=DEVICE, dtype=self.torch_dtype)
                ] * SIZE
                for i in self.neighboring_procs:
                    buff_send[i] = torch.empty(
                        [int(self.n_nodes_to_exchange[i]), n_features],
                        dtype=self.torch_dtype,
                        device=DEVICE,
                    )
                    buff_recv[i] = torch.empty(
                        [int(self.n_nodes_to_exchange[i]), n_features],
                        dtype=self.torch_dtype,
                        device=DEVICE,
                    )

            # for i in self.neighboring_procs:
            #    buff_send[i] = torch.empty([len(self.mask_send[i]), n_features], dtype=torch.float32, device=DEVICE_ID)
            #    buff_recv[i] = torch.empty([len(self.mask_recv[i]), n_features], dtype=torch.float32, device=DEVICE_ID)

            # Measure the size of the buffers
            buff_send_sz = [0] * SIZE
            buff_recv_sz = [0] * SIZE
            for i in range(SIZE):
                buff_send_sz[i] = (
                    torch.numel(buff_send[i])
                    * buff_send[i].element_size()
                    / 1024
                )
                buff_recv_sz[i] = (
                    torch.numel(buff_recv[i])
                    * buff_recv[i].element_size()
                    / 1024
                )

            # Print information about the buffers
            if RANK == 0:
                log.info(
                    "[RANK %d]: Created send and receive buffers for %s halo exchange:"
                    % (RANK, self.cfg.halo_swap_mode)
                )
                log.info(
                    f"[RANK {RANK}]: Send buffers of size [KB]: {buff_send_sz}"
                )
                log.info(
                    f"[RANK {RANK}]: Receive buffers of size [KB]: {buff_recv_sz}"
                )
            elif self.cfg.verbose:
                log.info(
                    "[RANK %d]: Created send and receive buffers for %s halo exchange:"
                    % (RANK, self.cfg.halo_swap_mode)
                )
                log.info(
                    f"[RANK {RANK}]: Send buffers of size [KB]: {buff_send_sz}"
                )
                log.info(
                    f"[RANK {RANK}]: Receive buffers of size [KB]: {buff_recv_sz}"
                )

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

        n_nodes_procs = (
            list(torch.empty([1], dtype=torch.int64, device=DEVICE)) * SIZE
        )
        if WITH_CUDA or WITH_XPU:
            n_nodes = n_nodes.to(self.device)
        dist.all_gather(n_nodes_procs, n_nodes)

        gather_list = None
        if RANK == 0:
            gather_list = [None] * SIZE
            for i in range(SIZE):
                gather_list[i] = torch.empty(
                    [n_nodes_procs[i], n_features], dtype=dtype, device=DEVICE
                )
        dist.gather(input_tensor, gather_list, dst=0)
        return gather_list

    def load_data(
        self,
        file_name,
        dtype: Optional[type] = np.float64,
        extension: Optional[str] = "",
    ):
        if not self.cfg.online:
            # check extension anyway
            ext = file_name.split(".")[-1]
            if extension == ".bin" or ext == "bin":
                data = np.fromfile(file_name + extension, dtype=dtype)
            elif extension == ".npy" or ext == "npy":
                data = np.load(file_name + extension)
            elif extension == ".npz" or ext == "npz":
                data = np.load(file_name + extension)
            else:
                data = np.loadtxt(file_name, dtype=dtype)
        else:
            data = self.client.get_array(file_name).astype(dtype)
            if isinstance(file_name, str):
                if "edge_index" not in file_name:
                    data = data.T
            else:
                data = data.T
        return data

    def load_graph_data(self):
        """
        Load in the local graph
        """
        if RANK == 0:
            log.info("Setting up the graph ...")
        if not self.cfg.online:
            main_path = self.cfg.gnn_outputs_path + "/"
        else:
            main_path = ""

        path_to_pos_full = main_path + "pos_node_rank_%d_size_%d" % (RANK, SIZE)
        path_to_ei = main_path + "edge_index_rank_%d_size_%d" % (RANK, SIZE)
        path_to_overlap = main_path + "overlap_ids_rank_%d_size_%d" % (
            RANK,
            SIZE,
        )
        path_to_glob_ids = main_path + "global_ids_rank_%d_size_%d" % (
            RANK,
            SIZE,
        )
        path_to_unique_local = (
            main_path + "local_unique_mask_rank_%d_size_%d" % (RANK, SIZE)
        )
        path_to_unique_halo = main_path + "halo_unique_mask_rank_%d_size_%d" % (
            RANK,
            SIZE,
        )

        # Polynomial order
        self.Np = np.array([0], dtype=np.float32)
        if RANK == 0:
            path_to_Np = main_path + "Np_rank_%d_size_%d" % (RANK, SIZE)
            self.Np = self.load_data(path_to_Np, dtype=np.float32)
        COMM.Bcast(self.Np, root=0)

        # Node positions
        if self.cfg.verbose:
            log.info(
                "[RANK %d]: Loading positions and global node index" % (RANK)
            )
        # pos = np.fromfile(self.cfg.gnn_outputs_path+'/'+path_to_pos_full + ".bin", dtype=np.float64).reshape((-1,3))
        pos = self.load_data(path_to_pos_full, extension=".bin").reshape((
            -1,
            3,
        ))

        # Global node index
        # gli = np.fromfile(self.cfg.gnn_outputs_path+'/'+path_to_glob_ids + ".bin", dtype=np.int64).reshape((-1,1))
        gli = self.load_data(
            path_to_glob_ids, dtype=np.int64, extension=".bin"
        ).reshape((-1, 1))

        # Edge index
        if self.cfg.verbose:
            log.info("[RANK %d]: Loading edge index" % (RANK))
        # ei = np.fromfile(self.cfg.gnn_outputs_path+'/'+path_to_ei + ".bin", dtype=np.int32).reshape((-1,2)).T
        ei = self.load_data(path_to_ei, dtype=np.int32, extension=".bin")
        if not self.cfg.online:
            ei = ei.reshape((-1, 2)).T
        ei = ei.astype(np.int64)  # sb: int64 for edge_index

        # Local unique mask
        if self.cfg.verbose:
            log.info("[RANK %d]: Loading local unique mask" % (RANK))
        # local_unique_mask = np.fromfile(self.cfg.gnn_outputs_path+'/'+path_to_unique_local + ".bin", dtype=np.int32)
        local_unique_mask = self.load_data(
            path_to_unique_local, dtype=np.int32, extension=".bin"
        )

        # Halo unique mask
        halo_unique_mask = np.array([])
        if SIZE > 1:
            # halo_unique_mask = np.fromfile(self.cfg.gnn_outputs_path+'/'+path_to_unique_halo + ".bin", dtype=np.int32)
            halo_unique_mask = self.load_data(
                path_to_unique_halo, dtype=np.int32, extension=".bin"
            )

        # Conditional node features
        cond_node_features = np.array([])
        if self.cfg.cond_node_features:
            path_to_cond_node_features = (
                main_path + "cond_node_features_rank_%d_size_%d" % (RANK, SIZE)
            )
            cond_node_features = self.load_data(
                path_to_cond_node_features, extension=".bin"
            ).reshape((pos.shape[0], -1))

        return (
            pos,
            gli,
            ei,
            local_unique_mask,
            halo_unique_mask,
            cond_node_features,
        )

    def setup_local_graph(self):
        """
        Setup the local graph
        """
        # ~~~~ Read the graph data structures
        (
            pos,
            gli,
            ei,
            local_unique_mask,
            halo_unique_mask,
            cond_node_features,
        ) = self.load_graph_data()

        # Address periodicity
        pos = pos.astype(NP_FLOAT_DTYPE)
        pos_orig = np.copy(pos)
        pos = pos.astype(NP_FLOAT_DTYPE)
        pos_orig = np.copy(pos)
        if self.cfg.transform_x:
            xmin_loc = np.amin(pos[:, 0])
            xmin_glob = np.zeros_like(xmin_loc)
            COMM.Allreduce(xmin_loc, xmin_glob, op=MPI.MIN)
            xmax_loc = np.amax(pos[:, 0])
            xmax_glob = np.zeros_like(xmax_loc)
            COMM.Allreduce(xmax_loc, xmax_glob, op=MPI.MAX)
            L_x = (xmax_glob - xmin_glob) / 2.0
            pos[:, 0] = np.abs((pos[:, 0] % L_x) - L_x / 2)  # piecewise linear

        if self.cfg.transform_y:
            ymin_loc = np.amin(pos[:, 1])
            ymin_glob = np.zeros_like(ymin_loc)
            COMM.Allreduce(ymin_loc, ymin_glob, op=MPI.MIN)
            ymax_loc = np.amax(pos[:, 1])
            ymax_glob = np.zeros_like(ymax_loc)
            COMM.Allreduce(ymax_loc, ymax_glob, op=MPI.MAX)
            L_y = (ymax_glob - ymin_glob) / 2.0
            pos[:, 1] = np.abs((pos[:, 1] % L_y) - L_y / 2)  # piecewise linear

        if self.cfg.transform_z:
            zmin_loc = np.amin(pos[:, 2])
            zmin_glob = np.zeros_like(zmin_loc)
            COMM.Allreduce(zmin_loc, zmin_glob, op=MPI.MIN)
            zmax_loc = np.amax(pos[:, 2])
            zmax_glob = np.zeros_like(zmax_loc)
            COMM.Allreduce(zmax_loc, zmax_glob, op=MPI.MAX)
            L_z = (zmax_glob - zmin_glob) / 2.0
            # pos[:,2] = np.cos(2.*np.pi*pos[:,2]/L_z) # cosine
            pos[:, 2] = np.abs((pos[:, 2] % L_z) - L_z / 2)  # piecewise linear

        # ~~~~ Compute global pos min/max per coordinate.
        # Used by graph_transformer to normalize coordinates for RoPE
        pos_min_loc = np.amin(pos, axis=0).astype(NP_FLOAT_DTYPE)
        pos_max_loc = np.amax(pos, axis=0).astype(NP_FLOAT_DTYPE)
        pos_min_glob = np.zeros_like(pos_min_loc)
        pos_max_glob = np.zeros_like(pos_max_loc)
        COMM.Allreduce(pos_min_loc, pos_min_glob, op=MPI.MIN)
        COMM.Allreduce(pos_max_loc, pos_max_glob, op=MPI.MAX)

        # ~~~~ Make the full graph:
        if self.cfg.verbose:
            log.info(
                "[RANK %d]: Making the FULL GLL-based graph with overlapping nodes"
                % (RANK)
            )
        data_full = Data(
            x=None,
            edge_index=torch.tensor(ei),
            pos_orig=torch.tensor(pos_orig),
            pos=torch.tensor(pos),
            global_ids=torch.tensor(gli.squeeze()),
            local_unique_mask=torch.tensor(local_unique_mask),
            halo_unique_mask=torch.tensor(halo_unique_mask),
            cond_node_features=torch.tensor(
                cond_node_features, dtype=self.torch_dtype
            ),
        )
        data_full.edge_index = pyg_utils.remove_self_loops(
            data_full.edge_index
        )[0]
        data_full.edge_index = pyg_utils.coalesce(data_full.edge_index)
        data_full.edge_index = pyg_utils.to_undirected(data_full.edge_index)
        data_full.local_ids = torch.tensor(range(data_full.pos.shape[0]))

        # ~~~~ Get reduced (non-overlapping) graph and indices to go from full to reduced
        if self.cfg.verbose:
            log.info(
                "[RANK %d]: Making the REDUCED GLL-based graph with non-overlapping nodes"
                % (RANK)
            )
        data_reduced, idx_full2reduced = gcon.get_reduced_graph(data_full)

        # Stash global pos bounds on data_reduced so they flow into self.data["graph"]
        data_reduced.pos_min = torch.tensor(
            pos_min_glob, dtype=self.torch_dtype
        )
        data_reduced.pos_max = torch.tensor(
            pos_max_glob, dtype=self.torch_dtype
        )

        # Snapshot the raw global ids before get_upsample_indices runs --
        # update_global_ids() inside it mutates data_reduced.global_ids by
        # overwriting zero sentinels with *rank-local* consecutive negatives,
        # which destroys cross-rank consistency. Keep the raw mesh ids here
        # so _consistent_noise can hash a physically-consistent key. Zero
        # sentinels still collide (all gid=0 nodes share noise) but that
        # collision is identical on every rank.
        data_reduced.global_ids_raw = data_reduced.global_ids.clone()

        # ~~~~ Get the indices to go from reduced back to full graph
        if self.cfg.verbose:
            log.info("[RANK %d]: Getting idx_reduced2full" % (RANK))
        idx_reduced2full = gcon.get_upsample_indices(
            data_full, data_reduced, idx_full2reduced
        )

        # Checks on mappings
        try:
            assert torch.allclose(
                data_full.pos[idx_full2reduced], data_reduced.pos
            )
        except AssertionError as e:
            idx = torch.where(
                data_full.pos[idx_full2reduced] != data_reduced.pos
            )
            log.error(
                "RANK %i: AssertionError: Non-matching nodes found in idx_full2reduced",
                RANK,
            )
            log.error("Number of non-matching nodes:", len(idx[0]))
            log.error("Non-matching nodes:", idx[0])
            raise e
        try:
            assert torch.allclose(
                data_reduced.pos[idx_reduced2full], data_full.pos
            )
        except AssertionError as e:
            idx = torch.where(
                data_reduced.pos[idx_reduced2full] != data_full.pos
            )
            log.error(
                "RANK %i: AssertionError: Non-matching nodes found in idx_reduced2full",
                RANK,
            )
            log.error("Number of non-matching nodes:", len(idx[0]))
            log.error("Non-matching nodes:", idx[0])
            raise e

        return data_reduced, data_full, idx_full2reduced, idx_reduced2full

    def setup_halo(self):
        if SIZE > 1 and self.cfg.consistency:
            if self.cfg.verbose:
                log.info(
                    "[RANK %d]: Assembling halo_ids_list using reduced graph"
                    % (RANK)
                )
            if not self.cfg.online:
                path_to_ew = (
                    self.cfg.gnn_outputs_path
                    + "/edge_weights_rank_%d_size_%d" % (RANK, SIZE)
                )
                path_to_node_degree = (
                    self.cfg.gnn_outputs_path
                    + "/node_degree_rank_%d_size_%d" % (RANK, SIZE)
                )
                path_to_halo_info = (
                    self.cfg.gnn_outputs_path
                    + "/halo_info_rank_%d_size_%d" % (RANK, SIZE)
                )
                edge_freq = torch.tensor(
                    self.load_data(path_to_ew, extension=".npy"),
                    dtype=self.torch_dtype,
                )
                edge_weight = 1.0 / edge_freq
                node_degree = torch.tensor(
                    self.load_data(path_to_node_degree, extension=".npy"),
                    dtype=self.torch_dtype,
                )
                halo_info = torch.tensor(
                    self.load_data(path_to_halo_info, extension=".npy")
                )
            else:
                if self.client.file_exists(
                    f"halo_info_rank_{RANK}_size_{SIZE}"
                ):
                    halo_info = torch.tensor(
                        self.client.get_array(
                            f"halo_info_rank_{RANK}_size_{SIZE}"
                        )
                    )
                    node_degree = torch.tensor(
                        self.client.get_array(
                            f"node_degree_rank_{RANK}_size_{SIZE}"
                        )
                    )
                    edge_weight = torch.tensor(
                        self.client.get_array(
                            f"edge_weight_rank_{RANK}_size_{SIZE}"
                        )
                    )
                else:
                    tic = time.time()
                    halo_ids = create_halo_info_par.get_reduced_halo_ids(
                        self.data_reduced
                    )
                    halo_info_glob = create_halo_info_par.get_halo_info_fast(
                        self.data_reduced, halo_ids
                    )
                    if RANK == 0:
                        log.info(
                            "[RANK %d]: computed halo info in %f sec"
                            % (RANK, time.time() - tic)
                        )
                    halo_info = halo_info_glob[RANK]
                    self.client.put_array(
                        f"halo_info_rank_{RANK}_size_{SIZE}", halo_info.numpy()
                    )

                    tic = time.time()
                    node_degree = create_halo_info_par.get_node_degree(
                        self.data_reduced, halo_info
                    )
                    if RANK == 0:
                        log.info(
                            "[RANK %d]: computed node degree in %f sec"
                            % (RANK, time.time() - tic)
                        )
                    self.client.put_array(
                        f"node_degree_rank_{RANK}_size_{SIZE}",
                        node_degree.numpy(),
                    )

                    tic = time.time()
                    edge_freq = create_halo_info_par.get_edge_weights(
                        self.data_reduced, halo_info_glob
                    )
                    edge_weight = (1.0 / edge_freq).to(self.torch_dtype)
                    if RANK == 0:
                        log.info(
                            "[RANK %d]: computed edge weights in %f sec"
                            % (RANK, time.time() - tic)
                        )
                    self.client.put_array(
                        f"edge_weight_rank_{RANK}_size_{SIZE}",
                        edge_weight.to(torch.float32).numpy(),
                    )

            self.neighboring_procs = np.unique(halo_info[:, 3])
            n_nodes_local = self.data_reduced.pos.shape[0]
            n_nodes_halo = halo_info.shape[0]
            if self.cfg.verbose:
                log.info(
                    f"[RANK {RANK}]: Found {len(self.neighboring_procs)} neighboring processes: {self.neighboring_procs}"
                )
            else:
                if RANK == 0:
                    log.info(
                        f"[RANK {RANK}]: Found {len(self.neighboring_procs)} neighboring processes: {self.neighboring_procs}"
                    )

            # mpi4py converts torch buffers via the numpy buffer protocol,
            # which has no native bf16 -- so an Allreduce on a bf16 tensor
            # forces IPEX to do a non-zero-copy conversion and prints the
            # "calling in ipex numpy ..." warning every rank, every run.
            # Do the reduction in fp64 (one scalar -- negligible) and cast
            # back to the model dtype for downstream consumers. The result
            # is a node count, so fp64 is more accurate than bf16 anyway.
            effective_nodes_local = torch.sum(
                1.0 / node_degree[:n_nodes_local]
            ).to(torch.float64)
            effective_nodes = torch.zeros(1, dtype=torch.float64)
            COMM.Allreduce(effective_nodes_local, effective_nodes, op=MPI.SUM)
            effective_nodes_local = effective_nodes_local.to(self.torch_dtype)
            effective_nodes = effective_nodes.to(self.torch_dtype)
        else:
            halo_info = torch.zeros(1, dtype=self.torch_dtype)
            n_nodes_local = self.data_reduced.pos.shape[0]
            n_nodes_halo = 0
            n_edges_local = self.data_reduced.edge_index.shape[1]
            edge_weight = torch.ones(n_edges_local, dtype=self.torch_dtype)
            node_degree = torch.ones(n_nodes_local, dtype=self.torch_dtype)
            effective_nodes_local = n_nodes_local
            effective_nodes = torch.tensor(
                effective_nodes_local, dtype=self.torch_dtype
            )

        self.data_reduced.n_nodes_local = torch.tensor(
            n_nodes_local, dtype=torch.int64
        )
        self.data_reduced.n_nodes_halo = torch.tensor(
            n_nodes_halo, dtype=torch.int64
        )
        self.data_reduced.halo_info = halo_info
        self.data_reduced.edge_weight = edge_weight
        self.data_reduced.node_degree = node_degree
        self.data_reduced.effective_nodes_local = effective_nodes_local
        self.data_reduced.effective_nodes = effective_nodes

        # Build pos_orig_full: a length-(n_nodes_local + n_nodes_halo) tensor
        # of untransformed mesh positions for every local slot (tier-1 owned,
        # tier-2 boundary, tier-3 halo). Used by _consistent_noise to hash
        # physically-coincident nodes to identical noise across ranks.
        # Positions are the key because the C++-side "global_ids" turn out to
        # be partition-local: only halo-unique nodes get consistent ids across
        # ranks, interior nodes do not. Positions are stable in nekRS' mesh
        # representation (float64 from the C++ binary file).
        pos_orig_local = self.data_reduced.pos_orig.to(torch.float64)
        n_dim = pos_orig_local.shape[1]
        pos_orig_full = torch.zeros(
            n_nodes_local + n_nodes_halo, n_dim, dtype=torch.float64
        )
        pos_orig_full[:n_nodes_local] = pos_orig_local

        if self.cfg.consistency and SIZE > 1:
            # One-shot per-neighbor MPI Sendrecv of positions. halo_info[k]
            # tells us: my local tier-2 node at column-0 is a shared copy of
            # rank column-3's tier-2 node, and I should place that rank's
            # position into my tier-3 slot at column-1. Since the partner's
            # halo_info has the same structure with us as the source rank, the
            # exchange is symmetric and we can pair sends with receives by
            # neighbor rank.
            neighbor_ranks = sorted(set(int(r) for r in self.neighboring_procs))
            send_reqs = []
            recv_buffers = {}
            for nr in neighbor_ranks:
                sel = halo_info[:, 3] == nr
                send_idx = halo_info[sel, 0].long().numpy()
                recv_idx = halo_info[sel, 1].long().numpy()
                # Send our positions at our local tier-2 nodes; receive the
                # neighbor's positions for our tier-3 slots.
                send_buf = pos_orig_local.numpy()[send_idx].astype(
                    np.float64, copy=True
                )
                recv_buf = np.empty((len(recv_idx), n_dim), dtype=np.float64)
                recv_buffers[nr] = (recv_buf, recv_idx)
                # Non-blocking sendrecv pair. Tag by (RANK,nr) ordered pair to
                # disambiguate if multiple exchanges happen later.
                tag = (min(RANK, nr) << 16) | max(RANK, nr)
                req_s = COMM.Isend([send_buf, MPI.DOUBLE], dest=nr, tag=tag)
                req_r = COMM.Irecv([recv_buf, MPI.DOUBLE], source=nr, tag=tag)
                send_reqs.append((req_s, send_buf))
                send_reqs.append((req_r, None))
            for req, _ in send_reqs:
                req.Wait()
            for nr, (recv_buf, recv_idx) in recv_buffers.items():
                pos_orig_full[recv_idx] = torch.from_numpy(recv_buf)
        self.data_reduced.pos_orig_full = pos_orig_full

        # Halo-exchange cond_node_features the same way as positions: it is a
        # static per-node tensor (walldist/inflowdist/ycoord etc.), so a single
        # exchange at setup time is enough. Without this, tier-3 halo slots
        # hold zeros and the node encoder sees a fictional conditioning on the
        # partition boundary — which then leaks into owned nodes via message
        # passing and breaks rank-consistency of the loss.
        if (
            self.cfg.consistency
            and SIZE > 1
            and self.cfg.cond_node_features
            and hasattr(self.data_reduced, "cond_node_features")
            and self.data_reduced.cond_node_features.numel() > 0
        ):
            cnf_local = self.data_reduced.cond_node_features
            n_features_cnf = cnf_local.shape[1]
            cnf_full = torch.zeros(
                n_nodes_local + n_nodes_halo,
                n_features_cnf,
                dtype=cnf_local.dtype,
            )
            cnf_full[:n_nodes_local] = cnf_local
            # Cast to fp64 BEFORE .numpy() so we don't trigger IPEX's "not
            # share memory" warning on bf16 (numpy has no native bf16, so the
            # conversion must copy anyway -- the .to(fp64) makes that explicit
            # and gives us the dtype we'd cast to next).
            cnf_local_np = cnf_local.cpu().to(torch.float64).numpy()
            send_reqs = []
            recv_buffers = {}
            for nr in neighbor_ranks:
                sel = halo_info[:, 3] == nr
                send_idx = halo_info[sel, 0].long().numpy()
                recv_idx = halo_info[sel, 1].long().numpy()
                send_buf = cnf_local_np[send_idx]
                recv_buf = np.empty(
                    (len(recv_idx), n_features_cnf), dtype=np.float64
                )
                recv_buffers[nr] = (recv_buf, recv_idx)
                # Different tag from the position exchange to avoid collisions.
                tag = ((min(RANK, nr) << 16) | max(RANK, nr)) ^ 0x5A5A
                req_s = COMM.Isend([send_buf, MPI.DOUBLE], dest=nr, tag=tag)
                req_r = COMM.Irecv([recv_buf, MPI.DOUBLE], source=nr, tag=tag)
                send_reqs.append((req_s, send_buf))
                send_reqs.append((req_r, None))
            for req, _ in send_reqs:
                req.Wait()
            for nr, (recv_buf, recv_idx) in recv_buffers.items():
                cnf_full[recv_idx] = torch.from_numpy(recv_buf).to(
                    cnf_local.dtype
                )
            self.data_reduced.cond_node_features = cnf_full
        return

    def prepare_snapshot_data(self, data_x: np.ndarray):
        data_x = data_x.astype(NP_FLOAT_DTYPE)  # force NP_FLOAT_DTYPE

        # Retain only N_gll = Np*Ne elements
        N_gll = self.data_full.pos.shape[0]
        data_x = data_x[:N_gll, :]

        # get data in reduced format
        data_x_reduced = data_x[self.idx_full2reduced, :]
        x = torch.tensor(data_x_reduced, dtype=self.torch_dtype)

        # Add halo nodes by appending the end of the node arrays
        if self.cfg.consistency:
            n_nodes_halo = self.data_reduced.n_nodes_halo
            n_features_x = data_x_reduced.shape[1]
            data_x_halo = torch.zeros(
                (n_nodes_halo, n_features_x), dtype=self.torch_dtype
            )
            x = torch.cat((x, data_x_halo), dim=0)
        return x

    def compute_statistics(self, data_list: list, var: str):
        device = "cpu"
        n_features = data_list[0][var].shape[1]
        n_nodes_local = self.data_reduced.n_nodes_local
        n_snaps = len(data_list)
        x_full = torch.zeros(
            (n_snaps, n_nodes_local, n_features), dtype=self.torch_dtype
        )
        for i in range(len(data_list)):
            x_full[i, :, :] = data_list[i][var][:n_nodes_local, :]

        # Weight each row by 1/node_degree so halo-unique rows shared with a
        # neighbor rank are not double-counted in the global mean/variance.
        weights = (1.0 / self.data_reduced.node_degree[:n_nodes_local]).to(
            self.torch_dtype
        )
        w = weights.view(1, -1, 1)
        n_scale_local = weights.sum() * n_snaps

        data_mean_ = (x_full * w).sum(dim=(0, 1)).to(device) / n_scale_local
        data_var_ = (((x_full - data_mean_.view(1, 1, -1)) ** 2) * w).sum(
            dim=(0, 1)
        ).to(device) / n_scale_local
        n_scale_ = torch.tensor(
            [n_scale_local.item()], dtype=self.torch_dtype, device=device
        )

        data_mean_gather = [
            torch.zeros(n_features, dtype=self.torch_dtype, device=device)
            for _ in range(SIZE)
        ]
        data_mean_gather = utils.mpi_all_gather(data_mean_)

        data_var_gather = [
            torch.zeros(n_features, dtype=self.torch_dtype, device=device)
            for _ in range(SIZE)
        ]
        data_var_gather = utils.mpi_all_gather(data_var_)

        n_scale_gather = [
            torch.zeros(1, dtype=self.torch_dtype, device=device)
            for _ in range(SIZE)
        ]
        n_scale_gather = utils.mpi_all_gather(n_scale_)

        data_mean_gather = torch.stack(data_mean_gather)
        data_var_gather = torch.stack(data_var_gather)
        n_scale_gather = torch.stack(n_scale_gather)

        data_mean = torch.sum(
            n_scale_gather * data_mean_gather, axis=0
        ) / torch.sum(n_scale_gather)
        data_mean = data_mean.unsqueeze(0)

        num_1 = torch.sum(
            n_scale_gather * data_var_gather, axis=0
        )  # n_i * var_i
        num_2 = torch.sum(
            n_scale_gather * (data_mean_gather - data_mean) ** 2, axis=0
        )
        data_var = (num_1 + num_2) / torch.sum(n_scale_gather)
        data_std = torch.sqrt(data_var)
        data_std = data_std.unsqueeze(0)
        return data_mean, data_std

    def load_field_data(self, data_dir: str):
        if RANK == 0:
            log.info("Loading field data...")
        field_name = "u"  # velocity

        # read files
        if not self.cfg.online:
            file_list = os.listdir(data_dir)
            files = [
                item
                for item in file_list
                if (f"fld_{field_name}" in item) and (f"rank_{RANK}_" in item)
            ]
            files.sort(key=lambda x: int(x.split(".")[0].split("_")[-1]))
        else:
            log.warning("Online backends not implemented for this model yet")
            COMM.Abort(1)

        # populate dataset
        if not self.cfg.online:
            path_prepend = data_dir + "/"
            files = [path_prepend + file for file in files]
        log.info(f"[RANK {RANK}]: Found {len(files)} field files to load")
        for i in range(len(files)):
            tic = time.time()
            data_x = self.load_data(files[i], dtype=np.float64).reshape((
                -1,
                self.cfg.input_node_features,
            ))
            toc = time.time()
            if self.cfg.online:
                self.online_timers["trainDataTime"].append(toc - tic)
                self.online_timers["trainDataThroughput"].append(
                    data_x.nbytes / GB_SIZE / (toc - tic)
                )
            data_x = self.prepare_snapshot_data(data_x)
            self.data_list.append({"x": data_x})

        # split into train/validation
        data = {"train": [], "validation": []}
        fraction_valid = 0.0
        if fraction_valid > 0 and len(self.data_list) * fraction_valid > 1:
            # How many total snapshots to extract
            n_full = len(self.data_list)
            n_valid = int(np.floor(fraction_valid * n_full))

            # Get validation set indices
            idx_valid = np.sort(
                np.random.choice(n_full, n_valid, replace=False)
            )

            # Get training set indices
            idx_train = np.array(
                list(set(list(range(n_full))) - set(list(idx_valid)))
            )

            # Train/validation split
            data["train"] = [self.data_list[i] for i in idx_train]
            data["validation"] = [self.data_list[i] for i in idx_valid]
        else:
            data["train"] = self.data_list
            data["validation"] = [{}]

        if RANK == 0:
            log.info(f"Number of training snapshots: {len(data['train'])}")
        if RANK == 0:
            log.info(f"Number of validation snapshots: {0}")

        # Compute statistics for normalization
        stats = {"x": []}
        if "stats" not in self.data.keys():
            if os.path.exists(data_dir + f"/data_stats.npz"):
                n_features = self.cfg.input_node_features
                if RANK == 0:
                    npzfile = np.load(data_dir + f"/data_stats.npz")
                    stats_arr_x = np.stack([
                        npzfile["x_mean"][0, :n_features],
                        npzfile["x_std"][0, :n_features],
                    ])
                else:
                    stats_arr_x = np.zeros((2, n_features), dtype=np.float32)
                COMM.Bcast(stats_arr_x, root=0)
                stats["x"] = [stats_arr_x[0], stats_arr_x[1]]
                if RANK == 0:
                    log.info(
                        f"Read training data statistics from {data_dir}/data_stats.npz"
                    )
            else:
                x_mean, x_std = self.compute_statistics(data["train"], "x")
                if RANK == 0 and not self.cfg.online:
                    np.savez(
                        data_dir + f"/data_stats.npz",
                        x_mean=x_mean,
                        x_std=x_std,
                    )
                stats["x"] = [x_mean, x_std]
                if RANK == 0:
                    log.info(
                        f"Computed training data statistics for each node feature"
                    )
        return data, stats

    def setup_graph_data(self):
        """
        Generate the PyTorch Geometric Dataset
        """
        if RANK == 0:
            log.info("In setup_graph_data...")

        device_for_loading = "cpu"

        # Get dictionary
        reduced_graph_dict = self.data_reduced.to_dict()

        # Create training dataset -- only 1 snapshot for demo
        data_graph = Data()
        for key in reduced_graph_dict.keys():
            data_graph[key] = reduced_graph_dict[key]
        if self.cfg.consistency:
            n_nodes_halo = self.data_reduced.n_nodes_halo
            n_features_pos = self.data_reduced.pos.shape[1]
            pos_halo = torch.zeros(
                (n_nodes_halo, n_features_pos), dtype=self.torch_dtype
            )
            data_graph.pos = torch.cat((data_graph.pos, pos_halo), dim=0)
            # cond_node_features is already (n_local + n_halo) with halo slots
            # populated by setup_halo's one-shot MPI exchange — no zero-pad here.

        # Populate edge_attrs
        cart = torch_geometric.transforms.Cartesian(
            norm=False, max_value=None, cat=False
        )
        dist = torch_geometric.transforms.Distance(
            norm=False, max_value=None, cat=True
        )
        data_graph = cart(data_graph)  # adds cartesian/component-wise distance
        data_graph = dist(data_graph)  # adds euclidean distance
        data_graph = data_graph.to(device_for_loading)

        # Normalize edge_attrs by length of the longest edge
        distance = data_graph.edge_attr[:, -1]
        distance_max_ = distance.max().to(self.device)
        distance_max = distnn.all_reduce(
            distance_max_, op=distnn.ReduceOp.MAX
        ).to(device_for_loading)
        data_graph.edge_attr = (data_graph.edge_attr / distance_max).to(
            self.torch_dtype
        )

        if RANK == 0:
            log.info(f"{data_graph}")

        self.data["graph"] = data_graph

    def setup_train_data(self):
        """
        Load the training data and prepare the data loader
        """
        if RANK == 0:
            log.info("In setup_train_data...")

        data_dir = self.cfg.gnn_outputs_path
        data, stats = self.load_field_data(data_dir)

        # Materialize per-feature stats as torch tensors in the model dtype.
        # If we leave them as numpy (their on-disk form), the per-snapshot
        # arithmetic below would be `bf16_torch_tensor - numpy_fp32_array`,
        # which forces torch to convert each numpy operand via the buffer
        # protocol -- and IPEX prints "calling in ipex numpy ... bfloat16"
        # for every single op (~160 warnings per rank with 80 snapshots).
        # Casting once here keeps the loop fully in torch land.
        stats_x_mean_t = torch.as_tensor(stats["x"][0], dtype=self.torch_dtype)
        stats_x_std_t = torch.as_tensor(stats["x"][1], dtype=self.torch_dtype)

        # ~~~~ Populate the data loader
        # No need for distributed sampler -- create standard dataset loader
        # We can use the standard pytorch dataloader on (x,y)
        train_data_scaled = []
        for item in data["train"]:
            train_data_scaled.append(
                Data(
                    x=(
                        (
                            item["x"][:, : self.cfg.input_node_features]
                            - stats_x_mean_t
                        )
                        / (stats_x_std_t + SMALL)
                    ).to(self.torch_dtype)
                )
            )
        # Explicit generator so the shuffle order is deterministic and
        # independent of any other torch.randn / dropout / etc. that may
        # consume the global RNG between epochs.
        train_loader = DataLoader(
            train_data_scaled,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(self.cfg.seed),
        )
        if RANK == 0:
            train_loader_example = train_loader.dataset[0]
            log.info(f"shape of x: {train_loader_example.x.shape}")

        val_data_scaled = data["validation"].copy()
        if val_data_scaled[0]:
            for item in val_data_scaled:
                val_data_scaled.append(
                    Data(
                        x=(
                            (item["x"] - stats_x_mean_t)
                            / (stats_x_std_t + SMALL)
                        ).to(self.torch_dtype)
                    )
                )
        valid_loader = DataLoader(
            val_data_scaled, batch_size=self.cfg.val_batch_size, shuffle=False
        )

        self.data["train"] = {
            "loader": train_loader,
            "example": data["train"][0],
        }
        self.data["validation"] = {
            "loader": valid_loader,
            "example": data["validation"][0],
        }
        self.data["stats"] = {
            "x_mean": stats["x"][0],
            "x_std": stats["x"][1],
        }

    def load_stats(self):
        """
        Load the normalizationstatistics for the training data
        """
        data_dir = self.cfg.gnn_outputs_path
        stats = {"x": []}
        if os.path.exists(data_dir + f"/data_stats.npz"):
            n_features = self.cfg.input_node_features
            if RANK == 0:
                npzfile = np.load(data_dir + f"/data_stats.npz")
                stats_arr_x = np.stack([
                    npzfile["x_mean"][0, :n_features],
                    npzfile["x_std"][0, :n_features],
                ])
            else:
                stats_arr_x = np.zeros((2, n_features), dtype=np.float32)
            COMM.Bcast(stats_arr_x, root=0)
            stats["x"] = [stats_arr_x[0], stats_arr_x[1]]
        self.data["stats"] = {
            "x_mean": stats["x"][0],
            "x_std": stats["x"][1],
        }

    def setup_timers(self, n_record: int) -> dict:
        timers = {}
        timers["forwardPass"] = np.zeros(n_record)
        timers["backwardPass"] = np.zeros(n_record)
        timers["loss"] = np.zeros(n_record)
        timers["optimizerStep"] = np.zeros(n_record)
        timers["dataTransfer"] = np.zeros(n_record)
        timers["bufferInit"] = np.zeros(n_record)
        timers["collectives"] = np.zeros(n_record)
        timers["dataTransfer"] = np.zeros(n_record)
        return timers

    def setup_online_timers(self) -> dict:
        timers = {}
        timers["metaData"] = []
        timers["trainDataTime"] = []
        timers["trainDataSize"] = []
        timers["trainDataThroughput"] = []
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
                t_avg = t_avg / SIZE
                COMM.Allreduce(t_data, t_min, op=MPI.MIN)
                COMM.Allreduce(t_data, t_max, op=MPI.MAX)
            else:
                t_avg = t_data
                t_min = t_data
                t_max = t_data
            self.timers_avg[key][i] = (
                t_avg  # metric_average(torch.tensor( self.timers[key][i] )).item()
            )
            self.timers_min[key][i] = (
                t_min  # metric_min(torch.tensor( self.timers[key][i] )).item()
            )
            self.timers_max[key][i] = (
                t_max  # metric_max(torch.tensor( self.timers[key][i] )).item()
            )
            # if RANK == 0:
            #    log.info(f"t_{key} [min,max,avg] = [{self.timers_min[key][i]},{self.timers_max[key][i]},{self.timers_avg[key][i]}]")
        return

    def collect_timer_stats(self) -> None:
        self.timer_stats = {}
        for key, val in self.timers.items():
            times = np.delete(val, [0, 1])
            times = times[times != 0]
            collected_arr = np.zeros((times.size * SIZE))
            COMM.Gather(times, collected_arr, root=0)
            avg = np.mean(collected_arr)
            std = np.std(collected_arr)
            minn = np.amin(collected_arr)
            min_loc = [minn, 0]
            maxx = np.amax(collected_arr)
            max_loc = [maxx, 0]
            summ = np.sum(collected_arr)
            stats = {
                "avg": avg,
                "std": std,
                "sum": summ,
                "min": [min_loc[0], min_loc[1]],
                "max": [max_loc[0], max_loc[1]],
            }
            self.timer_stats[key] = stats

    def print_timer_stats(self) -> None:
        for key, val in self.timer_stats.items():
            stats_string = (
                f": min = {val['min'][0]:>6e} , "
                + f"max = {val['max'][0]:>6e} , "
                + f"avg = {val['avg']:>6e} , "
                + f"std = {val['std']:>6e} "
            )
            log.info(f"{key} [s] " + stats_string)

    def synchronize(self):
        if WITH_CUDA:
            torch.cuda.synchronize()
        if WITH_XPU:
            torch.xpu.synchronize()

    def get_posterior_mean_and_variance_from_output(
        self,
        model_output: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        field_r: torch.Tensor,
        r: torch.Tensor,
        batch: torch.Tensor = None,
        diffusion_process: DiffusionProcess = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the posterior mean and variance from the model output.

        Adapthed from https://github.com/tum-pbs/dgn4cfd/blob/main/dgn4cfd/nn/diffusion/diffusion_model.py
        Supports both epsilon and x0 prediction types.
        """
        if batch is None:
            batch = torch.zeros(
                field_r.size(0), dtype=torch.long, device=self.device
            )

        dp = (
            self.diffusion_process
            if diffusion_process is None
            else diffusion_process
        )

        # Extract model prediction and learnable variance
        if self.cfg.learnable_variance:
            if isinstance(model_output, tuple):
                model_pred, v = model_output
            else:
                raise ValueError(
                    "Expected tuple output when learnable_variance=True"
                )
            v = (v + 1) / 2
            # For min_log we use the clipped posterior variance to avoid nan values in the backward pass
            min_log = dp.get_index_from_list(
                dp.posterior_log_variance_clipped, batch, r
            )
            max_log = torch.log(dp.get_index_from_list(dp.betas, batch, r))
            log_variance = v * max_log + (1 - v) * min_log
            variance = torch.exp(log_variance)
        else:
            if isinstance(model_output, tuple):
                model_pred = model_output[0]
            else:
                model_pred = model_output
            variance = dp.get_index_from_list(dp.posterior_variance, batch, r)

        # Compute posterior mean based on prediction type
        # All prediction types ultimately need x_0 to compute the posterior mean:
        #   posterior_mean = coef1 * x_0 + coef2 * x_t
        if self.cfg.prediction_type == "x0":
            # model_pred is the predicted clean field x_0 directly
            x_0_pred = model_pred
        elif self.cfg.prediction_type == "v":
            # model_pred is the predicted v; recover x_0 from v
            x_0_pred = dp.predict_x0_from_v(field_r, model_pred, r, batch)
        else:
            # epsilon prediction (default): model_pred is predicted noise eps; recover x_0
            x_0_pred = dp.predict_x0_from_noise(field_r, model_pred, r, batch)

        # Clip x_0 prediction to prevent error accumulation during reverse sampling.
        # For z-normalized data, values beyond ±5 sigma are extremely unlikely.
        x_0_pred = x_0_pred.clamp(-5.0, 5.0)

        posterior_mean_coef1 = dp.get_index_from_list(
            dp.posterior_mean_coef1, batch, r
        )
        posterior_mean_coef2 = dp.get_index_from_list(
            dp.posterior_mean_coef2, batch, r
        )
        mean = posterior_mean_coef1 * x_0_pred + posterior_mean_coef2 * field_r
        return mean, variance

    def train_step(self, data: Data) -> Tensor:
        loss = torch.tensor([0.0])
        graph = self.data["graph"]
        tic = time.time()
        if WITH_CUDA or WITH_XPU:
            data = data.to(self.device)
            graph.edge_index = graph.edge_index.to(self.device)
            graph.edge_attr = graph.edge_attr.to(self.device)
            graph.batch = (
                graph.batch.to(self.device) if graph.batch is not None else None
            )
            graph.halo_info = graph.halo_info.to(self.device)
            graph.edge_weight = graph.edge_weight.to(self.device)
            graph.node_degree = graph.node_degree.to(self.device)
            graph.effective_nodes = graph.effective_nodes.to(self.device)
            if self.cfg.cond_node_features:
                graph.cond_node_features = graph.cond_node_features.to(
                    self.device
                )
            if self.cfg.model_name == "graph_transformer":
                graph.pos = graph.pos.to(self.device)
                graph.global_ids = graph.global_ids.to(self.device)
                graph.pos_min = graph.pos_min.to(self.device)
                graph.pos_max = graph.pos_max.to(self.device)
                self.idx_reduced2full = self.idx_reduced2full.to(self.device)
                self.idx_full2reduced = self.idx_full2reduced.to(self.device)
            loss = loss.to(self.device)
        if self.cfg.timers:
            self.update_timer(
                "dataTransfer", self.timer_step, time.time() - tic
            )

        if (
            self.cfg.postprocess
            and self.iteration == 0
            and self.cfg.cond_node_features
        ):
            n_local = int(graph.n_nodes_local)
            postprocess.plot_2d_field(
                COMM,
                graph.pos[:n_local].to(torch.float32).numpy(),
                graph.cond_node_features[:n_local].cpu().to(torch.float32).numpy(),
                f"cond_node_features.png",
            )
            COMM.Barrier()

        self.s_optimizer.zero_grad()

        # re-allocate send buffer
        tic = time.time()
        if self.cfg.halo_swap_mode != "none":
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
        if self.cfg.timers:
            self.update_timer("bufferInit", self.timer_step, time.time() - tic)

        # Sample a batch of random diffusion steps (all ranks need same sample)
        batch_size = torch.max(data.batch) + 1
        r, importance_weights = self.step_sampler.sample(batch_size=batch_size)
        # Broadcast r and importance_weights from rank 0 for cross-rank consistency.
        # Same-seeded RNGs are not enough: the first per-rank-sized random op (the
        # noise tensor below) advances each rank's RNG state by a different number
        # of draws, so subsequent calls to step_sampler.sample would pick a
        # different r on each rank from iteration 1 onward.
        if SIZE > 1 and self.cfg.consistency:
            dist.broadcast(r, src=0)
            dist.broadcast(importance_weights, src=0)
        if self.cfg.verbose and RANK == 0:
            log.info(f"Sampled diffusion steps: {r.cpu().numpy().tolist()}")

        # Diffuse the solution/target field. Noise is drawn by hashing each
        # node's global id, so physically-coincident nodes on different ranks
        # receive identical noise without any collective and the per-iteration
        # loss matches what a single-rank run would compute.
        BC_mask = None  # no BCs for now
        field_start = data.x[:, : self.cfg.input_node_features]
        noise = self._consistent_noise(
            batch_size=int(batch_size),
            n_features=self.cfg.input_node_features,
            salt=0,
        )
        if BC_mask is not None:
            noise = noise * (~BC_mask)
        field_r, noise, snr = self.diffusion_process.forward(
            field_start,
            r,
            batch=data.batch,
            dirichlet_mask=BC_mask,
            noise=noise,
        )
        if self.cfg.postprocess and self.iteration % 100 == 0:
            n_local = int(graph.n_nodes_local)
            pos_owned = graph.pos[:n_local].to(torch.float32).numpy()
            # data.batch == 0 selects the first batch element's full
            # (n_local + n_halo) rows; further [:n_local] drops the halo.
            postprocess.plot_2d_field(
                COMM,
                pos_owned,
                field_r[data.batch == 0][:n_local].cpu().to(torch.float32).numpy(),
                f"field_r_r{r[0]}_iter{self.iteration}.png",
            )
            postprocess.plot_2d_field(
                COMM,
                pos_owned,
                data
                .x[data.batch == 0, : self.cfg.input_node_features][:n_local]
                .cpu()
                .to(torch.float32)
                .numpy(),
                f"data_x_r{r[0]}_iter{self.iteration}.png",
            )
            COMM.Barrier()

        # Prediction
        tic = time.time()
        debug_perlayer: Optional[Dict[str, Any]] = (
            {} if self.iteration == 0 else None
        )
        if self.cfg.model_name == "gnn":
            model_pred, model_var = self.model(
                field_r=field_r,
                r=r,
                edge_index=graph.edge_index,
                edge_attr=graph.edge_attr,
                edge_weight=graph.edge_weight,
                halo_info=graph.halo_info,
                mask_send=self.mask_send,
                mask_recv=self.mask_recv,
                buffer_send=self.buffer_send,
                buffer_recv=self.buffer_recv,
                neighboring_procs=self.neighboring_procs,
                SIZE=SIZE,
                cond_node_features=graph.cond_node_features
                if self.cfg.cond_node_features
                else None,
                batch=data.batch,
                debug_dump=debug_perlayer,
            )
        elif self.cfg.model_name == "graph_transformer":
            model_pred, model_var = self.model(
                field_r=field_r,
                r=r,
                pos=graph.pos,
                pos_min=graph.pos_min,
                pos_max=graph.pos_max,
                index=graph.global_ids.reshape(-1),
                mask_send=self.mask_send,
                mask_recv=self.mask_recv,
                buffer_send=self.buffer_send,
                buffer_recv=self.buffer_recv,
                halo_info=graph.halo_info,
                idx_reduced2full=self.idx_reduced2full,
                idx_full2reduced=self.idx_full2reduced,
                neighboring_procs=self.neighboring_procs,
                SIZE=SIZE,
                cond_node_features=graph.cond_node_features
                if self.cfg.cond_node_features
                else None,
                batch=data.batch,
                debug_dump=debug_perlayer,
            )
        else:
            raise ValueError("Unknown model name: %s" % self.cfg.model_name)
        if self.cfg.timers:
            self.update_timer("forwardPass", self.timer_step, time.time() - tic)

        # Accumulate loss
        tic = time.time()
        # MSE loss: target depends on prediction type
        if self.cfg.prediction_type == "x0":
            # x0-prediction: model predicts the clean field directly
            mse_target = data.x[:, : self.cfg.input_node_features]
        elif self.cfg.prediction_type == "v":
            # v-prediction: model predicts v = sqrt(alpha_bar)*eps - sqrt(1-alpha_bar)*x_0
            mse_target = self.diffusion_process.get_v_target(
                data.x[:, : self.cfg.input_node_features], noise, r, data.batch
            )
        else:
            # epsilon-prediction (default): model predicts the noise
            mse_target = noise
        if self.cfg.postprocess and self.iteration % 100 == 0:
            n_local = int(graph.n_nodes_local)
            pos_owned = graph.pos[:n_local].to(torch.float32).numpy()
            postprocess.plot_2d_field(
                COMM,
                pos_owned,
                model_pred[data.batch == 0][:n_local].detach().cpu().to(torch.float32).numpy(),
                f"model_pred_r{r[0]}_iter{self.iteration}.png",
            )
            postprocess.plot_2d_field(
                COMM,
                pos_owned,
                mse_target[data.batch == 0][:n_local].cpu().to(torch.float32).numpy(),
                f"target_r{r[0]}_iter{self.iteration}.png",
            )
            COMM.Barrier()

        if SIZE == 1 or not self.cfg.consistency:
            mse_term = batch_wise_mean(
                (model_pred - mse_target) ** 2, data.batch
            )  # Dimension (batch_size)
            loss = mse_term
            if self.cfg.learnable_variance:
                # Hybrid loss function for diffusion models from the paper
                # Improved Denoising Diffusion Probabilistic Models (https://arxiv.org/abs/2102.09672).
                # Adapted from https://github.com/tum-pbs/dgn4cfd/blob/main/dgn4cfd/nn/losses.py
                lambda_vlb = 0.001
                true_posterior_mean, true_posterior_variance = (
                    self.diffusion_process.get_posterior_mean_and_variance(
                        data.x[:, : self.cfg.input_node_features],
                        field_r,
                        data.batch,
                        r,
                    )
                )
                model_posterior_mean, model_posterior_variance = (
                    self.get_posterior_mean_and_variance_from_output(
                        (model_pred.detach(), model_var),
                        field_r,
                        r,
                        batch=data.batch,
                    )
                )
                vlb_term = vlb_loss(
                    data.x[:, : self.cfg.input_node_features],
                    (true_posterior_mean, true_posterior_variance),
                    (model_posterior_mean, model_posterior_variance),
                    data.batch,
                    r,
                )  # Dimension (batch_size)
                vlb_term = vlb_term * lambda_vlb
                loss = loss + vlb_term  # Dimension (batch_size)
            # Log unweighted per-step losses (always, regardless of learnable_variance)
            if self.cfg.verbose and RANK == 0:
                mse_log = mse_term.detach().cpu().to(torch.float32)
                log.info(
                    f"MSE loss term: {mse_log.numpy().tolist()}, mean = {mse_log.mean().numpy().tolist()}"
                )
                if self.cfg.learnable_variance:
                    vlb_log = vlb_term.detach().cpu().to(torch.float32)
                    log.info(
                        f"VLB loss term: {vlb_log.numpy().tolist()}, mean = {vlb_log.mean().numpy().tolist()}"
                    )
        else:  # custom consistent loss
            if self.cfg.learnable_variance:
                if RANK == 0:
                    log.error(
                        "Custom consistent loss not yet implemented for VLB term"
                    )
                COMM.Abort(1)
            n_output_features = model_pred.shape[1]
            mse_term = torch.zeros(
                batch_size, dtype=self.torch_dtype, device=self.device
            )
            for batch_idx in range(batch_size):
                model_pred_local = model_pred[data.batch == batch_idx]
                mse_target_local = mse_target[data.batch == batch_idx]
                squared_errors_local = torch.pow(
                    model_pred_local[: graph.n_nodes_local]
                    - mse_target_local[: graph.n_nodes_local],
                    2,
                )
                squared_errors_local = squared_errors_local / graph.node_degree[
                    : graph.n_nodes_local
                ].unsqueeze(-1)
                sum_squared_errors_local = squared_errors_local.sum()
                sum_squared_errors = distnn.all_reduce(sum_squared_errors_local)
                mse_term[batch_idx] = (
                    1.0 / (graph.effective_nodes * n_output_features)
                ) * sum_squared_errors
            if self.cfg.verbose and RANK == 0:
                mse_log = mse_term.detach().cpu().to(torch.float32)
                log.info(
                    f"[RANK {RANK}] MSE loss term: {mse_log.numpy().tolist()}, mean = {mse_log.mean().numpy().tolist()}"
                )
            loss = mse_term  # Dimension (batch_size)

        # Apply loss weighting
        # 1) Min-SNR weights (only on MSE, not VLB — VLB has correct per-step weighting from the KL)
        # 2) Importance weights from the step sampler (corrects for non-uniform sampling)
        if self.cfg.loss_weighting == "min_snr":
            # Min-SNR-gamma weighting from "Efficient Diffusion Training via Min-SNR Weighting Strategy"
            # (Hang et al., 2023, https://arxiv.org/abs/2303.09556)
            # For epsilon-prediction: w(t) = min(SNR(t), gamma)
            # For x0-prediction:      w(t) = min(SNR(t), gamma) / SNR(t)
            # For v-prediction:       w(t) = min(SNR(t), gamma) / (SNR(t) + 1)
            clamped_snr = torch.clamp(
                snr, max=self.cfg.min_snr_gamma
            )  # Dimension (batch_size)
            if self.cfg.prediction_type == "x0":
                min_snr_weights = clamped_snr / snr
            elif self.cfg.prediction_type == "v":
                min_snr_weights = clamped_snr / (snr + 1)
            else:
                min_snr_weights = clamped_snr
            # Apply min-SNR only to MSE term, then add VLB separately
            weighted_mse = mse_term * min_snr_weights
            if self.cfg.learnable_variance:
                loss = (weighted_mse + vlb_term).mean()
            else:
                loss = weighted_mse.mean()
            # Log weighted per-step losses (actual gradient contribution)
            if self.cfg.verbose and RANK == 0:
                w_log = weighted_mse.detach().cpu().to(torch.float32)
                log.info(
                    f"[RANK {RANK}] Weighted MSE loss: {w_log.numpy().tolist()}, mean = {w_log.mean().numpy().tolist()}"
                )
        else:
            # Uniform weighting (default)
            loss = loss.mean()

        if self.cfg.timers:
            self.update_timer("loss", self.timer_step, time.time() - tic)

        tic = time.time()
        loss.backward()
        if self.cfg.timers:
            self.update_timer(
                "backwardPass", self.timer_step, time.time() - tic
            )

        tic = time.time()
        self.s_optimizer.step_and_update_lr()
        if self.cfg.timers:
            self.update_timer(
                "optimizerStep", self.timer_step, time.time() - tic
            )

        # Update timers
        self.synchronize()
        if self.cfg.timers:
            if self.timer_step < self.timer_step_max - 1:
                self.update_timer_stats()
                self.timer_step += 1
        return loss

    @torch.no_grad()
    def sample(self, steps: list[int] = None) -> Tensor:
        self.model.eval()

        # Update the counter for the random number seed
        self.sample_counter += 1
        sc = int(self.sample_counter)

        # Assert step list is all integers and is sorted
        if steps is not None:
            if RANK == 0:
                log.error("Passing a list of steps is not supported yet")
            COMM.Abort(1)
        else:
            steps = list(range(self.cfg.num_diffusion_steps))

        # Offload graph to device first so _consistent_noise can read
        # pos_for_noise from the device.
        n_nodes = self.data["graph"].pos_orig.size(0)
        n_nodes_halo = (
            self.data["graph"].n_nodes_halo
            if self.cfg.consistency and SIZE > 1
            else 0
        )
        n_nodes_total = n_nodes + n_nodes_halo
        batch = torch.zeros(n_nodes_total, dtype=torch.long)

        tic = time.time()
        if WITH_CUDA or WITH_XPU:
            self.data["graph"].edge_index = self.data["graph"].edge_index.to(
                self.device
            )
            self.data["graph"].edge_attr = self.data["graph"].edge_attr.to(
                self.device
            )
            self.data["graph"].batch = batch.to(self.device)
            self.data["graph"].halo_info = self.data["graph"].halo_info.to(
                self.device
            )
            self.data["graph"].edge_weight = self.data["graph"].edge_weight.to(
                self.device
            )
            self.data["graph"].node_degree = self.data["graph"].node_degree.to(
                self.device
            )
            self.data["graph"].effective_nodes = self.data[
                "graph"
            ].effective_nodes.to(self.device)
            if self.cfg.cond_node_features:
                self.data["graph"].cond_node_features = self.data[
                    "graph"
                ].cond_node_features.to(self.device)
            if self.cfg.model_name == "graph_transformer":
                self.data["graph"].pos = self.data["graph"].pos.to(self.device)
                self.data["graph"].global_ids = self.data[
                    "graph"
                ].global_ids.to(self.device)
                self.data["graph"].pos_min = self.data["graph"].pos_min.to(
                    self.device
                )
                self.data["graph"].pos_max = self.data["graph"].pos_max.to(
                    self.device
                )
                self.idx_reduced2full = self.idx_reduced2full.to(self.device)
                self.idx_full2reduced = self.idx_full2reduced.to(self.device)
        if self.cfg.timers:
            self.update_timer(
                "dataTransfer", self.timer_step, time.time() - tic
            )

        # Initialize sample field via rank-consistent noise so that on every
        # rank the tier-1/tier-2/tier-3 slots receive identical values at
        # physically-coincident nodes — the reverse diffusion is then a pure
        # function of model weights and the initial seed.
        diff_process = self.diffusion_process
        field_r = self._consistent_noise(
            batch_size=1,
            n_features=self.cfg.input_node_features,
            salt=-1,  # distinct from any per-step salt below
            extra_counter=sc,
        )

        # Dedicated halo-swap buffers sized for field_r (input_node_features
        # wide, vs the mlp_hidden_channels-wide self.buffer_* used inside
        # gnn.py). Only needed if we will actually do a swap below.
        field_r_buf_send = None
        field_r_buf_recv = None
        if (
            SIZE > 1
            and self.cfg.consistency
            and self.cfg.halo_swap_mode != "none"
        ):
            field_r_buf_send = [
                torch.empty(0, device=DEVICE, dtype=self.torch_dtype)
            ] * SIZE
            field_r_buf_recv = [
                torch.empty(0, device=DEVICE, dtype=self.torch_dtype)
            ] * SIZE
            for i in self.neighboring_procs:
                n_swap = int(self.n_nodes_to_exchange[i])
                field_r_buf_send[i] = torch.empty(
                    [n_swap, self.cfg.input_node_features],
                    dtype=self.torch_dtype,
                    device=DEVICE,
                )
                field_r_buf_recv[i] = torch.empty(
                    [n_swap, self.cfg.input_node_features],
                    dtype=self.torch_dtype,
                    device=DEVICE,
                )

        # re-allocate send buffer
        tic = time.time()
        if self.cfg.halo_swap_mode != "none":
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
        if self.cfg.timers:
            self.update_timer("bufferInit", self.timer_step, time.time() - tic)

        # Prediction (de-noise step by step)
        for step in diff_process.steps[::-1]:
            if RANK == 0 and self.cfg.verbose:
                log.info(f"Performing de-noise step {step}")
            r = torch.tensor([step], device=self.device)
            tic = time.time()
            if self.cfg.model_name == "gnn":
                model_pred, model_var = self.model(
                    field_r=field_r,
                    r=r,
                    edge_index=self.data["graph"].edge_index,
                    edge_attr=self.data["graph"].edge_attr,
                    edge_weight=self.data["graph"].edge_weight,
                    halo_info=self.data["graph"].halo_info,
                    mask_send=self.mask_send,
                    mask_recv=self.mask_recv,
                    buffer_send=self.buffer_send,
                    buffer_recv=self.buffer_recv,
                    neighboring_procs=self.neighboring_procs,
                    SIZE=SIZE,
                    cond_node_features=self.data["graph"].cond_node_features
                    if self.cfg.cond_node_features
                    else None,
                    batch=self.data["graph"].batch,
                )
            elif self.cfg.model_name == "graph_transformer":
                model_pred, model_var = self.model(
                    field_r=field_r,
                    r=r,
                    pos=self.data["graph"].pos,
                    pos_min=self.data["graph"].pos_min,
                    pos_max=self.data["graph"].pos_max,
                    index=self.data["graph"].global_ids.reshape(-1),
                    mask_send=self.mask_send,
                    mask_recv=self.mask_recv,
                    buffer_send=self.buffer_send,
                    buffer_recv=self.buffer_recv,
                    halo_info=self.data["graph"].halo_info,
                    idx_reduced2full=self.idx_reduced2full,
                    idx_full2reduced=self.idx_full2reduced,
                    neighboring_procs=self.neighboring_procs,
                    SIZE=SIZE,
                    cond_node_features=self.data["graph"].cond_node_features
                    if self.cfg.cond_node_features
                    else None,
                    batch=self.data["graph"].batch,
                )
            else:
                raise ValueError("Unknown model name: %s" % self.cfg.model_name)
            if self.cfg.timers:
                self.update_timer(
                    "forwardPass", self.timer_step, time.time() - tic
                )

            # Get the posterior mean and variance from the model output
            # get_posterior_mean_and_variance_from_output handles both epsilon and x0 prediction types
            model_posterior_mean, model_posterior_variance = (
                self.get_posterior_mean_and_variance_from_output(
                    (model_pred.detach(), model_var), field_r, r
                )
            )

            # Update field_r: add noise at all steps except the final one (step 0)
            if step > 0:
                # Salt with step so each reverse-diffusion step gets an
                # independent rank-consistent draw.
                gaussian_noise = self._consistent_noise(
                    batch_size=1,
                    n_features=model_posterior_mean.shape[1],
                    salt=step,
                    extra_counter=sc,
                )
                field_r = (
                    model_posterior_mean
                    + torch.sqrt(model_posterior_variance) * gaussian_noise
                )
            else:
                field_r = model_posterior_mean

            # Halo-sync field_r so every rank's tier-3 slot carries its
            # owner's value before the next reverse step. The node decoder is
            # *not* halo-synced inside gnn.py (only the per-MP edge_agg is),
            # so model_pred (and hence field_r) drifts by O(1e-2) at halo
            # slots. During training that drift never feeds back — each step
            # is independent — but during sampling we re-consume the halo
            # state every reverse step and the drift compounds into a visible
            # seam at partition boundaries. _consistent_noise already gives us
            # rank-identical noise at coincident nodes, so once we sync the
            # mean the sum is consistent too.
            if (
                SIZE > 1
                and self.cfg.consistency
                and self.cfg.halo_swap_mode != "none"
            ):
                field_r = self.halo_swap(
                    field_r, field_r_buf_send, field_r_buf_recv
                )

        # Update timers
        self.synchronize()
        if self.cfg.timers:
            if self.timer_step < self.timer_step_max - 1:
                self.update_timer_stats()
                self.timer_step += 1

        return field_r

    def writeGraphStatistics(self):
        if RANK == 0:
            log.info(f"In writeGraphStatistics")
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
        n_nodes_halo = (
            self.data_reduced.n_nodes_halo if self.cfg.consistency else 0
        )
        n_edges = self.data_reduced.edge_index.shape[1]

        if self.cfg.verbose:
            log.info(
                f"[RANK {RANK}] -- number of local nodes: {n_nodes_local}, number of halo nodes: {n_nodes_halo}, number of edges: {n_edges}"
            )
        else:
            if RANK == 0:
                log.info(
                    f"[RANK {RANK}] -- number of local nodes: {n_nodes_local}, number of halo nodes: {n_nodes_halo}, number of edges: {n_edges}"
                )

        a = {}
        a["n_nodes_local"] = n_nodes_local
        a["n_nodes_halo"] = n_nodes_halo
        a["n_edges"] = n_edges
        torch.save(a, savepath + "/%s.tar" % (model.get_save_header()))

    def grad_norm(self):
        """Do some analytics on the gradients"""
        grads = [
            param.grad.detach().flatten()
            for param in self.model.parameters()
            if param.grad is not None
        ]
        gradnorm = torch.cat(grads).norm()
        COMM.Barrier()
        return gradnorm
