import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import sys
from typing import Optional
from time import perf_counter

import torch

from gnn import MLP

log = logging.getLogger(__name__)

class MLPReprod():
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

        self.setup_torch()
        self.setup_data()
        self.build_model()

        self.loss_fn = torch.nn.MSELoss()
        self.loss_fn.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
    
    def setup_torch(self):
        # Random seeds
        torch.manual_seed(self.cfg.seed)

        # Find device type
        self.with_cuda = torch.cuda.is_available()
        self.with_xpu = torch.xpu.is_available()
        if self.with_cuda:
            self.device = torch.device('cuda')
            self.n_devices = torch.cuda.device_count()
            self.device_id = 0
        elif self.with_xpu:
            self.device = torch.device('xpu')
            self.n_devices = torch.xpu.device_count()
            self.device_id = 0
        else:
            self.device = torch.device('cpu')
            self.device_id = 'cpu'

        # Set device and intra-op threads
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

    def setup_data(self):
        self.num_nodes = 160_000
        self.num_edges = 1_000_000

    def build_model(self, layer_norm: Optional[bool] = False):
        self.model = MLP(
                input_channels = self.cfg.hidden_channels,
                hidden_channels = [self.cfg.hidden_channels]*(self.cfg.n_mlp_hidden_layers+1),
                output_channels = self.cfg.hidden_channels,
                activation_layer = torch.nn.ELU(),
                norm_layer = torch.nn.LayerNorm(self.cfg.hidden_channels) if layer_norm else None
        )
        self.model.to(self.torch_dtype)
        self.model.to(self.device)

    def synchronize(self):
        if self.with_cuda:
            torch.cuda.synchronize()
        elif self.with_xpu:
            torch.xpu.synchronize()
        else:
            pass

    def train_step(self, x: torch.Tensor, y: torch.Tensor):
        self.model.train()

        self.optimizer.zero_grad()
        pred = self.model(x)
        loss = self.loss_fn(pred, y)
        loss.backward()
        self.optimizer.step()
        self.synchronize()

    def train(self, graph_component: Optional[str] = 'nodes'):
        if graph_component == 'nodes':
            N = self.num_nodes
        elif graph_component == 'edges':
            N = self.num_edges
        else:
            sys.exit('Invalid graph component')

        # Warmup 
        for _ in range(10):
            x = torch.randn(N, self.cfg.hidden_channels, dtype=self.torch_dtype, device=self.device)
            y = torch.randn(N, self.cfg.hidden_channels, dtype=self.torch_dtype, device=self.device)
            self.train_step(x, y)

        # Performance test
        times = []
        for _ in range(self.cfg.phase1_steps):
            x = torch.randn(N, self.cfg.hidden_channels, dtype=self.torch_dtype, device=self.device)
            y = torch.randn(N, self.cfg.hidden_channels, dtype=self.torch_dtype, device=self.device)
            
            tic = perf_counter()
            self.train_step(x, y)
            times.append(perf_counter() - tic)

        compute_time = sum(times) / len(times)
        mlp_flops = 2 * self.cfg.hidden_channels**2 * N * (self.cfg.n_mlp_hidden_layers + 2)
        flops_per_step = 3 * mlp_flops # backward pass is approx. 2x forward pass flops
        ai = (N * self.cfg.hidden_channels**2) \
                / (2 * N * self.cfg.hidden_channels + self.cfg.hidden_channels**2)
        sizeof_dtype = torch.tensor([], dtype=self.torch_dtype).element_size() # in bytes

        log.info(f"Training performance metrics (per step) with N={N}:")
        log.info(f"\tCompute time: {compute_time:.4f} seconds")
        log.info(f"\tFLOPS: {flops_per_step/1e9:.4f} GFLOPS")
        log.info(f"\tAI: {ai:.4f}")
        log.info(f"\tMemory traffic: {flops_per_step / ai * sizeof_dtype / 1e9:.4f} GB")
        log.info(f"\tFLOPS/s: {flops_per_step / compute_time / 1e12:.4f} TFLOPS/s")
        log.info(f"\tMemory bandwidth: {flops_per_step / ai * sizeof_dtype / 1e9 / compute_time:.4f} GB/s")
        log.info("\n")

    @torch.no_grad()
    def test_step(self, x: torch.Tensor):
        self.model.eval()

        pred = self.model(x)
        self.synchronize()

    def test(self, graph_component: Optional[str] = 'nodes'):
        if graph_component == 'nodes':
            N = self.num_nodes
        elif graph_component == 'edges':
            N = self.num_edges
        else:
            sys.exit('Invalid graph component')

        # Warmup 
        for _ in range(10):
            x = torch.randn(N, self.cfg.hidden_channels, dtype=self.torch_dtype, device=self.device)
            self.test_step(x)

        # Performance test
        times = []
        for _ in range(self.cfg.phase1_steps):
            x = torch.randn(N, self.cfg.hidden_channels, dtype=self.torch_dtype, device=self.device)

            tic = perf_counter()
            self.test_step(x)
            times.append(perf_counter() - tic)

        compute_time = sum(times) / len(times)
        flops_per_step = 2 * self.cfg.hidden_channels**2 * N * (self.cfg.n_mlp_hidden_layers + 2)
        ai = (N * self.cfg.hidden_channels**2) \
                / (2 * N * self.cfg.hidden_channels + self.cfg.hidden_channels**2)
        sizeof_dtype = torch.tensor([], dtype=self.torch_dtype).element_size() # in bytes

        log.info(f"Inference performance metrics (per step) with N={N}:")
        log.info(f"\tCompute time: {compute_time:.4f} seconds")
        log.info(f"\tFLOPS: {flops_per_step/1e9:.4f} GFLOPS")
        log.info(f"\tAI: {ai:.4f}")
        log.info(f"\tMemory traffic: {flops_per_step / ai * sizeof_dtype / 1e9:.4f} GB")
        log.info(f"\tFLOPS/s: {flops_per_step / compute_time / 1e12:.4f} TFLOPS/s")
        log.info(f"\tMemory bandwidth: {flops_per_step / ai * sizeof_dtype / 1e9 / compute_time:.4f} GB/s")
        log.info("\n")


@hydra.main(version_base=None, config_path='./conf', config_name='config')
def main(cfg: DictConfig) -> None:
    log.info('\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    log.info('RUNNING WITH INPUTS:')
    log.info(f'{OmegaConf.to_yaml(cfg)}') 
    log.info('\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

    mlp_reprod = MLPReprod(cfg)

    mlp_reprod.train(graph_component='nodes')
    mlp_reprod.train(graph_component='edges')

    mlp_reprod.test(graph_component='nodes')
    mlp_reprod.test(graph_component='edges')


if __name__ == "__main__":
    main()