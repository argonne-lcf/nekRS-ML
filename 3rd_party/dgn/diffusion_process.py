import torch
from torch import nn
import torch.nn.functional as F
from typing import Tuple
import math
import numpy as np


class DiffusionProcess:
    r"""Defines a diffusion process that can be used to diffuse a field defined on the nodes of a graph.

    Args:
        num_steps (int): The number of diffusion steps.
        schedule_type (str, optional): The type of schedule to use for the beta parameter of the diffusion process. It can be 'linear' or 'cosine'. Defaults to 'linear'.
        beta_start (float, optional): The initial value of the beta parameter of the diffusion process. Defaults to 0.0001.
        beta_end (float, optional): The final value of the beta parameter of the diffusion process. Defaults to 0.02.
        max_beta (float, optional): The maximum value of the beta parameter of the diffusion process. Defaults to 0.999.
    
    Methods:
        get_betas: Returns the schedule of the beta parameter of the diffusion process.
        __call__: Forwards the diffusion process from 'field_start' for 'r' diffusion-steps.
        sample_r: Samples the index of the diffusion step 'r' from a uniform distribution.
        get_posterior_mean_and_variance: Returns the posterior mean and variance of the field after 'r' diffusion steps.
        get_index_from_list: Returns the 'r'-th element of 'values' for each node in 'batch'.
    """
    def __init__(
        self,
        num_steps:     int,
        schedule_type: str   = 'linear',
        beta_start:    float = 0.0001,
        beta_end:      float = 0.02,
        max_beta:      float = 0.999
    ) -> None:
        super().__init__()
        # Validate inputs
        assert schedule_type in ['linear', 'cosine'], f"Schedule type {schedule_type} not supported. Supported types are 'linear' and 'cosine'."
        # Define parameters of the diffusion process
        self.num_steps     = num_steps
        self.schedule_type = schedule_type
        self.beta_start    = beta_start
        self.beta_end      = beta_end
        self.max_beta      = max_beta
        # Get beta schedule (on the CPU)
        self.betas = self.get_betas()
        # Precompute some constant coefficients (on the CPU)
        self.init_coefficients()

    def init_coefficients(self):
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1. / self.alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_log_variance_clipped = torch.log(torch.cat([self.posterior_variance[[1]], self.posterior_variance[1:]])) # This is clipped to avoind nan in the backward pass
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1. - self.alphas_cumprod)
    
    @property
    def steps(self) -> list[int]:
        return list(range(self.num_steps))

    def __repr__(self):
        if self.schedule_type == 'linear':
            return f"DiffusionProcess(num_steps={self.num_steps}, schedule_type={self.schedule_type}, beta_start={self.beta_start}, beta_end={self.beta_end}, max_beta={self.max_beta})"
        elif self.schedule_type == 'cosine':
            return f"DiffusionProcess(num_steps={self.num_steps}, schedule_type={self.schedule_type}, max_beta={self.max_beta})"
    
    def __str__(self) -> str:
        return self.__repr__()

    def get_betas(self) -> torch.Tensor:
        r"""Returns the schedule of the beta parameter of the diffusion process.

        Returns:
            torch.Tensor: The betas of the diffusion process. Dimensions: [num_steps].
        """
        if self.schedule_type == 'linear':
            scale = 1000 / self.num_steps
            beta_start = scale * self.beta_start
            beta_end = scale * self.beta_end
            betas = torch.linspace(beta_start, beta_end, self.num_steps)
        elif self.schedule_type == "cosine":        
            f_t = lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
            betas = []
            for i in range(self.num_steps):
                t1 = i / self.num_steps
                t2 = (i + 1) / self.num_steps
                betas.append(1 - f_t(t2) / f_t(t1))
            betas = torch.tensor(betas)
        # Truncate the betas to the maximum value
        betas = torch.minimum(betas, torch.tensor(self.max_beta))
        return betas # Dimensions: (num_steps)

    def forward(
        self,
        field_start:    torch.Tensor,
        r:              torch.Tensor, 
        batch:          torch.Tensor = None,
        dirichlet_mask: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Forwards the diffusion process from 'field_start' to diffusion-step `r`.

        Args:
            field_start (torch.Tensor): The initial field defined on the nodes of a graph. Dimensions: [num_nodes, num_fields].
            r (torch.Tensor): The number of diffusion steps to perform. Dimensions: [batch_size].
            batch (torch.Tensor): The batch indices of the nodes of the graph. Dimensions: [num_nodes]. Defaults to 'None'.
                If 'None', then it is assumed that all nodes belong to the same graph.
            dirichlet_mask (torch.Tensor, optional): A mask that indicates which nodes and features have a Dirichlet boudnary condition.
                Dimensions: [num_nodes, num_fields]. Wherever the mask is 1, the field is not diffused.
                If 'None', then it is assumed that there are no Dirichlet boundary conditions. Defaults to 'None'.

        Returns:
            torch.Tensor: The field after 'r' diffusion steps, defined on the nodes of a graph. Dimensions: [num_nodes, num_fields].
            torch.Tensor: The (normalised Gaussian) noise employed to diffuse 'field_start'. Dimensions: [num_nodes, num_fields].
        """
        if dirichlet_mask is not None:
            noise = torch.randn_like(field_start) * (~dirichlet_mask) # (num_nodes, num_fields)
        else:
            noise = torch.randn_like(field_start) # (num_nodes, num_fields)
        if batch is None:
            batch = torch.zeros(field_start.size(0), device=field_start.device)
        
        # Get the coefficients for the diffusion process
        sqrt_alphas_cumprod_t = self.get_index_from_list(self.sqrt_alphas_cumprod, batch, r) # Dimensions: (num_nodes, 1)
        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(self.sqrt_one_minus_alphas_cumprod, batch, r) # Dimensions: (num_nodes, 1)
        # Apply the diffusion process. The coefficients are shared across the field dimension.
        return sqrt_alphas_cumprod_t * field_start + sqrt_one_minus_alphas_cumprod_t * noise, noise # Dimensions: (num_nodes, num_fields), (num_nodes, num_fields)

    def sample_r(
        self,
        batch_size: int          = 1,
        device:     torch.device = torch.device('cpu')
    ) -> torch.Tensor:
        r"""Samples the index of the diffusion step `r` from a uniform distribution.

        Args:
            batch_size (int, optional): The number of diffusion steps to sample. Defaults to 1.

        Returns:
            torch.Tensor: The index of the diffusion step `r`. Dimensions: [batch_size].
        """
        return torch.randint(0, self.num_steps, (batch_size,), device=device).long() # Dimensions: [batch_size]
        

    def get_posterior_mean_and_variance(
        self,
        field_start: torch.Tensor,
        field_r: torch.Tensor,
        batch: torch.Tensor,
        r: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Returns the posterior mean and variance of the field after 'r' diffusion steps.

        Args:
            field_start (torch.Tensor): The initial field defined on the nodes of a graph. Dimensions: [num_nodes, num_fields].
            field_r (torch.Tensor): The field after 'r' diffusion steps, defined on the nodes of a graph. Dimensions: [num_nodes, num_fields].
            batch (torch.Tensor): The batch indices of the nodes of the graph. Dimensions: [num_nodes].
            r (torch.Tensor): The number of diffusion steps to perform. Dimensions: [batch_size].

        Returns:
            torch.Tensor: The posterior mean of the field after 'r' diffusion steps, defined on the nodes of a graph. Dimensions: (num_nodes, num_fields).
            torch.Tensor: The posterior variance of the field after 'r' diffusion steps, defined on the nodes of a graph. Dimensions: (num_nodes,).
        """
        posterior_mean = self.get_index_from_list(self.posterior_mean_coef1, batch, r) * field_start + self.get_index_from_list(self.posterior_mean_coef2, batch, r) * field_r
        posterior_variance = self.get_index_from_list(self.posterior_variance, batch, r)
        return posterior_mean, posterior_variance

    @staticmethod
    def get_index_from_list(
        values: torch.Tensor,
        batch: torch.Tensor,
        r: torch.Tensor
    ) -> torch.Tensor:
        r"""Returns the `r`-th element of `values` for each node in `batch`. These values are the same for all the nodes in the same graph.

        Args:
            values (torch.Tensor): The values to index. Dimensions: [num_steps].
            batch (torch.Tensor): The batch indices of the nodes in the graph. Dimensions: [num_nodes].
            r (torch.Tensor): The indices to use. Dimensions: [batch_size].
        """
        assert batch.device == r.device, f"The device of batch and r must be the same."
        device = batch.device
        batch_size = len(r)
        # Validate that the batch_size is the same as the number of graphs in 'batch'
        assert batch.max().item() + 1 == batch_size, f"The batch_size of r and the number of graphs in batch must be the same."
        # Get the 'r'-th element of 'fields' for each graph in 'batch'
        node_r = values.to(device)[r] # Dimensions: [batch_size]
        # Get the number of nodes in each graph
        num_nodes_per_graph = torch.bincount(batch)
        # Stack the repeated values for each graph in the batch
        node_r = node_r.repeat_interleave(num_nodes_per_graph) # Dimensions: [num_nodes]
        # Add a pseudo-field dimension
        return node_r.unsqueeze(-1) # Dimensions: (num_nodes, 1)