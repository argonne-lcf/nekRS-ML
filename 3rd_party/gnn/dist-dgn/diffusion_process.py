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
        num_steps: int,
        schedule_type: str = "linear",
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        max_beta: float = 0.999,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        # Validate inputs
        assert schedule_type in ["linear", "cosine"], (
            f"Schedule type {schedule_type} not supported. Supported types are 'linear' and 'cosine'."
        )
        # Define parameters of the diffusion process
        self.num_steps = num_steps
        self.schedule_type = schedule_type
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.max_beta = max_beta
        # Float dtype for the precomputed coefficient tensors. Should match the
        # model's parameter dtype so the multiplications in ``forward`` /
        # ``get_v_target`` etc. don't silently promote bf16 model inputs back
        # to fp32. Default fp32 preserves the original behaviour for standalone
        # use; the trainer always passes its ``self.torch_dtype``.
        self.dtype = dtype
        # Get beta schedule (on the CPU)
        self.betas = self.get_betas()
        # Precompute some constant coefficients (on the CPU)
        self.init_coefficients()

    def init_coefficients(self):
        # Compute all coefficients in fp64 for numerical stability of the
        # cumulative-product chain, then cast each precomputed tensor to
        # self.dtype so subsequent multiplications with model tensors stay
        # in the model's dtype (no silent fp32 promotion of bf16 inputs).
        betas64 = self.betas.to(torch.float64)
        alphas64 = 1.0 - betas64
        alphas_cumprod64 = torch.cumprod(alphas64, axis=0)
        alphas_cumprod_prev64 = F.pad(
            alphas_cumprod64[:-1], (1, 0), value=1.0
        )
        sqrt_recip_alphas64 = torch.sqrt(1.0 / alphas64)
        sqrt_alphas_cumprod64 = torch.sqrt(alphas_cumprod64)
        sqrt_one_minus_alphas_cumprod64 = torch.sqrt(1.0 - alphas_cumprod64)
        posterior_variance64 = (
            betas64
            * (1.0 - alphas_cumprod_prev64)
            / (1.0 - alphas_cumprod64)
        )
        posterior_log_variance_clipped64 = torch.log(
            torch.cat([
                posterior_variance64[[1]],
                posterior_variance64[1:],
            ])
        )  # This is clipped to avoid nan in the backward pass
        posterior_mean_coef1_64 = (
            betas64
            * torch.sqrt(alphas_cumprod_prev64)
            / (1.0 - alphas_cumprod64)
        )
        posterior_mean_coef2_64 = (
            (1.0 - alphas_cumprod_prev64)
            * torch.sqrt(alphas64)
            / (1.0 - alphas_cumprod64)
        )

        d = self.dtype
        self.alphas = alphas64.to(d)
        self.alphas_cumprod = alphas_cumprod64.to(d)
        self.alphas_cumprod_prev = alphas_cumprod_prev64.to(d)
        self.sqrt_recip_alphas = sqrt_recip_alphas64.to(d)
        self.sqrt_alphas_cumprod = sqrt_alphas_cumprod64.to(d)
        self.sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod64.to(d)
        self.posterior_variance = posterior_variance64.to(d)
        self.posterior_log_variance_clipped = posterior_log_variance_clipped64.to(d)
        self.posterior_mean_coef1 = posterior_mean_coef1_64.to(d)
        self.posterior_mean_coef2 = posterior_mean_coef2_64.to(d)

    @property
    def steps(self) -> list[int]:
        return list(range(self.num_steps))

    def __repr__(self):
        if self.schedule_type == "linear":
            return f"DiffusionProcess(num_steps={self.num_steps}, schedule_type={self.schedule_type}, beta_start={self.beta_start}, beta_end={self.beta_end}, max_beta={self.max_beta})"
        elif self.schedule_type == "cosine":
            return f"DiffusionProcess(num_steps={self.num_steps}, schedule_type={self.schedule_type}, max_beta={self.max_beta})"

    def __str__(self) -> str:
        return self.__repr__()

    def get_betas(self) -> torch.Tensor:
        r"""Returns the schedule of the beta parameter of the diffusion process.

        Returns:
            torch.Tensor: The betas of the diffusion process. Dimensions: [num_steps].
        """
        # Compute the schedule in fp64 internally for numerical stability of
        # the downstream cumulative products (init_coefficients does cumprod
        # over self.num_steps factors close to 1, which is precision-sensitive
        # in low-precision dtypes). Cast to the configured ``self.dtype`` at
        # the end so the precomputed coefficients match the model dtype.
        if self.schedule_type == "linear":
            scale = 1000 / self.num_steps
            beta_start = scale * self.beta_start
            beta_end = scale * self.beta_end
            betas = torch.linspace(
                beta_start, beta_end, self.num_steps, dtype=torch.float64
            )
        elif self.schedule_type == "cosine":
            f_t = lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
            betas = []
            for i in range(self.num_steps):
                t1 = i / self.num_steps
                t2 = (i + 1) / self.num_steps
                betas.append(1 - f_t(t2) / f_t(t1))
            betas = torch.tensor(betas, dtype=torch.float64)
        # Truncate the betas to the maximum value, then cast to target dtype.
        betas = torch.minimum(betas, torch.tensor(self.max_beta, dtype=torch.float64))
        return betas.to(self.dtype)  # Dimensions: (num_steps)

    def forward(
        self,
        field_start: torch.Tensor,
        r: torch.Tensor,
        batch: torch.Tensor = None,
        dirichlet_mask: torch.Tensor = None,
        noise: torch.Tensor = None,
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
            noise (torch.Tensor, optional): Pre-sampled noise to use instead of drawing internally.
                Dimensions: [num_nodes, num_fields]. Used by the distributed trainer to pass in
                noise that has already been halo-synchronized across MPI ranks so boundary copies
                of the same physical node carry identical values. If 'None', noise is sampled
                internally via torch.randn_like. Defaults to 'None'.

        Returns:
            torch.Tensor: The field after 'r' diffusion steps, defined on the nodes of a graph. Dimensions: [num_nodes, num_fields].
            torch.Tensor: The (normalised Gaussian) noise employed to diffuse 'field_start'. Dimensions: [num_nodes, num_fields].
        """
        device = field_start.device
        if noise is None:
            if dirichlet_mask is not None:
                noise = torch.randn_like(field_start) * (
                    ~dirichlet_mask
                )  # (num_nodes, num_fields)
            else:
                noise = torch.randn_like(field_start)  # (num_nodes, num_fields)
        elif dirichlet_mask is not None:
            noise = noise * (~dirichlet_mask)
        if batch is None:
            batch = torch.zeros(
                field_start.size(0), device=device, dtype=torch.long
            )

        # Get the coefficients for the diffusion process
        sqrt_alphas_cumprod_t = self.get_index_from_list(
            self.sqrt_alphas_cumprod, batch, r
        )  # Dimensions: (num_nodes, 1)
        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(
            self.sqrt_one_minus_alphas_cumprod, batch, r
        )  # Dimensions: (num_nodes, 1)
        sig_to_noise_ratio_t = (
            sqrt_alphas_cumprod_t / sqrt_one_minus_alphas_cumprod_t
        ) ** 2  # Dimensions: (num_nodes, 1)
        idx = torch.cat([
            torch.tensor([0], device=device),
            (batch[1:] != batch[:-1]).nonzero(as_tuple=True)[0] + 1,
        ])
        sig_to_noise_ratio_t = sig_to_noise_ratio_t[
            idx
        ]  # Dimensions: (batch_size,)

        # Apply the diffusion process. The coefficients are shared across the field dimension.
        field_r = (
            sqrt_alphas_cumprod_t * field_start
            + sqrt_one_minus_alphas_cumprod_t * noise
        )
        return (
            field_r,
            noise,
            sig_to_noise_ratio_t.squeeze(-1),
        )  # Dimensions: (num_nodes, num_fields), (num_nodes, num_fields), (batch_size)

    def sample_r(
        self, batch_size: int = 1, device: torch.device = torch.device("cpu")
    ) -> torch.Tensor:
        r"""Samples the index of the diffusion step `r` from a uniform distribution.

        Args:
            batch_size (int, optional): The number of diffusion steps to sample. Defaults to 1.

        Returns:
            torch.Tensor: The index of the diffusion step `r`. Dimensions: [batch_size].
        """
        return torch.randint(
            0, self.num_steps, (batch_size,), device=device
        ).long()  # Dimensions: [batch_size]

    def predict_x0_from_noise(
        self,
        field_r: torch.Tensor,
        noise: torch.Tensor,
        r: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        r"""Recovers the clean field x_0 from the noisy field x_t and the noise.

        Uses the relationship: x_0 = (x_t - sigma_t * eps) / alpha_bar_t

        Args:
            field_r (torch.Tensor): The noisy field x_t. Dimensions: [num_nodes, num_fields].
            noise (torch.Tensor): The noise eps. Dimensions: [num_nodes, num_fields].
            r (torch.Tensor): The diffusion step indices. Dimensions: [batch_size].
            batch (torch.Tensor): The batch indices of the nodes. Dimensions: [num_nodes].

        Returns:
            torch.Tensor: The predicted clean field x_0. Dimensions: [num_nodes, num_fields].
        """
        sqrt_alphas_cumprod_t = self.get_index_from_list(
            self.sqrt_alphas_cumprod, batch, r
        )
        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(
            self.sqrt_one_minus_alphas_cumprod, batch, r
        )
        return (
            field_r - sqrt_one_minus_alphas_cumprod_t * noise
        ) / sqrt_alphas_cumprod_t

    def predict_noise_from_x0(
        self,
        field_r: torch.Tensor,
        field_start: torch.Tensor,
        r: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        r"""Recovers the noise eps from the noisy field x_t and the clean field x_0.

        Uses the relationship: eps = (x_t - alpha_bar_t * x_0) / sigma_t

        Args:
            field_r (torch.Tensor): The noisy field x_t. Dimensions: [num_nodes, num_fields].
            field_start (torch.Tensor): The clean field x_0. Dimensions: [num_nodes, num_fields].
            r (torch.Tensor): The diffusion step indices. Dimensions: [batch_size].
            batch (torch.Tensor): The batch indices of the nodes. Dimensions: [num_nodes].

        Returns:
            torch.Tensor: The predicted noise eps. Dimensions: [num_nodes, num_fields].
        """
        sqrt_alphas_cumprod_t = self.get_index_from_list(
            self.sqrt_alphas_cumprod, batch, r
        )
        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(
            self.sqrt_one_minus_alphas_cumprod, batch, r
        )
        return (
            field_r - sqrt_alphas_cumprod_t * field_start
        ) / sqrt_one_minus_alphas_cumprod_t

    def get_v_target(
        self,
        field_start: torch.Tensor,
        noise: torch.Tensor,
        r: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        r"""Computes the v-prediction target: v = sqrt(alpha_bar) * eps - sqrt(1 - alpha_bar) * x_0.

        This is the target for v-prediction as defined in
        "Progressive Distillation for Fast Sampling of Diffusion Models"
        (Salimans & Ho, 2022, https://arxiv.org/abs/2202.00512).

        Args:
            field_start (torch.Tensor): The clean field x_0. Dimensions: [num_nodes, num_fields].
            noise (torch.Tensor): The noise eps. Dimensions: [num_nodes, num_fields].
            r (torch.Tensor): The diffusion step indices. Dimensions: [batch_size].
            batch (torch.Tensor): The batch indices of the nodes. Dimensions: [num_nodes].

        Returns:
            torch.Tensor: The v-prediction target. Dimensions: [num_nodes, num_fields].
        """
        sqrt_alphas_cumprod_t = self.get_index_from_list(
            self.sqrt_alphas_cumprod, batch, r
        )
        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(
            self.sqrt_one_minus_alphas_cumprod, batch, r
        )
        return (
            sqrt_alphas_cumprod_t * noise
            - sqrt_one_minus_alphas_cumprod_t * field_start
        )

    def predict_x0_from_v(
        self,
        field_r: torch.Tensor,
        v: torch.Tensor,
        r: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        r"""Recovers the clean field x_0 from the noisy field x_t and the predicted v.

        Uses the relationship: x_0 = sqrt(alpha_bar) * x_t - sqrt(1 - alpha_bar) * v

        Args:
            field_r (torch.Tensor): The noisy field x_t. Dimensions: [num_nodes, num_fields].
            v (torch.Tensor): The predicted v. Dimensions: [num_nodes, num_fields].
            r (torch.Tensor): The diffusion step indices. Dimensions: [batch_size].
            batch (torch.Tensor): The batch indices of the nodes. Dimensions: [num_nodes].

        Returns:
            torch.Tensor: The predicted clean field x_0. Dimensions: [num_nodes, num_fields].
        """
        sqrt_alphas_cumprod_t = self.get_index_from_list(
            self.sqrt_alphas_cumprod, batch, r
        )
        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(
            self.sqrt_one_minus_alphas_cumprod, batch, r
        )
        return (
            sqrt_alphas_cumprod_t * field_r
            - sqrt_one_minus_alphas_cumprod_t * v
        )

    def predict_noise_from_v(
        self,
        field_r: torch.Tensor,
        v: torch.Tensor,
        r: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        r"""Recovers the noise eps from the noisy field x_t and the predicted v.

        Uses the relationship: eps = sqrt(1 - alpha_bar) * x_t + sqrt(alpha_bar) * v

        Args:
            field_r (torch.Tensor): The noisy field x_t. Dimensions: [num_nodes, num_fields].
            v (torch.Tensor): The predicted v. Dimensions: [num_nodes, num_fields].
            r (torch.Tensor): The diffusion step indices. Dimensions: [batch_size].
            batch (torch.Tensor): The batch indices of the nodes. Dimensions: [num_nodes].

        Returns:
            torch.Tensor: The predicted noise eps. Dimensions: [num_nodes, num_fields].
        """
        sqrt_alphas_cumprod_t = self.get_index_from_list(
            self.sqrt_alphas_cumprod, batch, r
        )
        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(
            self.sqrt_one_minus_alphas_cumprod, batch, r
        )
        return (
            sqrt_one_minus_alphas_cumprod_t * field_r
            + sqrt_alphas_cumprod_t * v
        )

    def get_posterior_mean_and_variance(
        self,
        field_start: torch.Tensor,
        field_r: torch.Tensor,
        batch: torch.Tensor,
        r: torch.Tensor,
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
        posterior_mean = (
            self.get_index_from_list(self.posterior_mean_coef1, batch, r)
            * field_start
            + self.get_index_from_list(self.posterior_mean_coef2, batch, r)
            * field_r
        )
        posterior_variance = self.get_index_from_list(
            self.posterior_variance, batch, r
        )
        return posterior_mean, posterior_variance

    @staticmethod
    def get_index_from_list(
        values: torch.Tensor, batch: torch.Tensor, r: torch.Tensor
    ) -> torch.Tensor:
        r"""Returns the `r`-th element of `values` for each node in `batch`. These values are the same for all the nodes in the same graph.

        Args:
            values (torch.Tensor): The values to index. Dimensions: [num_steps].
            batch (torch.Tensor): The batch indices of the nodes in the graph. Dimensions: [num_nodes].
            r (torch.Tensor): The indices to use. Dimensions: [batch_size].
        """
        assert batch.device == r.device, (
            f"The device of batch and r must be the same."
        )
        device = batch.device
        batch_size = len(r)
        # Validate that the batch_size is the same as the number of graphs in 'batch'
        assert batch.max().item() + 1 == batch_size, (
            f"The batch_size of r and the number of graphs in batch must be the same."
        )
        # Get the 'r'-th element of 'fields' for each graph in 'batch'
        node_r = values.to(device)[r]  # Dimensions: [batch_size]
        # Get the number of nodes in each graph
        try:
            num_nodes_per_graph = torch.bincount(batch)
        except NotImplementedError:
            # torch.bincount is not implemented for float on xpu devices
            size = int(batch.max().item()) + 1
            num_nodes_per_graph = torch.zeros(
                size, dtype=torch.long, device=device
            )
            num_nodes_per_graph.scatter_add_(
                0, batch, torch.ones_like(batch, dtype=torch.long)
            )
        # Stack the repeated values for each graph in the batch
        node_r = node_r.repeat_interleave(
            num_nodes_per_graph
        )  # Dimensions: [num_nodes]
        # Add a pseudo-field dimension
        return node_r.unsqueeze(-1)  # Dimensions: (num_nodes, 1)
