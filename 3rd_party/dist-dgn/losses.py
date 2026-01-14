from typing import Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import scatter


def batch_wise_mean(
    field: torch.Tensor,
    batch: torch.LongTensor,
) -> torch.Tensor:
    r"""Compute the batch-wise mean of a field.
        Args:
            field (Tensor): The field to average. Dimension: (num_nodes, num_features).
            batch (LongTensor): The batch vector. Dimension: (num_nodes).
        Returns:
            Tensor: The batch-wise mean. Dimension: (batch_size).
    """
    field = field.mean(dim=1) # Dimension: (num_nodes)
    batch_size = batch.max().item() + 1
    return scatter(field, batch, dim=0, dim_size=batch_size, reduce='mean') # Dimension: (batch_size)


def vlb_loss(
    field_start: torch.Tensor,
    true_distribution: Tuple[torch.Tensor, torch.Tensor],
    model_distribution: Tuple[torch.Tensor, torch.Tensor],
    batch: torch.LongTensor,
    r: torch.Tensor,
) -> torch.Tensor:
    """Compute the variational lower bound (VLB) loss used in the hybrid loss function 
       for diffusion models from the paper 
       Improved Denoising Diffusion Probabilistic Models (https://arxiv.org/abs/2102.09672).
       Adapted from https://github.com/tum-pbs/dgn4cfd/blob/main/dgn4cfd/nn/losses.py
        Args:
            model_noise (Tensor): The model noise. Dimension: (num_nodes, num_features).
            noise (Tensor): The noise. Dimension: (num_nodes, num_features).
            batch (LongTensor): The batch vector. Dimension: (num_nodes).
        Returns:
            Tensor: The VLB loss. Dimension: (batch_size).
    """
    kl = kl_divergence(true_distribution[0], true_distribution[1], model_distribution[0], model_distribution[1]) # Dimension: (num_nodes, num_features)
    kl = batch_wise_mean(kl, batch) / math.log(2.0) # Dimension (batch_size)
    if (r == 0).any():
        decoder_nll = F.gaussian_nll_loss(model_distribution[0], field_start, model_distribution[1], reduction='none') # Dimension: (num_nodes, num_features)
        decoder_nll = batch_wise_mean(decoder_nll, batch) # Dimension (batch_size)
        kl = torch.where((r == 0), decoder_nll, kl)
    return kl # Dimension (batch_size)


def kl_divergence(
    mean1:     torch.Tensor,
    variance1: torch.Tensor,
    mean2:     torch.Tensor,
    variance2: torch.Tensor
) -> torch.Tensor:
    """Compute the KL divergence between two normal distributions.
        Args:
            mean1 (Tensor): The mean of the first normal distribution. Dimension: (num_nodes, num_features).
            variance1 (Tensor): The variance of the first normal distribution. Dimension: (num_nodes, num_features).
            mean2 (Tensor): The mean of the second normal distribution. Dimension: (num_nodes, num_features).
            variance2 (Tensor): The variance of the second normal distribution. Dimension: (num_nodes, num_features).
        Returns:
            Tensor: The KL divergence. Dimension: (num_nodes, num_features).
    """
    return 0.5 * (torch.log(variance2) - torch.log(variance1) + variance1 / variance2 + (mean1 - mean2)**2 / variance2 - 1) 