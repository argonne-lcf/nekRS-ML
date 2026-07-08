"""Attention utilities shared between graph_transformer.py and hierarchical.py.

Kept in its own module so importers (e.g. the unit tests for the hierarchical
attention modules) don't have to pull in the rest of the dist-dgn package
(which transitively imports torch_geometric via gnn.py).

Currently exports ``apply_rope`` only. Add to this module any other small,
dependency-free helpers that need to be shared.
"""

import einops
import torch


def apply_rope(
    x: torch.Tensor,
    coords: torch.Tensor,
    max_wavelength: int = 100,
) -> torch.Tensor:
    """Rotary positional embedding applied per spatial coordinate.

    Args:
        x: (batch, heads, n, c) tensor of queries or keys.
        coords: (batch, n, dim) tensor of positions (one entry per spatial
            coordinate). Typically ``dim == 3`` for the nekRS 3D mesh; 2D cases
            extrude with a constant z so dim is still 3.
        max_wavelength: RoPE wavelength upper bound.
    """
    n_dim = coords.shape[-1]
    feature_dim = x.shape[-1]

    per_dim_features = 2 * (feature_dim // (2 * n_dim))
    rotated_chunks = []
    for i in range(n_dim):
        current_x_chunk = x[
            ..., i * per_dim_features : (i + 1) * per_dim_features
        ]
        current_coords = coords[..., i]

        head_dim = per_dim_features
        half_head_dim = head_dim // 2
        fraction = 2 * torch.arange(half_head_dim, device=x.device) / head_dim
        timescale = max_wavelength**fraction

        theta = current_coords.unsqueeze(-1) / timescale
        sin = torch.sin(theta)
        cos = torch.cos(theta)

        first_half, second_half = torch.chunk(current_x_chunk, 2, dim=-1)
        sin = einops.repeat(sin, "b n c -> b h n c", h=first_half.shape[1])
        cos = einops.repeat(cos, "b n c -> b h n c", h=first_half.shape[1])

        rotated_first_half = first_half * cos - second_half * sin
        rotated_second_half = second_half * cos + first_half * sin

        rotated_chunk = torch.cat(
            [rotated_first_half, rotated_second_half], dim=-1
        )
        rotated_chunks.append(rotated_chunk)

    result = torch.cat(rotated_chunks, dim=-1)
    return result.to(x.dtype)
