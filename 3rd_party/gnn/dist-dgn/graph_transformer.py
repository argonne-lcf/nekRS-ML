"""
Distributed Diffusion Graph Transformer (DGT).

Sibling of DistributedDGN that swaps the message-passing processor for an
element-wise transformer processor (element-restricted self-attention with
RoPE, redistribution across nodes that share a global ID, halo swap, FFN).

Adapted from 3rd_party/gnn/dist-gnn/graph_transformer.py with the additions
required by the diffusion model: diffusion-step embedding injection per block,
optional cond_node_features, learnable variance, batching, and the
(pred, var) return contract used by DistributedDGN.
"""

from typing import Any, Dict, Optional

import einops
import torch
import torch.distributed.nn as distnn
import torch.nn as nn
from torch.nn.functional import scaled_dot_product_attention as sdpa

from gnn import SinusoidalPositionEmbedding

try:
    from torch_scatter import scatter_mean

    TORCH_SCATTER_AVAIL = True
except ModuleNotFoundError:
    TORCH_SCATTER_AVAIL = False


def scatter_mean_native(src, index, dim: int = 0, dim_size: int = None):
    """Native torch fallback when torch_scatter is unavailable."""
    assert dim == 0, "scatter_mean_native only supports dim=0"
    if dim_size is None:
        dim_size = index.max().item() + 1
    out = torch.zeros(
        dim_size, src.shape[1], dtype=src.dtype, device=src.device
    )
    counts = torch.zeros(dim_size, 1, dtype=src.dtype, device=src.device)
    idx = index.unsqueeze(1)
    out.scatter_add_(0, idx.expand_as(src), src)
    counts.scatter_add_(0, idx, torch.ones_like(idx, dtype=src.dtype))
    return out / counts.clamp(min=1)


class GeGLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = torch.chunk(x, 2, dim=-1)
        return x * torch.nn.functional.gelu(gate)


class SwiGLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = torch.chunk(x, 2, dim=-1)
        return x * torch.nn.functional.silu(gate)


def apply_rope(
    x: torch.Tensor,
    coords: torch.Tensor,
    max_wavelength: int = 100,
) -> torch.Tensor:
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


class MlpBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        use_bias: bool = False,
        activation: str = "swish",
    ):
        super().__init__()
        self.dense1 = nn.Linear(in_dim, hidden_dim * 2, bias=use_bias)
        if activation == "gelu":
            self.glu = GeGLU()
        elif activation == "swish":
            self.glu = SwiGLU()
        self.dense2 = nn.Linear(hidden_dim, out_dim, bias=use_bias)

        nn.init.xavier_uniform_(self.dense1.weight)
        nn.init.xavier_uniform_(self.dense2.weight)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.dense1(inputs)
        x = self.glu(x)
        x = self.dense2(x)
        return x


class DGTAttentionBlock(nn.Module):
    """
    Element-wise attention transformer block with diffusion-step embedding
    injection. Performs:
      1) inject the diffusion-step embedding into x (per batch element)
      2) for each batch element:
           - pre-norm + element-restricted multi-head self-attention with RoPE
           - average attention output across nodes that share a global ID
             (the "redistribution" step, load-bearing for consistency)
           - halo swap (when SIZE>1 and halo_swap_mode != "none")
           - residual + FFN
    """

    def __init__(
        self,
        hidden_channels: int,
        num_heads: int,
        emb_features: int,
        poly_order: int = 7,
        use_bias: bool = False,
        mlp_ratio: float = 1.0,
        activation: str = "swish",
        halo_swap_mode: str = "none",
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_heads = num_heads
        self.emb_features = emb_features
        self.poly_order = poly_order
        self.halo_swap_mode = halo_swap_mode

        # Diffusion-step embedding projection into the node embedding space
        if self.emb_features > 0:
            self.node_emb_linear = nn.Linear(emb_features, hidden_channels)

        # Attention layers
        self.norm1 = nn.LayerNorm(hidden_channels)
        self.q_proj = nn.Linear(hidden_channels, hidden_channels, bias=use_bias)
        self.k_proj = nn.Linear(hidden_channels, hidden_channels, bias=use_bias)
        self.v_proj = nn.Linear(hidden_channels, hidden_channels, bias=use_bias)
        self.o_proj = nn.Linear(hidden_channels, hidden_channels, bias=use_bias)

        # FFN
        self.norm2 = nn.LayerNorm(hidden_channels)
        self.ffn = MlpBlock(
            hidden_channels, int(hidden_channels * mlp_ratio), hidden_channels
        )

    def _attention_with_consistency(
        self,
        x_batch: torch.Tensor,
        pos: torch.Tensor,
        pos_min: torch.Tensor,
        pos_max: torch.Tensor,
        grouping_index: torch.Tensor,
        mask_send,
        mask_recv,
        buffer_send,
        buffer_recv,
        halo_info: torch.Tensor,
        idx_reduced2full: torch.Tensor,
        idx_full2reduced: torch.Tensor,
        neighboring_procs,
        SIZE,
    ) -> torch.Tensor:
        """Run attention + redistribution + halo swap on a single batch slice.

        Returns the updated x_batch (out-of-place; does not mutate input).
        """
        poly_order = self.poly_order
        nodes_per_element = (poly_order + 1) ** 3
        num_elements = idx_reduced2full.shape[0] // nodes_per_element

        res = x_batch

        # Expand reduced -> full
        x_full = x_batch[idx_reduced2full]
        pos_full = pos[idx_reduced2full]

        # Reshape so num_elements acts as the batch dim, nodes_per_element as seq len
        x_full = x_full.reshape(num_elements, nodes_per_element, x_batch.shape[-1])
        pos_full = pos_full.reshape(
            num_elements, nodes_per_element, pos.shape[-1]
        )
        # Normalize positions to [0,1] using globally-reduced per-coordinate bounds
        pos_full = (pos_full - pos_min) / (pos_max - pos_min)

        # Pre-norm
        x_full = self.norm1(x_full)

        # Multi-head attention with RoPE on q,k
        q = self.q_proj(x_full)
        k = self.k_proj(x_full)
        v = self.v_proj(x_full)

        q = einops.rearrange(q, "ne np (h c) -> ne h np c", h=self.num_heads)
        k = einops.rearrange(k, "ne np (h c) -> ne h np c", h=self.num_heads)
        v = einops.rearrange(v, "ne np (h c) -> ne h np c", h=self.num_heads)

        q = apply_rope(q, pos_full)
        k = apply_rope(k, pos_full)
        attn_output = sdpa(q, k, v)
        attn_output = einops.rearrange(attn_output, "ne h np c -> ne np (h c)")
        attn_output = self.o_proj(attn_output)

        # Redistribution: average attention output across nodes that share a
        # global ID, so coincident physical nodes carry identical values.
        ne, np_per, c = attn_output.shape
        attn_output = attn_output.reshape(ne * np_per, c)
        if TORCH_SCATTER_AVAIL:
            attn_output = scatter_mean(attn_output, grouping_index, dim=0)[
                grouping_index
            ]
        else:
            attn_output = scatter_mean_native(
                attn_output, grouping_index, dim=0
            )[grouping_index]
        # Project full -> reduced
        attn_output = attn_output[idx_full2reduced]

        # Add the attention update + halo sync. We mirror dist-gnn's pattern:
        # work on a fresh copy of x_batch so the per-batch loop above can
        # safely reassemble the output without aliasing across batches.
        num_halo_nodes = halo_info.shape[0]
        if SIZE > 1:
            # Internal slots: residual + attention. Halo slots: keep residual;
            # they will be (re-)populated by the halo swap below.
            x_new = res.clone()
            x_new[:-num_halo_nodes] = res[:-num_halo_nodes] + attn_output
            if (
                self.halo_swap_mode == "all_to_all"
                or self.halo_swap_mode == "all_to_all_opt"
            ):
                x_new = self.halo_swap(
                    x_new,
                    mask_send,
                    mask_recv,
                    buffer_send,
                    buffer_recv,
                    neighboring_procs,
                    SIZE,
                )
            else:
                assert self.halo_swap_mode == "none", "Invalid halo swap mode"

            # Aggregate contributions from neighboring ranks
            idx_recv = halo_info[:, 0]
            idx_send = halo_info[:, 1]
            x_new.index_add_(0, idx_recv, x_new.index_select(0, idx_send))
            # Mean across all contributors (self + incoming)
            counts = torch.ones(x_new.size(0), device=x_new.device)
            ones = torch.ones(idx_recv.size(0), device=x_new.device)
            counts.index_add_(0, idx_recv, ones)
            x_new = x_new / counts.unsqueeze(-1)
        else:
            x_new = res + attn_output

        # Residual FFN
        y = self.norm2(x_new)
        y = self.ffn(y)
        x_new = x_new + y
        return x_new

    def forward(
        self,
        x: torch.Tensor,
        emb: torch.Tensor,
        pos: torch.Tensor,
        pos_min: torch.Tensor,
        pos_max: torch.Tensor,
        grouping_index: torch.Tensor,
        mask_send,
        mask_recv,
        buffer_send,
        buffer_recv,
        halo_info: torch.Tensor,
        idx_reduced2full: torch.Tensor,
        idx_full2reduced: torch.Tensor,
        neighboring_procs,
        SIZE,
        batch: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        if batch is None:
            batch = torch.zeros(
                x.size(0), device=x.device, dtype=torch.long
            )
        batch_size = int(torch.max(batch).item()) + 1

        # Inject diffusion-step embedding (out-of-place)
        if self.emb_features > 0:
            x = x + self.node_emb_linear(emb)[batch]

        # Per-batch attention + redistribution + halo swap + FFN. The graph
        # topology (idx_reduced2full, idx_full2reduced, halo_info) is shared
        # across batches; only the x slice differs.
        if batch_size == 1:
            return self._attention_with_consistency(
                x,
                pos,
                pos_min,
                pos_max,
                grouping_index,
                mask_send,
                mask_recv,
                buffer_send,
                buffer_recv,
                halo_info,
                idx_reduced2full,
                idx_full2reduced,
                neighboring_procs,
                SIZE,
            )

        out = torch.empty_like(x)
        for b in range(batch_size):
            mask = batch == b
            x_b = x[mask]
            x_b_new = self._attention_with_consistency(
                x_b,
                pos,
                pos_min,
                pos_max,
                grouping_index,
                mask_send,
                mask_recv,
                buffer_send,
                buffer_recv,
                halo_info,
                idx_reduced2full,
                idx_full2reduced,
                neighboring_procs,
                SIZE,
            )
            out[mask] = x_b_new
        return out

    def halo_swap(
        self,
        input_tensor,
        mask_send,
        mask_recv,
        buff_send,
        buff_recv,
        neighboring_procs,
        SIZE,
    ):
        """Halo swap via send/receive buffers (all_to_all variants)."""
        if SIZE > 1:
            for i in neighboring_procs:
                n_send = len(mask_send[i])
                buff_send[i][:n_send, :] = input_tensor[mask_send[i]]

            distnn.all_to_all(buff_recv, buff_send)

            for i in neighboring_procs:
                n_recv = len(mask_recv[i])
                input_tensor[mask_recv[i]] = buff_recv[i][:n_recv, :]

        return input_tensor


class DistributedDGT(nn.Module):
    """Distributed Diffusion Graph Transformer (DGT).

    Args:
        arch (Dict[str, Any]): Architecture configuration. Keys:
            input_node_features (int): per-node input dimension
            cond_node_features (int): per-node conditional feature dimension (0 if unused)
            hidden_channels (int): attention/FFN hidden width
            n_transformer_layers (int): number of attention blocks
            num_heads (int): heads in multi-head attention
            poly_order (int): spectral element polynomial order (nodes/element = (p+1)^3)
            emb_width (int): width of the diffusion-step embedding
            halo_swap_mode (str): one of {none, all_to_all, all_to_all_opt}
            learnable_variance (bool): if True the decoder outputs 2x input_node_features
            mlp_ratio (float): FFN hidden ratio
            name (str): tag used in checkpoint filename
    """

    def __init__(self, arch: Dict[str, Any]):
        super().__init__()
        self.parse_arch(arch)

        # ~~~~ Diffusion-step embedding (mirrors DistributedDGN exactly)
        freq_width = max(self.hidden_channels, 128)
        emb_width = max(self.emb_width, freq_width * 4)
        self.diffusion_step_embedding = nn.Sequential(
            SinusoidalPositionEmbedding(freq_width),
            nn.Linear(freq_width, emb_width),
            nn.SiLU(),
            nn.Linear(emb_width, emb_width),
            nn.SiLU(),
        )

        # ~~~~ Diffusion-step encoder injected into node features at the input
        self.diffusion_step_encoder = nn.ModuleList([
            nn.Linear(emb_width, self.hidden_channels),
            nn.SELU(),
            nn.Linear(self.hidden_channels * 2, self.hidden_channels),
        ])

        # ~~~~ Encoder MLP
        self.encoder = MlpBlock(
            self.input_node_features + self.cond_node_features,
            self.hidden_channels,
            self.hidden_channels,
        )

        # ~~~~ Processor: stack of DGTAttentionBlock
        self.processor = nn.ModuleList()
        for _ in range(self.n_transformer_layers):
            self.processor.append(
                DGTAttentionBlock(
                    hidden_channels=self.hidden_channels,
                    num_heads=self.num_heads,
                    emb_features=emb_width,
                    poly_order=self.poly_order,
                    mlp_ratio=self.mlp_ratio,
                    halo_swap_mode=self.halo_swap_mode,
                )
            )

        # ~~~~ Decoder MLP. Width matches DistributedDGN's contract.
        self.decoder = MlpBlock(
            self.hidden_channels,
            self.hidden_channels,
            self.output_node_features,
        )

    def parse_arch(self, arch: Dict[str, Any]):
        self.arch = arch
        self.input_node_features = arch["input_node_features"]
        self.cond_node_features = arch["cond_node_features"]
        self.hidden_channels = arch["hidden_channels"]
        self.n_transformer_layers = arch["n_transformer_layers"]
        self.num_heads = arch["num_heads"]
        self.poly_order = arch["poly_order"]
        self.emb_width = arch.get("emb_width", 128)
        self.halo_swap_mode = arch["halo_swap_mode"]
        self.learnable_variance = arch.get("learnable_variance", False)
        self.mlp_ratio = arch.get("mlp_ratio", 1.0)
        self.output_node_features = (
            self.input_node_features * 2
            if self.learnable_variance
            else self.input_node_features
        )
        self.name = arch["name"]

    def get_arch(self) -> dict:
        return self.arch

    def get_save_header(self) -> str:
        header = self.name
        header += (
            f"_H{self.arch['hidden_channels']}"
            f"_T{self.arch['n_transformer_layers']}"
            f"_heads{self.arch['num_heads']}"
            f"_{self.arch['halo_swap_mode']}"
            f"_emb{self.arch['emb_width']}"
            f"_lv{self.arch['learnable_variance']}"
            f"_condf{self.arch['cond_node_features']}"
        )
        return header

    def forward(
        self,
        field_r: torch.Tensor,
        r: torch.Tensor,
        pos: torch.Tensor,
        pos_min: torch.Tensor,
        pos_max: torch.Tensor,
        index: torch.Tensor,
        mask_send: list,
        mask_recv: list,
        buffer_send: list,
        buffer_recv: list,
        halo_info: torch.Tensor,
        idx_reduced2full: torch.Tensor,
        idx_full2reduced: torch.Tensor,
        neighboring_procs,
        SIZE,
        cond_node_features: Optional[torch.Tensor] = None,
        batch: Optional[torch.LongTensor] = None,
        debug_dump: Optional[Dict[str, Any]] = None,
    ):
        if batch is None:
            batch = torch.zeros(
                field_r.size(0), device=field_r.device, dtype=torch.long
            )
        batch_size = int(torch.max(batch).item()) + 1

        # ~~~~ Diffusion step embedding
        emb = self.diffusion_step_embedding(r)

        # ~~~~ Node encoder (with optional conditional features)
        if cond_node_features is not None:
            x = torch.cat(
                [field_r, cond_node_features.repeat(batch_size, 1)], dim=1
            )
        else:
            x = field_r
        x = self.encoder(x)

        # ~~~~ Inject diffusion step into encoder output (mirrors DGN)
        emb_proj = self.diffusion_step_encoder[0](emb)
        x = torch.cat([x, emb_proj[batch]], dim=1)
        for layer in self.diffusion_step_encoder[1:]:
            x = layer(x)

        # ~~~~ Precompute the grouping index used by the redistribution step.
        # `index` is the per-node global ID over the full graph (length
        # idx_reduced2full.shape[0]). torch.unique gives ascending grouping
        # IDs in [0, n_unique). The transformer's per-batch loop reuses this
        # since the graph topology is identical across batches.
        _, grouping_index = torch.unique(
            index[idx_reduced2full], return_inverse=True
        )

        # ~~~~ Processor
        for i in range(self.n_transformer_layers):
            x = self.processor[i](
                x,
                emb,
                pos,
                pos_min,
                pos_max,
                grouping_index,
                mask_send,
                mask_recv,
                buffer_send,
                buffer_recv,
                halo_info,
                idx_reduced2full,
                idx_full2reduced,
                neighboring_procs,
                SIZE,
                batch=batch,
            )

        # ~~~~ Decoder
        x = self.decoder(x)

        if self.learnable_variance:
            return torch.chunk(x, 2, dim=1)
        else:
            return x, torch.zeros_like(x)
