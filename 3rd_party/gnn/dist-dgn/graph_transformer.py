"""
Distributed Diffusion Graph Transformer (DGT).

Sibling of DistributedDGN that swaps the message-passing processor for an
element-wise transformer processor (element-restricted self-attention with
RoPE, redistribution across nodes that share a global ID, halo swap, MLP).

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

from attn_utils import apply_rope
from gnn import SinusoidalPositionEmbedding

try:
    from torch_scatter import scatter_add, scatter_max, scatter_mean
    from torch_scatter.composite import scatter_softmax

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


def scatter_add_native(src, index, dim: int = 0, dim_size: int = None):
    """Native torch fallback: sum src rows by group index along dim 0."""
    assert dim == 0, "scatter_add_native only supports dim=0"
    if dim_size is None:
        dim_size = index.max().item() + 1
    out = torch.zeros(
        dim_size, src.shape[1], dtype=src.dtype, device=src.device
    )
    out.scatter_add_(0, index.unsqueeze(1).expand_as(src), src)
    return out


def scatter_softmax_native(src, index, dim: int = 0):
    """Native torch fallback: numerically-stable softmax of `src` (shape
    (N, C)) over rows that share an `index`. Returns per-row weights of the
    same shape as `src`."""
    assert dim == 0, "scatter_softmax_native only supports dim=0"
    dim_size = int(index.max().item()) + 1
    # Per-group max for numerical stability
    group_max = torch.full(
        (dim_size, src.shape[1]),
        float("-inf"),
        dtype=src.dtype,
        device=src.device,
    )
    group_max.scatter_reduce_(
        0,
        index.unsqueeze(1).expand_as(src),
        src,
        reduce="amax",
        include_self=True,
    )
    src_shifted = src - group_max[index]
    exp_src = src_shifted.exp()
    denom = scatter_add_native(exp_src, index, dim=0, dim_size=dim_size)
    return exp_src / denom[index].clamp(min=1e-20)


def softmax_weighted_aggregate(
    values: torch.Tensor,
    index: torch.Tensor,
    score: torch.Tensor,
    dim_size: int = None,
) -> torch.Tensor:
    """Aggregate `values` (N, C) over the groups defined by `index` (N,) using
    softmax weights computed from `score` (N, 1). Returns (G, C) where
    G = dim_size (or index.max()+1).

    Falls back to a pure-torch implementation when torch_scatter is missing."""
    if dim_size is None:
        dim_size = int(index.max().item()) + 1
    if TORCH_SCATTER_AVAIL:
        weight = scatter_softmax(score, index, dim=0)  # (N, 1)
        return scatter_add(values * weight, index, dim=0, dim_size=dim_size)
    weight = scatter_softmax_native(score, index, dim=0)  # (N, 1)
    return scatter_add_native(values * weight, index, dim=0, dim_size=dim_size)


class GeGLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = torch.chunk(x, 2, dim=-1)
        return x * torch.nn.functional.gelu(gate)


class SwiGLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = torch.chunk(x, 2, dim=-1)
        return x * torch.nn.functional.silu(gate)


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
           - learned softmax-weighted aggregation across nodes that share a
             global ID (the within-rank "redistribution" step, load-bearing
             for consistency)
           - halo swap with learned softmax-weighted aggregation across rank
             boundaries (when SIZE>1 and halo_swap_mode != "none")
           - residual + MLP
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

        # Redistribution gates. Each scores a node's "trustworthiness" so the
        # softmax-weighted aggregation can prefer one coincident copy over
        # another instead of always averaging (which low-pass-filters the
        # signal across element / rank boundaries).
        # - redist_gate_intra: scores attn_output prior to within-rank
        #   coincident-node aggregation.
        # - redist_gate_inter: scores post-attention node features prior to
        #   halo-boundary aggregation across ranks.
        self.redist_gate_intra = nn.Linear(hidden_channels, 1, bias=False)
        self.redist_gate_inter = nn.Linear(hidden_channels, 1, bias=False)

        # MLP
        self.norm2 = nn.LayerNorm(hidden_channels)
        self.mlp = MlpBlock(
            hidden_channels, int(hidden_channels * mlp_ratio), hidden_channels
        )

    def _attention_pre_mlp(
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
        """Run attention + intra-rank redistribution + halo swap + inter-rank
        redistribution on a single batch slice, stopping BEFORE the MLP
        residual. Returns x_new in the reduced layout (size N_reduced + halo
        slots, where halo slots have been refreshed by the halo swap when
        SIZE>1). The MLP tail lives in ``_post_mlp``; the full local block is
        ``_attention_with_consistency`` and composes both halves.

        Split into pre/post-MLP halves so the hierarchical attention layer can
        insert a global summary update between the local attention and the MLP
        without reimplementing the (subtle) consistency logic.
        """
        poly_order = self.poly_order
        nodes_per_element = (poly_order + 1) ** 3
        num_elements = idx_reduced2full.shape[0] // nodes_per_element

        res = x_batch

        # Expand reduced -> full
        x_full = x_batch[idx_reduced2full]
        pos_full = pos[idx_reduced2full]

        # Reshape so num_elements acts as the batch dim, nodes_per_element as seq len
        x_full = x_full.reshape(
            num_elements, nodes_per_element, x_batch.shape[-1]
        )
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

        # Redistribution: aggregate attention output across nodes that share a
        # global ID, so coincident physical nodes carry identical values. Each
        # coincident copy gets a learned softmax weight (instead of an
        # unweighted mean) so the model can preserve signed high-frequency
        # content across element boundaries instead of cancelling it.
        ne, np_per, c = attn_output.shape
        attn_output = attn_output.reshape(ne * np_per, c)
        score_intra = self.redist_gate_intra(attn_output)  # (N_full, 1)
        attn_output = softmax_weighted_aggregate(
            attn_output, grouping_index, score_intra
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

            # Aggregate contributions from neighboring ranks via learned
            # softmax-weighted mean instead of an unweighted count-mean. Mirrors
            # the within-rank redistribution: the gate scores each contributor
            # so the model can preserve signed high-frequency content across
            # rank boundaries instead of cancelling it.
            #
            # For receive-slot i the contributors are {i itself} ∪
            # {idx_send[k] : idx_recv[k] == i}. We materialize this as one
            # synthetic group per row of x_new by concatenating the self rows
            # with the sent rows (group = [arange(N), idx_recv]) and running a
            # single softmax-weighted scatter. Interior nodes (no incoming
            # contributions) have a singleton group, softmax weight 1, so
            # their value is unchanged -- identical to the old count-mean for
            # interior slots.
            idx_recv = halo_info[:, 0]
            idx_send = halo_info[:, 1]
            N = x_new.size(0)
            score_self = self.redist_gate_inter(x_new)  # (N, 1)
            values_in = x_new.index_select(0, idx_send)  # (H, C)
            score_in = score_self.index_select(0, idx_send)  # (H, 1)
            values_concat = torch.cat([x_new, values_in], dim=0)
            score_concat = torch.cat([score_self, score_in], dim=0)
            group_concat = torch.cat(
                [
                    torch.arange(N, device=x_new.device, dtype=idx_recv.dtype),
                    idx_recv,
                ],
                dim=0,
            )
            x_new = softmax_weighted_aggregate(
                values_concat, group_concat, score_concat, dim_size=N
            )
        else:
            x_new = res + attn_output

        return x_new

    def _post_mlp(self, x_new: torch.Tensor) -> torch.Tensor:
        """Residual MLP tail. Counterpart to ``_attention_pre_mlp``."""
        y = self.norm2(x_new)
        y = self.mlp(y)
        return x_new + y

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
        """Local-only path: pre-MLP attention block + MLP residual. Preserved
        as a single-call entry point for the non-hierarchical processor stack.
        Bit-equivalent to the pre-refactor implementation.
        """
        x_new = self._attention_pre_mlp(
            x_batch,
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
        return self._post_mlp(x_new)

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
            batch = torch.zeros(x.size(0), device=x.device, dtype=torch.long)
        batch_size = int(torch.max(batch).item()) + 1

        # Inject diffusion-step embedding (out-of-place)
        if self.emb_features > 0:
            x = x + self.node_emb_linear(emb)[batch]

        # Per-batch attention + redistribution + halo swap + MLP. The graph
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
            hidden_channels (int): attention/MLP hidden width
            n_transformer_layers (int): number of attention blocks
            num_heads (int): heads in multi-head attention
            poly_order (int): spectral element polynomial order (nodes/element = (p+1)^3)
            emb_width (int): width of the diffusion-step embedding
            halo_swap_mode (str): one of {none, all_to_all, all_to_all_opt}
            learnable_variance (bool): if True the decoder outputs 2x input_node_features
            mlp_ratio (float): MLP hidden ratio
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

        # ~~~~ Processor: stack of DGTAttentionBlock, optionally wrapped in
        # HierarchicalLayer when arch["hierarchical_attention"] is true.
        # Lazy import so the non-hierarchical path doesn't pay the cost.
        if self.hierarchical_attention:
            from hierarchical import HierarchicalLayer
        self.processor = nn.ModuleList()
        for ilayer in range(self.n_transformer_layers):
            inner = DGTAttentionBlock(
                hidden_channels=self.hidden_channels,
                num_heads=self.num_heads,
                emb_features=emb_width,
                poly_order=self.poly_order,
                mlp_ratio=self.mlp_ratio,
                halo_swap_mode=self.halo_swap_mode,
            )
            if (
                self.hierarchical_attention and 
                (ilayer + 1) % self.hierarchical_interleve_freq == 0
            ) :
                self.processor.append(
                    HierarchicalLayer(
                        inner_block=inner,
                        hidden_channels=self.hidden_channels,
                        num_heads=self.num_heads,
                        k_summary=self.k_summary,
                        poly_order=self.poly_order,
                        readout_chunk_size=self.readout_chunk_size,
                        activation_checkpointing=self.activation_checkpointing,
                    )
                )
            else:
                self.processor.append(inner)

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
        self.hierarchical_attention = arch.get("hierarchical_attention", False)
        self.hierarchical_interleve_freq = arch.get("hierarchical_interleve_freq", 2)
        self.k_summary = arch.get("k_summary", 4)
        # Cap on the SummaryReadout attention matrix per SDPA call (in element
        # count). Picked so the per-chunk attention buffer (chunk * h * np * n_kv)
        # is bounded to the same scale as the local DGT block; small enough
        # to keep Intel IPEX's SDPA fallback from page-faulting on the full
        # readout in one shot.
        self.readout_chunk_size = arch.get("readout_chunk_size", 16)
        # When True, each HierarchicalLayer wraps its per-batch sequence with
        # torch.utils.checkpoint so activations from completed batches do not
        # accumulate. Strongly recommended at batch_size > 1 because the
        # per-batch loop otherwise pins ~8x the readout's attention buffers
        # in the autograd graph until backward runs.
        self.activation_checkpointing = arch.get(
            "activation_checkpointing", False
        )
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
        if self.arch.get("hierarchical_attention", False):
            header += f"_hier{self.arch.get('k_summary', 4)}"
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
