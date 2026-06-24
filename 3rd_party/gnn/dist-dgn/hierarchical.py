"""
Hierarchical attention modules for the Distributed Diffusion Graph Transformer.

Implements the perceiver-style pooling, distributed cross-rank summary
attention, and cross-attention readout described in Appendix A of "A Consistent
Transformer for Mesh-Based Data" (ML4PS @ NeurIPS 2025).

Three modules:
  - PerceiverPool: per-element cross-attention from k learned latent queries to
    that element's nodes. Produces k summary tokens per element.
  - SummaryReadout: per-node cross-attention from local nodes (queries) to the
    full set of gathered summary tokens (keys/values). Returns a per-node delta
    to be added to the local node features.
  - HierarchicalLayer: wraps one DGTAttentionBlock with the perceiver pool +
    cross-rank all_gather + summary readout, inserting the global update
    between the local block's attention and FFN.

Summary tokens are allocated fresh per layer. Backend for the cross-rank
gather is `torch.distributed.nn.functional.all_gather`, which is autograd-aware
(vanilla `torch.distributed.all_gather` silently drops cross-rank gradients).
Padding to `max(num_elements_per_rank) * k_summary` with an additive key
padding mask handles uneven element counts across ranks.
"""

from typing import Optional

import einops
import torch
import torch.distributed as dist
# torch.distributed.nn exposes autograd-aware collectives (all_gather etc.).
# graph_transformer.py uses the same alias for all_to_all in the halo swap.
import torch.distributed.nn as distnn
import torch.nn as nn
from torch.nn.functional import scaled_dot_product_attention as sdpa

# Import apply_rope from the dependency-free helper module so the unit tests
# for PerceiverPool / SummaryReadout don't transitively pull in gnn.py (which
# requires torch_geometric).
from attn_utils import apply_rope


def _split_heads(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    """(*, n, c) -> (*, h, n, c_per_head)."""
    return einops.rearrange(x, "... n (h c) -> ... h n c", h=num_heads)


def _merge_heads(x: torch.Tensor) -> torch.Tensor:
    """(*, h, n, c_per_head) -> (*, n, h*c_per_head)."""
    return einops.rearrange(x, "... h n c -> ... n (h c)")


class PerceiverPool(nn.Module):
    """Pool the nodes within each element into ``k_summary`` learned summary
    tokens via cross-attention.

    Args:
        hidden_channels: input/output feature width (matches DGTAttentionBlock).
        num_heads: attention heads.
        k_summary: number of learned latent query tokens per element.
        use_bias: matches DGTAttentionBlock convention (default False).

    Shape contract:
        forward(nodes, centroids_norm, node_pos_norm) ->
            summary tensor of shape (num_elements, k_summary, hidden_channels)
    """

    def __init__(
        self,
        hidden_channels: int,
        num_heads: int,
        k_summary: int,
        use_bias: bool = False,
    ):
        super().__init__()
        assert hidden_channels % num_heads == 0
        self.hidden_channels = hidden_channels
        self.num_heads = num_heads
        self.k_summary = k_summary

        # Learnable latent queries, one set per layer instance, broadcast across
        # elements at forward time.
        self.latents = nn.Parameter(torch.empty(k_summary, hidden_channels))
        nn.init.xavier_uniform_(self.latents)

        # Pre-norm on queries and on keys/values separately (cross-attention).
        self.norm_q = nn.LayerNorm(hidden_channels)
        self.norm_kv = nn.LayerNorm(hidden_channels)

        self.q_proj = nn.Linear(hidden_channels, hidden_channels, bias=use_bias)
        self.k_proj = nn.Linear(hidden_channels, hidden_channels, bias=use_bias)
        self.v_proj = nn.Linear(hidden_channels, hidden_channels, bias=use_bias)
        self.o_proj = nn.Linear(hidden_channels, hidden_channels, bias=use_bias)

        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.o_proj.weight)

    def forward(
        self,
        nodes: torch.Tensor,
        centroids_norm: torch.Tensor,
        node_pos_norm: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            nodes: (num_elements, nodes_per_element, C). Per-element node
                features (keys/values).
            centroids_norm: (num_elements, dim). Normalized element centroids
                (used as the query position for RoPE; all k latent tokens of an
                element share its centroid).
            node_pos_norm: (num_elements, nodes_per_element, dim). Normalized
                node positions (used as the key position for RoPE).

        Returns:
            (num_elements, k_summary, C). Summary tokens per element.
        """
        ne, np_per, c = nodes.shape
        k = self.k_summary
        dim = centroids_norm.shape[-1]

        # Broadcast learned latents across the element dimension.
        latents = self.latents.unsqueeze(0).expand(ne, k, c)

        # Query positions: replicate the centroid across all k latent slots.
        latent_pos = centroids_norm.unsqueeze(1).expand(ne, k, dim)

        q_in = self.norm_q(latents)
        kv_in = self.norm_kv(nodes)

        q = _split_heads(self.q_proj(q_in), self.num_heads)
        kk = _split_heads(self.k_proj(kv_in), self.num_heads)
        v = _split_heads(self.v_proj(kv_in), self.num_heads)

        q = apply_rope(q, latent_pos)
        kk = apply_rope(kk, node_pos_norm)

        out = sdpa(q, kk, v)
        out = _merge_heads(out)
        out = self.o_proj(out)
        return out


class SummaryReadout(nn.Module):
    """Cross-attention from local nodes (queries) to the full set of gathered
    summary tokens (keys/values). The output is a per-node delta that the
    caller adds to the local node features.

    Args:
        hidden_channels: input/output feature width.
        num_heads: attention heads.
        use_bias: matches DGTAttentionBlock convention.

    Shape contract:
        forward(summary, summary_centroids, nodes, node_pos_norm, mask) ->
            tensor of shape (num_elements_local, nodes_per_element,
            hidden_channels) -- a per-node delta.
    """

    def __init__(
        self,
        hidden_channels: int,
        num_heads: int,
        use_bias: bool = False,
    ):
        super().__init__()
        assert hidden_channels % num_heads == 0
        self.hidden_channels = hidden_channels
        self.num_heads = num_heads

        self.norm_q = nn.LayerNorm(hidden_channels)
        self.norm_kv = nn.LayerNorm(hidden_channels)

        self.q_proj = nn.Linear(hidden_channels, hidden_channels, bias=use_bias)
        self.k_proj = nn.Linear(hidden_channels, hidden_channels, bias=use_bias)
        self.v_proj = nn.Linear(hidden_channels, hidden_channels, bias=use_bias)
        self.o_proj = nn.Linear(hidden_channels, hidden_channels, bias=use_bias)

        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.o_proj.weight)

    def forward(
        self,
        summary: torch.Tensor,
        summary_centroids: torch.Tensor,
        nodes: torch.Tensor,
        node_pos_norm: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            summary: (num_elements_total, k_summary, C). Summary tokens from
                ALL ranks (already gathered, padded if applicable).
            summary_centroids: (num_elements_total, dim). Normalized element
                centroids matching ``summary`` (also padded).
            nodes: (num_elements_local, nodes_per_element, C). Local node
                features (queries).
            node_pos_norm: (num_elements_local, nodes_per_element, dim).
                Normalized local node positions.
            key_padding_mask: optional (num_elements_total,) boolean tensor.
                True = real element, False = padded slot. When provided, padded
                slots receive ``-inf`` in the attention logits so they make no
                contribution.

        Returns:
            (num_elements_local, nodes_per_element, C) per-node delta.
        """
        ne_local, np_per, c = nodes.shape
        ne_total, k, _ = summary.shape
        dim = summary_centroids.shape[-1]
        n_kv = ne_total * k
        n_q = ne_local * np_per

        # Flatten queries and keys into a single attention pass.
        q_flat = nodes.reshape(n_q, c)
        q_pos = node_pos_norm.reshape(n_q, dim)

        kv_flat = summary.reshape(n_kv, c)
        # Each summary token sits at its element's centroid.
        k_pos = summary_centroids.unsqueeze(1).expand(ne_total, k, dim).reshape(
            n_kv, dim
        )

        q_in = self.norm_q(q_flat)
        kv_in = self.norm_kv(kv_flat)

        # Project, then reshape to (1, h, N, c_per_head) so apply_rope and sdpa
        # treat this as a single batch with N tokens. The first dim is a
        # placeholder "element" dimension required by apply_rope's RoPE
        # broadcasting (see graph_transformer.apply_rope:144).
        h = self.num_heads
        c_per_head = c // h
        q = self.q_proj(q_in).reshape(n_q, h, c_per_head).permute(1, 0, 2).unsqueeze(0)
        kk = self.k_proj(kv_in).reshape(n_kv, h, c_per_head).permute(1, 0, 2).unsqueeze(0)
        v = self.v_proj(kv_in).reshape(n_kv, h, c_per_head).permute(1, 0, 2).unsqueeze(0)

        q = apply_rope(q, q_pos.unsqueeze(0))
        kk = apply_rope(kk, k_pos.unsqueeze(0))

        attn_mask = None
        if key_padding_mask is not None:
            # Expand the per-element mask to per-token, build an additive
            # (q_len, kv_len)-shaped mask, broadcast across batch and heads.
            mask_per_token = (
                key_padding_mask.unsqueeze(1).expand(ne_total, k).reshape(n_kv)
            )
            attn_mask = torch.zeros(
                (1, 1, 1, n_kv), dtype=q.dtype, device=q.device
            )
            attn_mask = attn_mask.masked_fill(
                ~mask_per_token.view(1, 1, 1, n_kv), float("-inf")
            )

        out = sdpa(q, kk, v, attn_mask=attn_mask)  # (1, h, n_q, c_per_head)
        out = out.squeeze(0).permute(1, 0, 2).reshape(n_q, c)
        out = self.o_proj(out)
        return out.reshape(ne_local, np_per, c)


def _autograd_all_gather(local: torch.Tensor, world_size: int) -> torch.Tensor:
    """Autograd-aware all_gather.

    Stacks the per-rank tensors along a new leading dim of size world_size.
    Uses ``torch.distributed.nn.all_gather`` (autograd-aware) so gradients
    flow back via reduce-scatter on the backward pass. Vanilla
    ``torch.distributed.all_gather`` silently drops cross-rank gradients.

    When DDP is unavailable (single-process / no process group), returns
    ``local.unsqueeze(0)``.
    """
    if world_size == 1 or not dist.is_initialized():
        return local.unsqueeze(0)
    gathered = distnn.all_gather(local)
    return torch.stack(list(gathered), dim=0)


class HierarchicalLayer(nn.Module):
    """One hierarchical attention layer: wraps a DGTAttentionBlock with a
    perceiver pool + cross-rank summary attention + readout. The global update
    is inserted between the inner block's attention and FFN.

    Sequence per forward call (per batch element):
      1. Run inner block's ``_attention_pre_ffn`` (local element attention +
         intra/inter redistribution). Returns the post-attention nodes (still
         in reduced-graph layout) along with the (num_elements, nodes_per_elem,
         C) view needed for pooling.
      2. Compute per-element centroids from ``pos[idx_reduced2full]``.
      3. ``PerceiverPool`` -> local summary tokens.
      4. Autograd-aware all_gather of summary tokens and centroids, padded to
         the max element count across ranks; build a key-padding mask.
      5. ``SummaryReadout`` over the global summary set -> per-node delta.
      6. Add the delta to the post-attention nodes.
      7. Run inner block's ``_post_ffn`` -> final output.
    """

    def __init__(
        self,
        inner_block,
        hidden_channels: int,
        num_heads: int,
        k_summary: int,
        poly_order: int,
        use_bias: bool = False,
    ):
        super().__init__()
        self.inner = inner_block
        self.k_summary = k_summary
        self.poly_order = poly_order
        self.hidden_channels = hidden_channels
        self.num_heads = num_heads

        self.pool = PerceiverPool(
            hidden_channels=hidden_channels,
            num_heads=num_heads,
            k_summary=k_summary,
            use_bias=use_bias,
        )
        self.readout = SummaryReadout(
            hidden_channels=hidden_channels,
            num_heads=num_heads,
            use_bias=use_bias,
        )

    def _global_attn_for_batch(
        self,
        x_post_attn: torch.Tensor,  # (N_reduced, C)
        pos: torch.Tensor,
        pos_min: torch.Tensor,
        pos_max: torch.Tensor,
        idx_reduced2full: torch.Tensor,
        SIZE: int,
    ) -> torch.Tensor:
        """Compute the per-node delta from the global summary path for a
        single batch slice. Returns a tensor of shape (N_reduced, C).
        """
        # Determine local element geometry.
        nodes_per_element = (self.poly_order + 1) ** 3
        n_full = idx_reduced2full.shape[0]
        ne_local = n_full // nodes_per_element

        # Reduced -> full view, then reshape to (ne_local, np, C).
        x_full = x_post_attn[idx_reduced2full]
        nodes_per_elem = x_full.reshape(
            ne_local, nodes_per_element, x_post_attn.shape[-1]
        )

        # Per-node positions, normalized via global bounds.
        pos_full = pos[idx_reduced2full]
        pos_per_elem = pos_full.reshape(
            ne_local, nodes_per_element, pos.shape[-1]
        )
        denom = (pos_max - pos_min).clamp(min=1e-12)
        node_pos_norm = (pos_per_elem - pos_min) / denom
        centroids = pos_per_elem.mean(dim=1)  # (ne_local, dim)
        centroids_norm = (centroids - pos_min) / denom

        # Per-element pool -> (ne_local, k, C).
        summary_local = self.pool(nodes_per_elem, centroids_norm, node_pos_norm)

        # Gather per-rank element counts so every rank can pad to the max.
        device = summary_local.device
        if SIZE > 1 and dist.is_initialized():
            ne_local_t = torch.tensor([ne_local], device=device, dtype=torch.long)
            counts = [
                torch.zeros(1, device=device, dtype=torch.long)
                for _ in range(SIZE)
            ]
            dist.all_gather(counts, ne_local_t)
            counts = torch.cat(counts, dim=0)  # (SIZE,)
        else:
            counts = torch.tensor(
                [ne_local], device=device, dtype=torch.long
            )

        max_ne = int(counts.max().item())

        # Pad local summary and centroids up to max_ne along the element dim.
        if max_ne > ne_local:
            pad_summary = torch.zeros(
                max_ne - ne_local,
                self.k_summary,
                self.hidden_channels,
                dtype=summary_local.dtype,
                device=device,
            )
            summary_local_padded = torch.cat(
                [summary_local, pad_summary], dim=0
            )
            pad_centroids = torch.zeros(
                max_ne - ne_local,
                centroids_norm.shape[-1],
                dtype=centroids_norm.dtype,
                device=device,
            )
            centroids_local_padded = torch.cat(
                [centroids_norm, pad_centroids], dim=0
            )
        else:
            summary_local_padded = summary_local
            centroids_local_padded = centroids_norm

        # Autograd-aware gather of padded summary + centroids.
        summary_all = _autograd_all_gather(summary_local_padded, SIZE)
        # (SIZE, max_ne, k, C) -> (SIZE * max_ne, k, C)
        summary_all = summary_all.reshape(
            SIZE * max_ne, self.k_summary, self.hidden_channels
        )

        centroids_all = _autograd_all_gather(centroids_local_padded, SIZE)
        centroids_all = centroids_all.reshape(
            SIZE * max_ne, centroids_norm.shape[-1]
        )

        # Build a per-element validity mask in the global order
        # [rank0_e0..rank0_e_max-1, rank1_e0..rank1_e_max-1, ...].
        arange = torch.arange(max_ne, device=device).unsqueeze(0)  # (1, max_ne)
        valid = arange < counts.unsqueeze(1)  # (SIZE, max_ne)
        key_padding_mask = valid.reshape(SIZE * max_ne)

        # Cross-attend local nodes to all gathered summary tokens.
        delta_per_elem = self.readout(
            summary_all,
            centroids_all,
            nodes_per_elem,
            node_pos_norm,
            key_padding_mask=key_padding_mask,
        )

        # (ne_local, np, C) -> (n_full, C) -> (N_reduced, C).
        delta_full = delta_per_elem.reshape(n_full, self.hidden_channels)
        # Scatter the full -> reduced. Use mean over coincident copies for
        # consistency (analogous to the intra-rank redistribution in the inner
        # block). Coincident copies should already be equal because the
        # readout's output for two nodes at the same physical position uses the
        # same query position and the same gathered summary set.
        delta_reduced = torch.zeros_like(x_post_attn)
        # Reverse-index: place every full row at idx_reduced2full[i] in
        # reduced. Multiple fulls map to the same reduced index; average them.
        counts_per_red = torch.zeros(
            x_post_attn.shape[0],
            1,
            dtype=delta_full.dtype,
            device=device,
        )
        delta_reduced.index_add_(0, idx_reduced2full, delta_full)
        ones = torch.ones(n_full, 1, dtype=delta_full.dtype, device=device)
        counts_per_red.index_add_(0, idx_reduced2full, ones)
        delta_reduced = delta_reduced / counts_per_red.clamp(min=1.0)
        return delta_reduced

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

        # Mirror the inner block's diffusion-step injection (out-of-place).
        if self.inner.emb_features > 0:
            x = x + self.inner.node_emb_linear(emb)[batch]

        def per_batch(x_b: torch.Tensor) -> torch.Tensor:
            x_post_attn = self.inner._attention_pre_ffn(
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
            delta = self._global_attn_for_batch(
                x_post_attn,
                pos,
                pos_min,
                pos_max,
                idx_reduced2full,
                SIZE,
            )
            x_with_global = x_post_attn + delta
            return self.inner._post_ffn(x_with_global)

        if batch_size == 1:
            return per_batch(x)

        out = torch.empty_like(x)
        for b in range(batch_size):
            mask = batch == b
            out[mask] = per_batch(x[mask])
        return out
