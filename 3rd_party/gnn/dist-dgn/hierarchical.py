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
from torch.utils.checkpoint import checkpoint as activation_checkpoint

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


# Caps the per-call SummaryReadout attention buffer at
# ~chunk * h * np * n_kv * 4 bytes. For the ext_cyl_dgn case (h=4, np=64,
# n_kv ~6k) that's ~6 MB * chunk per SDPA launch. With activation
# checkpointing on, peak memory is bounded by one batch's intermediates so a
# bigger chunk safely cuts the number of SDPA launches (and their per-launch
# overhead, which dominates at chunk=4). Lower if you see OOM, raise if SDPA
# launch overhead dominates.
DEFAULT_READOUT_CHUNK_SIZE = 16


class SummaryReadout(nn.Module):
    """Cross-attention from local nodes (queries) to the full set of gathered
    summary tokens (keys/values). The output is a per-node delta that the
    caller adds to the local node features.

    Memory note: the attention matrix has shape
    ``(num_elements_local, num_heads, nodes_per_element, n_kv)`` where
    ``n_kv = num_elements_total * k_summary``. For the ext_cyl_dgn case
    (~384 elements/rank, world_size=4, k=4, np=64) that's ~2.4 GB if computed
    in a single SDPA call. On Intel GPU's IPEX SDPA fallback (no xetla
    available) this either fails to allocate or produces a GPU page fault.
    To keep peak memory in line with the local DGT block (~100 MB
    attention matrix), we keep queries in per-element shape and process
    them in chunks of ``query_chunk_size`` elements via SDPA broadcasting
    on the leading dim.

    Args:
        hidden_channels: input/output feature width.
        num_heads: attention heads.
        use_bias: matches DGTAttentionBlock convention.
        query_chunk_size: number of query elements processed per SDPA call.
            Lower values trade speed for peak memory; default 16 caps each
            attention matrix at roughly ``chunk * h * np * n_kv * 4 bytes``.
    """

    def __init__(
        self,
        hidden_channels: int,
        num_heads: int,
        use_bias: bool = False,
        query_chunk_size: int = DEFAULT_READOUT_CHUNK_SIZE,
    ):
        super().__init__()
        assert hidden_channels % num_heads == 0
        assert query_chunk_size >= 1
        self.hidden_channels = hidden_channels
        self.num_heads = num_heads
        self.query_chunk_size = query_chunk_size

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
                slots are masked out in the attention via a boolean SDPA mask.

        Returns:
            (num_elements_local, nodes_per_element, C) per-node delta.
        """
        ne_local, np_per, c = nodes.shape
        ne_total, k, _ = summary.shape
        dim = summary_centroids.shape[-1]
        n_kv = ne_total * k
        h = self.num_heads
        c_per_head = c // h

        # Keep queries in per-element shape (ne_local, np_per, C). Reshape
        # only the keys/values, which become a single shared (1, h, n_kv, c)
        # sequence broadcast across the element dim of the queries.
        q_in = self.norm_q(nodes)                       # (ne_local, np, C)
        kv_in = self.norm_kv(summary.reshape(n_kv, c))   # (n_kv, C)

        # Per-element query projection then split heads.
        q = self.q_proj(q_in)                            # (ne_local, np, C)
        q = q.reshape(ne_local, np_per, h, c_per_head).permute(0, 2, 1, 3)
        # q: (ne_local, h, np_per, c_per_head)
        q = apply_rope(q, node_pos_norm)

        # Shared keys/values; placeholder element dim of 1 for SDPA + apply_rope.
        kk = self.k_proj(kv_in).reshape(n_kv, h, c_per_head).permute(1, 0, 2).unsqueeze(0)
        v = self.v_proj(kv_in).reshape(n_kv, h, c_per_head).permute(1, 0, 2).unsqueeze(0)
        # Each summary token sits at its element's centroid.
        k_pos = (
            summary_centroids.unsqueeze(1).expand(ne_total, k, dim).reshape(n_kv, dim)
        )
        kk = apply_rope(kk, k_pos.unsqueeze(0))

        # Boolean key-padding mask: True = attend, False = mask out. Built
        # once and broadcast across (chunk, h, np_per) on every SDPA call.
        attn_mask = None
        if key_padding_mask is not None:
            mask_per_token = (
                key_padding_mask.unsqueeze(1).expand(ne_total, k).reshape(n_kv)
            )
            attn_mask = mask_per_token.to(torch.bool).view(1, 1, 1, n_kv)

        # Process queries in chunks to bound peak attention-matrix memory.
        # Each chunk's attention is (chunk, h, np_per, n_kv) -- IPEX's SDPA
        # fallback on Intel GPU page-faults on the unchunked variant
        # (chunk == ne_local), which would be ~2.4 GB for ext_cyl_dgn.
        chunk_size = self.query_chunk_size
        if chunk_size >= ne_local:
            out = sdpa(q, kk, v, attn_mask=attn_mask)
        else:
            out_chunks = []
            for s in range(0, ne_local, chunk_size):
                e = min(s + chunk_size, ne_local)
                out_chunks.append(sdpa(q[s:e], kk, v, attn_mask=attn_mask))
            out = torch.cat(out_chunks, dim=0)
        # out: (ne_local, h, np_per, c_per_head) -> (ne_local, np_per, C)
        out = out.permute(0, 2, 1, 3).reshape(ne_local, np_per, c)
        out = self.o_proj(out)
        return out


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
        readout_chunk_size: int = DEFAULT_READOUT_CHUNK_SIZE,
        activation_checkpointing: bool = False,
    ):
        super().__init__()
        self.inner = inner_block
        self.k_summary = k_summary
        self.poly_order = poly_order
        self.hidden_channels = hidden_channels
        self.num_heads = num_heads
        # When True, the per-batch (pre_ffn -> global -> post_ffn) sequence is
        # wrapped with torch.utils.checkpoint so activations from completed
        # batches do not accumulate in the autograd graph. This is essential
        # at batch_size > 1: without it, the per-batch loop pins the readout's
        # attention buffers for ALL batches simultaneously, which OOMs at the
        # cumulative scale (8 batches x 4 layers x ~24 chunks per readout).
        self.activation_checkpointing = activation_checkpointing

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
            query_chunk_size=readout_chunk_size,
        )

    def _compute_global_static(
        self,
        pos: torch.Tensor,
        pos_min: torch.Tensor,
        pos_max: torch.Tensor,
        idx_reduced2full: torch.Tensor,
        SIZE: int,
    ):
        """Precompute everything in the global path that does NOT depend on
        the per-batch features (x): per-node normalized positions, per-element
        centroids, the int-count all_gather, the centroid all_gather, the
        key-padding mask. These are graph-static -- identical across every
        batch element and every forward call for a given graph topology -- so
        running them once per forward saves SIZE * batch_size * n_layers worth
        of collectives that the old per-batch path would otherwise do.
        """
        device = pos.device
        nodes_per_element = (self.poly_order + 1) ** 3
        n_full = idx_reduced2full.shape[0]
        ne_local = n_full // nodes_per_element
        dim = pos.shape[-1]

        pos_full = pos[idx_reduced2full]
        pos_per_elem = pos_full.reshape(ne_local, nodes_per_element, dim)
        denom = (pos_max - pos_min).clamp(min=1e-12)
        node_pos_norm = (pos_per_elem - pos_min) / denom
        centroids_norm = (pos_per_elem.mean(dim=1) - pos_min) / denom

        # Int-count gather (non-autograd; int tensor).
        if SIZE > 1 and dist.is_initialized():
            ne_local_t = torch.tensor([ne_local], device=device, dtype=torch.long)
            counts_list = [
                torch.zeros(1, device=device, dtype=torch.long)
                for _ in range(SIZE)
            ]
            dist.all_gather(counts_list, ne_local_t)
            counts = torch.cat(counts_list, dim=0)  # (SIZE,)
        else:
            counts = torch.tensor([ne_local], device=device, dtype=torch.long)
        max_ne = int(counts.max().item())

        # Pad centroids; gather. Centroids are constants (no requires_grad),
        # so we can use the plain (non-autograd) collective.
        if max_ne > ne_local:
            pad_c = torch.zeros(
                max_ne - ne_local, dim,
                dtype=centroids_norm.dtype, device=device,
            )
            centroids_padded = torch.cat([centroids_norm, pad_c], dim=0)
        else:
            centroids_padded = centroids_norm

        if SIZE > 1 and dist.is_initialized():
            centroids_padded = centroids_padded.contiguous()
            gather_list = [
                torch.zeros_like(centroids_padded) for _ in range(SIZE)
            ]
            dist.all_gather(gather_list, centroids_padded)
            centroids_all = torch.stack(gather_list, dim=0).reshape(
                SIZE * max_ne, dim
            )
        else:
            centroids_all = centroids_padded

        arange = torch.arange(max_ne, device=device).unsqueeze(0)
        valid = arange < counts.unsqueeze(1)
        key_padding_mask = valid.reshape(SIZE * max_ne)

        return {
            "nodes_per_element": nodes_per_element,
            "ne_local": ne_local,
            "max_ne": max_ne,
            "node_pos_norm": node_pos_norm,
            "centroids_norm": centroids_norm,
            "centroids_all": centroids_all,
            "key_padding_mask": key_padding_mask,
        }

    def _global_summary_for_batch(
        self,
        x_post_attn: torch.Tensor,
        idx_reduced2full: torch.Tensor,
        SIZE: int,
        static: dict,
    ) -> torch.Tensor:
        """Per-batch part of the global attention path: pool nodes, gather
        summaries across ranks, run the readout cross-attention, scatter the
        per-node delta back to the reduced layout. All graph-static quantities
        (centroids, gathered centroids, mask, max_ne) live in ``static`` and
        are precomputed once per forward by :meth:`_compute_global_static`.
        """
        ne_local = static["ne_local"]
        nodes_per_element = static["nodes_per_element"]
        max_ne = static["max_ne"]
        node_pos_norm = static["node_pos_norm"]
        centroids_norm = static["centroids_norm"]
        centroids_all = static["centroids_all"]
        key_padding_mask = static["key_padding_mask"]
        device = x_post_attn.device

        # Reduced -> full view, then reshape to (ne_local, np, C).
        x_full = x_post_attn[idx_reduced2full]
        nodes_per_elem = x_full.reshape(
            ne_local, nodes_per_element, x_post_attn.shape[-1]
        )

        # Per-element pool -> (ne_local, k, C). Only collective per batch
        # that actually depends on x.
        summary_local = self.pool(nodes_per_elem, centroids_norm, node_pos_norm)

        if max_ne > ne_local:
            pad_summary = torch.zeros(
                max_ne - ne_local, self.k_summary, self.hidden_channels,
                dtype=summary_local.dtype, device=device,
            )
            summary_local_padded = torch.cat(
                [summary_local, pad_summary], dim=0
            )
        else:
            summary_local_padded = summary_local

        # Autograd-aware gather. Force contiguous: oneCCL / xccl is strict
        # about send-buffer layout and silently faults on non-contiguous
        # inputs in some builds.
        summary_all = _autograd_all_gather(
            summary_local_padded.contiguous(), SIZE
        )
        summary_all = summary_all.reshape(
            SIZE * max_ne, self.k_summary, self.hidden_channels
        )

        delta_per_elem = self.readout(
            summary_all,
            centroids_all,
            nodes_per_elem,
            node_pos_norm,
            key_padding_mask=key_padding_mask,
        )

        # (ne_local, np, C) -> (n_full, C) -> reduced layout (N_reduced + halo, C).
        n_full = ne_local * nodes_per_element
        delta_full = delta_per_elem.reshape(n_full, self.hidden_channels)
        delta_reduced = torch.zeros_like(x_post_attn)
        counts_per_red = torch.zeros(
            x_post_attn.shape[0], 1,
            dtype=delta_full.dtype, device=device,
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

        # Precompute graph-static global-path quantities (centroids, mask,
        # max_ne, centroid all_gather). These do not depend on x, so doing
        # them ONCE per forward instead of once per batch element saves
        # batch_size collectives per layer per forward.
        static = self._compute_global_static(
            pos, pos_min, pos_max, idx_reduced2full, SIZE
        )

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
            delta = self._global_summary_for_batch(
                x_post_attn,
                idx_reduced2full,
                SIZE,
                static,
            )
            x_with_global = x_post_attn + delta
            return self.inner._post_ffn(x_with_global)

        def call_per_batch(x_b: torch.Tensor) -> torch.Tensor:
            if self.activation_checkpointing and self.training:
                return activation_checkpoint(per_batch, x_b, use_reentrant=False)
            return per_batch(x_b)

        if batch_size == 1:
            return call_per_batch(x)

        out = torch.empty_like(x)
        for b in range(batch_size):
            mask = batch == b
            out[mask] = call_per_batch(x[mask])
        return out
