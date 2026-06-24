"""Rank-invariance check for the hierarchical attention path.

Has two run modes; the second is the load-bearing one.

1) Single-process pytest (sanity):
        cd 3rd_party/gnn/dist-dgn && pytest tests/test_rank_invariance.py -v

   Runs HierarchicalLayer-free sanity checks on the autograd-aware all_gather
   wrapper (returns local.unsqueeze(0) when no process group is initialized)
   and on a single-process forward + backward through
   ``PerceiverPool -> SummaryReadout``. Does NOT exercise the distributed
   gather path.

2) Under torchrun for the actual rank-invariance check:

        cd 3rd_party/gnn/dist-dgn
        torchrun --nproc_per_node=1 tests/test_rank_invariance.py
        torchrun --nproc_per_node=2 tests/test_rank_invariance.py
        torchrun --nproc_per_node=4 tests/test_rank_invariance.py

   Each invocation partitions a fixed synthetic 8-element graph across the
   world, runs one ``PerceiverPool`` + autograd-aware all_gather +
   ``SummaryReadout`` pass, gathers per-rank outputs to rank 0, and saves
   them to ``/tmp/dgt_hier_invariance_N{world_size}.pt``.

   After running all the world sizes you want to check, compare:

        cd 3rd_party/gnn/dist-dgn
        python tests/test_rank_invariance.py compare

   The comparison asserts each multi-rank output matches the N=1 baseline
   to relative tolerance < 1e-5 (or whatever fp precision the backend
   provides for the additive accumulation).

Backend defaults to ``gloo`` for CPU portability. Use ``nccl`` / ``xccl`` if
running on GPU; results should be the same modulo fp accumulation order.
"""

import os
import sys

import pytest
import torch

# Make the dist-dgn package importable when this script is invoked directly
# (torchrun does not set PYTHONPATH).
_DIST_DGN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _DIST_DGN_DIR not in sys.path:
    sys.path.insert(0, _DIST_DGN_DIR)

from hierarchical import (  # noqa: E402
    PerceiverPool,
    SummaryReadout,
    _autograd_all_gather,
)


# ---------------------------------------------------------------------------
# Single-process pytest sanity (no distributed dependency)
# ---------------------------------------------------------------------------


def test_autograd_all_gather_no_ddp_passthrough():
    """When no process group is initialised, the wrapper must return
    local.unsqueeze(0) so single-process code paths behave."""
    x = torch.randn(3, 4, requires_grad=True)
    y = _autograd_all_gather(x, world_size=1)
    assert y.shape == (1, 3, 4)
    y.sum().backward()
    assert x.grad is not None
    assert torch.allclose(x.grad, torch.ones_like(x))


def test_pool_readout_single_process_end_to_end():
    torch.manual_seed(0)
    pool = PerceiverPool(hidden_channels=16, num_heads=4, k_summary=4)
    readout = SummaryReadout(hidden_channels=16, num_heads=4)

    nodes = torch.randn(5, 8, 16, requires_grad=True)
    centroids = torch.rand(5, 3)
    node_pos = torch.rand(5, 8, 3)

    summary = pool(nodes, centroids, node_pos)
    summary_gathered = _autograd_all_gather(summary, world_size=1).reshape(
        5, 4, 16
    )
    centroids_gathered = _autograd_all_gather(centroids, world_size=1).reshape(
        5, 3
    )
    delta = readout(summary_gathered, centroids_gathered, nodes, node_pos)
    assert delta.shape == nodes.shape
    delta.sum().backward()
    assert nodes.grad is not None and torch.isfinite(nodes.grad).all()


# ---------------------------------------------------------------------------
# torchrun entry point: build full graph, partition, run, save
# ---------------------------------------------------------------------------


# Fixed synthetic problem — same numbers on every torchrun invocation so saved
# outputs are comparable across world sizes.
SEED = 42
HIDDEN = 32
NUM_HEADS = 4
K_SUMMARY = 4
DIM = 3
NE_TOTAL = 8           # divisible by 1, 2, 4 — works for torchrun -n {1,2,4}
NP_PER = 16


def _run_distributed():
    import torch.distributed as dist

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    backend = os.environ.get("DGT_TEST_BACKEND", "gloo")
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    try:
        assert NE_TOTAL % world_size == 0, (
            f"NE_TOTAL={NE_TOTAL} not divisible by world_size={world_size}"
        )
        ne_local = NE_TOTAL // world_size

        # Build identical full inputs on every rank from the same seed.
        torch.manual_seed(SEED)
        full_nodes = torch.randn(NE_TOTAL, NP_PER, HIDDEN)
        full_centroids = torch.rand(NE_TOTAL, DIM)
        full_node_pos = torch.rand(NE_TOTAL, NP_PER, DIM)

        # Same module weights on every rank (same construction order + seed).
        # Re-seed before module construction so the input randn calls above
        # don't shift the module init RNG state across world sizes.
        torch.manual_seed(SEED + 1)
        pool = PerceiverPool(HIDDEN, NUM_HEADS, K_SUMMARY)
        readout = SummaryReadout(HIDDEN, NUM_HEADS)

        # This rank's element slice.
        s = rank * ne_local
        e = (rank + 1) * ne_local
        local_nodes = full_nodes[s:e].contiguous()
        local_centroids = full_centroids[s:e].contiguous()
        local_node_pos = full_node_pos[s:e].contiguous()

        # Pool local elements into summaries.
        local_summary = pool(local_nodes, local_centroids, local_node_pos)

        # Autograd-aware gather across ranks.
        summary_all = _autograd_all_gather(local_summary, world_size).reshape(
            NE_TOTAL, K_SUMMARY, HIDDEN
        )
        centroids_all = _autograd_all_gather(
            local_centroids, world_size
        ).reshape(NE_TOTAL, DIM)

        # Cross-attend local nodes to the gathered global summary set.
        delta_local = readout(
            summary_all, centroids_all, local_nodes, local_node_pos
        )

        # Gather per-rank outputs to rank 0 for comparison.
        out_list = [torch.zeros_like(delta_local) for _ in range(world_size)]
        dist.all_gather(out_list, delta_local)

        if rank == 0:
            out_full = torch.cat(out_list, dim=0)  # (NE_TOTAL, NP_PER, HIDDEN)
            save_path = f"/tmp/dgt_hier_invariance_N{world_size}.pt"
            torch.save(out_full, save_path)
            print(
                f"[rank 0] saved {save_path}\n"
                f"         shape={tuple(out_full.shape)}\n"
                f"         L2 norm={out_full.norm().item():.6e}"
            )
    finally:
        dist.destroy_process_group()


def _run_compare():
    """Compare /tmp/dgt_hier_invariance_N{ws}.pt across world sizes."""
    candidate_world_sizes = (1, 2, 4)
    ref_path = "/tmp/dgt_hier_invariance_N1.pt"
    if not os.path.exists(ref_path):
        raise SystemExit(
            f"Missing reference {ref_path}. Run torchrun --nproc_per_node=1 first."
        )
    ref = torch.load(ref_path, map_location="cpu")

    failed = []
    for ws in candidate_world_sizes:
        if ws == 1:
            continue
        path = f"/tmp/dgt_hier_invariance_N{ws}.pt"
        if not os.path.exists(path):
            print(f"N={ws}: skipped (no file at {path})")
            continue
        out = torch.load(path, map_location="cpu")
        diff = (ref - out).abs()
        max_diff = diff.max().item()
        rel = max_diff / (ref.abs().max().item() + 1e-12)
        print(
            f"N={ws}: max_abs_diff={max_diff:.3e}  "
            f"max_rel_diff={rel:.3e}"
        )
        if rel >= 1e-5:
            failed.append((ws, rel))

    if failed:
        raise SystemExit(
            "Rank-invariance violated: " + ", ".join(
                f"N={ws} (rel_diff={r:.2e})" for ws, r in failed
            )
        )
    print("OK: all multi-rank outputs match the N=1 baseline within 1e-5.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "compare":
        _run_compare()
    elif "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        _run_distributed()
    else:
        raise SystemExit(
            "Run under torchrun for the distributed test, e.g.:\n"
            "  torchrun --nproc_per_node=2 tests/test_rank_invariance.py\n"
            "Or with the 'compare' subcommand to compare saved outputs:\n"
            "  python tests/test_rank_invariance.py compare\n"
            "Single-process pytest sanity checks: pytest tests/test_rank_invariance.py"
        )
