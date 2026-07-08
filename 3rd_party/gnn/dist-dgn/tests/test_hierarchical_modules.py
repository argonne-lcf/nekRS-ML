"""Phase 1a unit tests: ``PerceiverPool`` and ``SummaryReadout`` in isolation.

Pure-torch, no distributed dependencies. Runs on CPU.

Run with:
    cd 3rd_party/gnn/dist-dgn && pytest tests/test_hierarchical_modules.py -v
"""

import pytest
import torch

from hierarchical import PerceiverPool, SummaryReadout

torch.manual_seed(0)

NE = 6
NP_PER = 16            # poly_order=3 in 2D-like setup
HIDDEN = 32
NUM_HEADS = 4
K_SUMMARY = 4
DIM = 3                # pos is always 3D in the existing pipeline


def _make_inputs(ne=NE, np_per=NP_PER, hidden=HIDDEN, dim=DIM):
    nodes = torch.randn(ne, np_per, hidden, dtype=torch.float32)
    centroids = torch.rand(ne, dim, dtype=torch.float32)  # already in [0, 1]
    node_pos = torch.rand(ne, np_per, dim, dtype=torch.float32)
    return nodes, centroids, node_pos


# ---------------------------------------------------------------------------
# PerceiverPool
# ---------------------------------------------------------------------------


def test_perceiverpool_shape():
    pool = PerceiverPool(HIDDEN, NUM_HEADS, K_SUMMARY)
    nodes, centroids, node_pos = _make_inputs()
    out = pool(nodes, centroids, node_pos)
    assert out.shape == (NE, K_SUMMARY, HIDDEN)
    assert out.dtype == nodes.dtype


def test_perceiverpool_element_permutation_equivariance():
    """Shuffling the element order should shuffle the output rows identically,
    provided centroids and node positions are shuffled the same way."""
    pool = PerceiverPool(HIDDEN, NUM_HEADS, K_SUMMARY)
    pool.eval()
    nodes, centroids, node_pos = _make_inputs()
    with torch.no_grad():
        out = pool(nodes, centroids, node_pos)

    perm = torch.tensor([3, 1, 5, 0, 4, 2])
    nodes_p = nodes[perm]
    centroids_p = centroids[perm]
    node_pos_p = node_pos[perm]
    with torch.no_grad():
        out_p = pool(nodes_p, centroids_p, node_pos_p)

    assert torch.allclose(out_p, out[perm], atol=1e-6)


def test_perceiverpool_gradient_flow():
    pool = PerceiverPool(HIDDEN, NUM_HEADS, K_SUMMARY)
    nodes, centroids, node_pos = _make_inputs()
    nodes.requires_grad_(True)
    out = pool(nodes, centroids, node_pos)
    out.sum().backward()
    assert nodes.grad is not None
    assert torch.isfinite(nodes.grad).all()
    for name, p in pool.named_parameters():
        assert p.grad is not None, f"{name} has no grad"
        assert torch.isfinite(p.grad).all(), f"{name} has non-finite grad"


# ---------------------------------------------------------------------------
# SummaryReadout
# ---------------------------------------------------------------------------


def _make_readout_inputs(ne_local=NE, ne_total=NE * 2, k=K_SUMMARY):
    summary = torch.randn(ne_total, k, HIDDEN, dtype=torch.float32)
    summary_centroids = torch.rand(ne_total, DIM, dtype=torch.float32)
    nodes = torch.randn(ne_local, NP_PER, HIDDEN, dtype=torch.float32)
    node_pos = torch.rand(ne_local, NP_PER, DIM, dtype=torch.float32)
    return summary, summary_centroids, nodes, node_pos


def test_summaryreadout_shape():
    readout = SummaryReadout(HIDDEN, NUM_HEADS)
    summary, summary_centroids, nodes, node_pos = _make_readout_inputs()
    out = readout(summary, summary_centroids, nodes, node_pos)
    assert out.shape == nodes.shape
    assert out.dtype == nodes.dtype


def test_summaryreadout_padding_mask_matches_unpadded():
    """Masking out padded keys must give the same output as not gathering them
    in the first place. We build a 'no-pad' reference using just the real
    elements, then a 'with-pad' input that appends random garbage and masks
    it, and check the two outputs agree."""
    readout = SummaryReadout(HIDDEN, NUM_HEADS)
    readout.eval()

    summary_real, centroids_real, nodes, node_pos = _make_readout_inputs(
        ne_total=NE
    )

    # Pad: append a couple of garbage element-summaries that the mask will reject.
    n_pad = 3
    summary_pad = torch.randn(n_pad, K_SUMMARY, HIDDEN, dtype=torch.float32)
    centroids_pad = torch.rand(n_pad, DIM, dtype=torch.float32)
    summary_full = torch.cat([summary_real, summary_pad], dim=0)
    centroids_full = torch.cat([centroids_real, centroids_pad], dim=0)

    mask = torch.zeros(NE + n_pad, dtype=torch.bool)
    mask[:NE] = True

    with torch.no_grad():
        out_pad = readout(
            summary_full, centroids_full, nodes, node_pos,
            key_padding_mask=mask,
        )
        out_ref = readout(summary_real, centroids_real, nodes, node_pos)

    assert torch.allclose(out_pad, out_ref, atol=1e-6)


def test_summaryreadout_gradient_flow():
    readout = SummaryReadout(HIDDEN, NUM_HEADS)
    summary, summary_centroids, nodes, node_pos = _make_readout_inputs()
    summary.requires_grad_(True)
    nodes.requires_grad_(True)
    out = readout(summary, summary_centroids, nodes, node_pos)
    out.sum().backward()
    assert summary.grad is not None and torch.isfinite(summary.grad).all()
    assert nodes.grad is not None and torch.isfinite(nodes.grad).all()
    for name, p in readout.named_parameters():
        assert p.grad is not None, f"{name} has no grad"
        assert torch.isfinite(p.grad).all(), f"{name} has non-finite grad"


# ---------------------------------------------------------------------------
# Roundtrip
# ---------------------------------------------------------------------------


def test_pool_then_readout_roundtrip_grad():
    """Pool nodes then read back, verify gradients flow through both modules
    and the output shape matches the input."""
    pool = PerceiverPool(HIDDEN, NUM_HEADS, K_SUMMARY)
    readout = SummaryReadout(HIDDEN, NUM_HEADS)

    nodes, centroids, node_pos = _make_inputs()
    nodes.requires_grad_(True)
    summary = pool(nodes, centroids, node_pos)
    delta = readout(summary, centroids, nodes, node_pos)
    assert delta.shape == nodes.shape

    (nodes + delta).sum().backward()
    assert nodes.grad is not None and torch.isfinite(nodes.grad).all()
    for mod in (pool, readout):
        for name, p in mod.named_parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all(), name


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
