"""
Rank-count-independent repartitioning of nekRS element-based GLL graphs.

Model-agnostic: reads the element-based GLL graph produced by the nekRS gnn
plugin (binary gnn_outputs files) or reconstructs it purely from nekRS .f
field files, redistributes whole elements across the current MPI
communicator, and regenerates the five per-rank arrays that define a
distributed GLL graph (pos, global_ids, edge_index, local_unique_mask,
halo_unique_mask), plus consistent routing for any node-level field data.
The core depends only on numpy and mpi4py; per-model halo metadata (e.g.
dist-gnn's halo_info / node_degree / edge_weights) is derivable from these
arrays at the new size (the CLI does this for dist-gnn).

Design notes (verified against src/plugins/gnn.cpp and gnn_connectivity.cpp):
- Graph nodes are the GLL points of whole elements, element-major in blocks
  of Np = (p+1)^3.
- Every edge lies inside a single element; the intra-element edge pattern is
  an identical template for all elements. Cross-element connectivity emerges
  from coincident nodes sharing a global id, which is partition-independent.
- Both uniqueness masks are pure functions of (global_ids, partition).
"""

from .api import Repartitioner
from .sources import BinSource

__all__ = ["BinSource", "Repartitioner"]
