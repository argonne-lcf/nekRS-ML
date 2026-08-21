"""
Rank-count-independent graph repartitioning for the dist-gnn model.

Reads the element-based GLL graph produced by the nekRS gnn plugin (binary
gnn_outputs files, ADIOS streams, or nekRS .f field files), redistributes
whole elements across the current MPI communicator, and regenerates the five
per-rank arrays the trainer consumes (pos, global_ids, edge_index,
local_unique_mask, halo_unique_mask). All halo-exchange metadata
(halo_info, node_degree, edge_weights) is then derivable at the new size by
the existing create_halo_info_par machinery.

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
