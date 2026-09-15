"""Top-level repartitioning driver."""

from .partition import partition_elements
from .rebuild import rebuild_graph_arrays
from .redistribute import redistribute_elements


class Repartitioner:
    """Reads a graph source, repartitions elements onto comm, and provides
    the trainer arrays plus consistent routing for node-level field data.

    Usage:
        src = BinSource("gnn_outputs_poly_7")
        rp = Repartitioner(src, comm, method="parrsb")
        arrays = rp.graph_arrays()
        u = rp.read_field(lambda s: f".../data_rank_{s}_size_{src.src_size}"
                          f"/u_step_10.bin", ncols=3)
    """

    def __init__(self, source, comm, method="parrsb"):
        self.source = source
        self.comm = comm
        self.method = method
        self.Np = source.Np

        elems_src, self.template = source.read_elements(comm)
        dest = partition_elements(elems_src, comm, method=method)
        self.elems, self.routing = redistribute_elements(elems_src, dest, comm)
        self._arrays = None

    def graph_arrays(self):
        """The five per-rank arrays the trainer consumes (see rebuild)."""
        if self._arrays is None:
            self._arrays = rebuild_graph_arrays(
                self.elems, self.template, self.comm
            )
        return self._arrays

    def read_field(self, path_for_src_rank, ncols):
        """Read a source node-level field and route it to the new layout."""
        local = self.source.read_node_field(self.comm, path_for_src_rank, ncols)
        return self.routing.route_node_array(local, self.comm)

    @property
    def n_nodes_local(self):
        return self.elems.n_nodes
