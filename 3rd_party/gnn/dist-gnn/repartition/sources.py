"""Element-block readers for the different graph/data sources.

BinSource reads the binary gnn_outputs_poly_* directory written by the nekRS
gnn plugin at any source rank count. Each reader rank takes a contiguous
range of global element ordinals (source-rank-major element order) so reads
are element-aligned regardless of the current communicator size.

File layouts (verified against gnn.cpp writeToFileBinary, gnn.cpp:174-213):
- pos_node_rank_R_size_S.bin      float64, N x 3 records (x, y, z)
- global_ids_rank_R_size_S.bin    int64,   N x 1
- edge_index_rank_R_size_S.bin    int32,   E x 2 records (neighbor, owner)
- local/halo_unique_mask_*.bin    int32,   N x 1
- Np_rank_0_size_S                ASCII scalar (GLL points per element)
Node-level field files (trajectories, fld_* snapshots) are float64 with
rows >= N per source rank (fieldOffset alignment padding at the end).
"""

import glob
import os
import re

import numpy as np

from .element_data import LocalElements


def _read_slice(path, dtype, ncols, row0, nrows):
    itemsize = np.dtype(dtype).itemsize
    return np.fromfile(
        path,
        dtype=dtype,
        count=nrows * ncols,
        offset=row0 * ncols * itemsize,
    ).reshape(nrows, ncols)


class BinSource:
    def __init__(self, src_dir, src_size=None, np_pts=None):
        self.src_dir = src_dir
        if src_size is None:
            src_size = self.detect_size(src_dir)
        self.src_size = src_size
        if np_pts is None:
            np_file = os.path.join(src_dir, f"Np_rank_0_size_{src_size}")
            np_pts = int(float(np.loadtxt(np_file)))
        self.Np = np_pts

        # elements per source rank from the pos_node file sizes
        self.ne_per_src = np.array(
            [
                os.path.getsize(self._path("pos_node", s)) // (self.Np * 24)
                for s in range(src_size)
            ],
            dtype=np.int64,
        )
        self.el_offsets = np.zeros(src_size + 1, dtype=np.int64)
        np.cumsum(self.ne_per_src, out=self.el_offsets[1:])
        self.n_elements_total = int(self.el_offsets[-1])

    @staticmethod
    def detect_size(src_dir):
        pat = os.path.join(src_dir, "pos_node_rank_0_size_*.bin")
        hits = glob.glob(pat)
        if not hits:
            raise FileNotFoundError(f"no graph files found: {pat}")
        sizes = sorted(
            int(re.search(r"_size_(\d+)\.bin$", h).group(1)) for h in hits
        )
        if len(sizes) > 1:
            raise RuntimeError(
                f"multiple source sizes in {src_dir}: {sizes}; "
                "pass src_size explicitly"
            )
        return sizes[0]

    def _path(self, name, src_rank, ext=".bin"):
        return os.path.join(
            self.src_dir, f"{name}_rank_{src_rank}_size_{self.src_size}{ext}"
        )

    def my_ordinal_range(self, comm):
        rank, size = comm.Get_rank(), comm.Get_size()
        n = self.n_elements_total
        return rank * n // size, (rank + 1) * n // size

    def _overlaps(self, o0, o1):
        """Yield (src_rank, local_el0, n_el) covering ordinals [o0, o1)."""
        for s in range(self.src_size):
            a = max(o0, int(self.el_offsets[s]))
            b = min(o1, int(self.el_offsets[s + 1]))
            if a < b:
                yield s, a - int(self.el_offsets[s]), b - a

    def _read_node_slices(self, o0, o1, path_fn, dtype, ncols):
        np_pts = self.Np
        parts = [
            _read_slice(
                path_fn(s), dtype, ncols, el0 * np_pts, nel * np_pts
            )
            for s, el0, nel in self._overlaps(o0, o1)
        ]
        if parts:
            return np.concatenate(parts, axis=0)
        return np.empty((0, ncols), dtype=dtype)

    def read_elements(self, comm):
        """Read this rank's ordinal range. Returns (LocalElements, template).

        template is the (Et, 2) int64 intra-element edge pattern extracted
        from element 0 of source rank 0 (identical for all elements; edges
        never cross elements, and coincident-copy augmentation edges never
        connect two nodes of the same element).
        """
        o0, o1 = self.my_ordinal_range(comm)
        pos = self._read_node_slices(
            o0, o1, lambda s: self._path("pos_node", s), np.float64, 3
        )
        gids = self._read_node_slices(
            o0, o1, lambda s: self._path("global_ids", s), np.int64, 1
        ).reshape(-1)

        template = None
        if comm.Get_rank() == 0:
            ei = np.fromfile(
                self._path("edge_index", 0), dtype=np.int32
            ).reshape(-1, 2)
            intra = ei[(ei[:, 0] < self.Np) & (ei[:, 1] < self.Np)]
            template = np.unique(intra.astype(np.int64), axis=0)
        template = comm.bcast(template, root=0)

        elems = LocalElements(
            Np=self.Np,
            ordinals=np.arange(o0, o1, dtype=np.int64),
            pos=pos,
            gids=gids,
        )
        return elems, template

    def read_node_field(self, comm, path_for_src_rank, ncols):
        """Read a node-level field laid out like the graph nodes (rows may
        be fieldOffset-padded past N on each source rank); returns this
        rank's ordinal-range slice, (n_local_nodes, ncols) float64."""
        o0, o1 = self.my_ordinal_range(comm)
        return self._read_node_slices(
            o0, o1, path_for_src_rank, np.float64, ncols
        )
