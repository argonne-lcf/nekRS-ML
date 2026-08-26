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


class ElementSource:
    """Shared element-ordinal bookkeeping for every graph source.

    A source exposes its elements as a single global sequence ordered by
    source rank (source-rank major). Every reader rank claims a contiguous
    range of that sequence, so reads are element-aligned no matter how many
    ranks read or how many wrote. Subclasses set ``Np``, ``src_size`` and
    ``ne_per_src``, then call ``_set_element_counts``.
    """

    def _set_element_counts(self, ne_per_src):
        self.ne_per_src = np.asarray(ne_per_src, dtype=np.int64)
        self.el_offsets = np.zeros(self.ne_per_src.shape[0] + 1, dtype=np.int64)
        np.cumsum(self.ne_per_src, out=self.el_offsets[1:])
        self.n_elements_total = int(self.el_offsets[-1])

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


class BinSource(ElementSource):
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
        self._set_element_counts([
            os.path.getsize(self._path("pos_node", s)) // (self.Np * 24)
            for s in range(src_size)
        ])

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

    def _read_node_slices(self, o0, o1, path_fn, dtype, ncols):
        np_pts = self.Np
        parts = [
            _read_slice(path_fn(s), dtype, ncols, el0 * np_pts, nel * np_pts)
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


ALIGN_SIZE_BYTES = 256


def align_stride(n, itemsize=8):
    """nekRS alignStride<T> (src/core/nekrsSys.hpp.in:167-175).

    fieldOffset = ceil(n / pageW) * pageW with pageW = 256 / sizeof(T).
    itemsize is 8 for the default dfloat=double build, 4 for the optional
    NEKRS_USE_DFLOAT_FLOAT build.
    """
    page_w = ALIGN_SIZE_BYTES // itemsize
    n = int(n)
    if n % page_w:
        n = (n // page_w + 1) * page_w
    return n


def open_bp_read(path, comm=None):
    """Open a BP file for reading, with or without an MPI-aware adios2.

    A serial adios2 build (e.g. the pip wheel) rejects a communicator; each
    rank then opens the file independently, which is fine for read-only
    access and lets the test suite run without an MPI-enabled adios2.
    """
    import adios2
    from adios2 import Stream

    if comm is None or getattr(adios2, "is_built_with_mpi", None) is False:
        return Stream(path, "r")
    try:
        return Stream(path, "r", comm)
    except (RuntimeError, TypeError, ValueError):
        return Stream(path, "r")


class AdiosSource(ElementSource):
    """Element-block reader for the ADIOS2 graph.bp written by gnn.cpp.

    graph.bp is a global concatenation of per-writer blocks. The only
    announcement of the writer count W is ``shape("N")[0]``; N and num_edges
    are per-writer arrays of length W, and the block start of writer w is the
    exclusive scan of those counts (gnn.cpp:263-330).

    Two layout facts drive this reader, both verified against the writer:

    - Inside a writer block, pos_node and edge_index are COMPONENT-MAJOR
      (SoA): pos is [x(0..N_w-1), y(...), z(...)] (gnn.cpp:338-343) and
      edge_index is [nei(0..E_w-1), own(...)] (gnn.cpp:798-799). This is the
      raw device buffer -- ADIOS does not transpose. The .bin files of the
      same data ARE transposed to interleaved rows by writeToFileBinary
      (gnn.cpp:49-56, ``index = j * nRows + i``), so BinSource reshapes (n,
      ncols) while this reader must reshape per block with order="F".
      Reshaping the whole global array at once is wrong for any W > 1.

    - graph.bp carries NO alignStride padding (N_w = Ne_w * Np exactly), but
      the in_u/out_u solution stream does: each writer block is
      3 * fieldOffset long with component stride fieldOffset =
      alignStride(N_w), and the global offsets are a true per-writer scan of
      fieldOffset (trajGen.cpp:225-247). read_node_field handles that.

    Np is read defensively: the writer declares it {1},{1},{1} -- start 1,
    one past the end of its own shape (gnn.cpp:303) -- so a plain
    ``read("Np")`` returns 0, not Np. See _resolve_np.
    """

    def __init__(
        self, graph_path="graph.bp", comm=None, np_pts=None, itemsize=8
    ):
        self.graph_path = graph_path
        self.comm = comm
        self.itemsize = itemsize

        with open_bp_read(graph_path, comm) as stream:
            stream.begin_step()
            shape = stream.available_variables()["N"]["Shape"]
            self.src_size = int(str(shape).split(",")[0])
            w = self.src_size
            n_list = np.asarray(
                stream.read("N", [0], [w]), dtype=np.int64
            ).reshape(-1)
            self.num_edges_per_src = np.asarray(
                stream.read("num_edges", [0], [w]), dtype=np.int64
            ).reshape(-1)
            self.Np = self._resolve_np(stream, n_list, np_pts)
            stream.end_step()

        self.n_per_src = n_list
        if np.any(n_list % self.Np):
            raise RuntimeError(
                f"graph.bp N={n_list.tolist()} not divisible by Np={self.Np}"
            )
        self._set_element_counts(n_list // self.Np)

        # exclusive scans giving each writer's block start in the global array
        self.node_offsets = np.zeros(self.src_size + 1, dtype=np.int64)
        np.cumsum(n_list, out=self.node_offsets[1:])
        self.edge_offsets = np.zeros(self.src_size + 1, dtype=np.int64)
        np.cumsum(self.num_edges_per_src, out=self.edge_offsets[1:])

        # per-writer padded stride of the in_u/out_u solution stream
        self.fo_per_src = np.array(
            [align_stride(n, itemsize) for n in n_list], dtype=np.int64
        )
        self.fo_offsets = np.zeros(self.src_size + 1, dtype=np.int64)
        np.cumsum(self.fo_per_src, out=self.fo_offsets[1:])

        self._field_stream = None

    @staticmethod
    def _resolve_np(stream, n_list, np_pts=None):
        """Read Np around the writer's start-{1} defect (gnn.cpp:303).

        The writer declares Np with shape {1} but start {1}, so the single
        value sits one past the end of the declared global shape. A read with
        no selection (what client.py does today) returns the default-filled
        in-bounds element, i.e. 0. A read of [1],[1] returns the true value.
        Both are tried and the candidate that is positive and divides every
        N_w is used, so this keeps working if the writer is ever fixed.
        """
        if np_pts is not None:
            return int(np_pts)
        cands = []
        for sel in ([1], [1]), ():
            try:
                raw = stream.read("Np", *sel)
            except Exception:
                continue
            arr = np.asarray(raw).reshape(-1)
            if arr.size:
                cands.append(int(arr[0]))
        good = [
            c for c in dict.fromkeys(cands) if c > 0 and not np.any(n_list % c)
        ]
        if len(good) == 1:
            return good[0]
        raise RuntimeError(
            f"cannot determine Np from graph.bp (candidates {cands}, "
            f"N={n_list.tolist()}); pass np_pts explicitly"
        )

    def _read_block_components(
        self, stream, name, base, stride, row0, nrows, ncols, dtype
    ):
        """Read ncols component-major sub-ranges out of one writer block.

        base is the block's global start, stride the per-component stride
        inside it, [row0, row0+nrows) the rows wanted. Returns (nrows, ncols).
        """
        out = np.empty((nrows, ncols), dtype=dtype)
        for c in range(ncols):
            start = int(base + c * stride + row0)
            out[:, c] = np.asarray(
                stream.read(name, [start], [int(nrows)])
            ).reshape(-1)
        return out

    def read_elements(self, comm):
        """Read this rank's ordinal range. Returns (LocalElements, template).

        Mirrors BinSource.read_elements exactly, including the template
        extraction from writer block 0.
        """
        o0, o1 = self.my_ordinal_range(comm)
        np_pts = self.Np
        pos_parts, gid_parts = [], []

        with open_bp_read(self.graph_path, self.comm or comm) as stream:
            stream.begin_step()
            for s, el0, nel in self._overlaps(o0, o1):
                n_s = int(self.n_per_src[s])
                base = int(self.node_offsets[s])
                pos_parts.append(
                    self._read_block_components(
                        stream,
                        "pos_node",
                        base * 3,
                        n_s,
                        el0 * np_pts,
                        nel * np_pts,
                        3,
                        np.float64,
                    )
                )
                gid_parts.append(
                    np.asarray(
                        stream.read(
                            "global_ids",
                            [base + el0 * np_pts],
                            [nel * np_pts],
                        ),
                        dtype=np.int64,
                    ).reshape(-1)
                )

            template = None
            if comm.Get_rank() == 0:
                e0 = int(self.num_edges_per_src[0])
                ei = np.asarray(
                    stream.read("edge_index", [0], [e0 * 2])
                ).reshape((-1, 2), order="F")
                intra = ei[(ei[:, 0] < np_pts) & (ei[:, 1] < np_pts)]
                template = np.unique(intra.astype(np.int64), axis=0)
            stream.end_step()

        template = comm.bcast(template, root=0)
        pos = (
            np.concatenate(pos_parts, axis=0)
            if pos_parts
            else np.empty((0, 3), dtype=np.float64)
        )
        gids = (
            np.concatenate(gid_parts)
            if gid_parts
            else np.empty(0, dtype=np.int64)
        )
        elems = LocalElements(
            Np=np_pts,
            ordinals=np.arange(o0, o1, dtype=np.int64),
            pos=pos,
            gids=gids,
        )
        return elems, template

    def attach_field_stream(self, stream):
        """Set the open (SST or BP) stream that read_node_field reads from.

        The solution stream is stepped by the caller, so AdiosSource borrows
        it rather than owning it; it must be inside a step when read.
        """
        self._field_stream = stream

    def read_node_field(self, comm, spec, ncols, padded=True):
        """Read a node-level field variable and slice this rank's ordinals.

        spec is a variable name, or a (stream, name) pair to read from a
        stream other than the attached one. padded=True uses the
        alignStride'd per-writer strides of in_u/out_u; padded=False assumes
        a graph.bp-style unpadded stride of N_w.
        """
        if isinstance(spec, tuple):
            stream, name = spec
        else:
            stream, name = self._field_stream, spec
        if stream is None:
            raise RuntimeError(
                "no field stream attached; call attach_field_stream() or "
                "pass spec as a (stream, name) pair"
            )

        strides = self.fo_per_src if padded else self.n_per_src
        offsets = self.fo_offsets if padded else self.node_offsets
        o0, o1 = self.my_ordinal_range(comm)
        np_pts = self.Np
        parts = [
            self._read_block_components(
                stream,
                name,
                int(offsets[s]) * ncols,
                int(strides[s]),
                el0 * np_pts,
                nel * np_pts,
                ncols,
                np.float64,
            )
            for s, el0, nel in self._overlaps(o0, o1)
        ]
        if parts:
            return np.concatenate(parts, axis=0)
        return np.empty((0, ncols), dtype=np.float64)
