"""Read nekRS/Nek5000 .f field files as a graph + data source.

Format (verified against 3rd_party/nek5000/core/prepost.f mfo_write_hdr and
ic.f parse_std_hdr / mfi_getv):
- bytes 0-131: ASCII header
  '#std wdsiz nx ny nz nelo nelgo time istep fid0 nfileo rdcode p0th ifpm'
- bytes 132-135: float32 6.54321 endianness probe
- bytes 136 .. 136+4*nelo: int32 global element id per element (file order)
- field data from offs = 136+4*nelo, fields in rdcode order (X, U, P, T,
  S..): per element, components interleaved: [c0(nxyz), c1(nxyz), ...],
  elements in map order, wdsiz = 4 or 8.

The graph is reconstructed from the mesh record (X): nodes are the GLL
points in the file's element order (lattice index i + j*nq + k*nq^2, i
fastest -- the same convention as the gnn plugin); global ids are assigned
by distributed coordinate coincidence matching (rendezvous hash over
quantized coordinates). Any consistent numbering of the coincidence classes
is valid downstream, since consumers only ever compare gids for equality.
"""

import numpy as np

from .element_data import LocalElements
from .mpiutil import alltoallv, alltoallv_2d
from .templates import stencil_template


class FldHeader:
    def __init__(self, path):
        with open(path, "rb") as f:
            hdr = f.read(132).decode("ascii", errors="replace").split()
            probe = np.frombuffer(f.read(4), dtype=np.float32)[0]
        assert hdr[0] == "#std", f"not a std .f file: {path}"
        self.wdsiz = int(hdr[1])
        self.nx, self.ny, self.nz = int(hdr[2]), int(hdr[3]), int(hdr[4])
        self.nelo = int(hdr[5])
        self.nelgo = int(hdr[6])
        self.time = float(hdr[7])
        self.istep = int(hdr[8])
        self.rdcode = hdr[11]
        self.swap = not np.isclose(probe, 6.54321, atol=1e-4)
        if self.swap:
            probe2 = probe.byteswap()
            assert np.isclose(probe2, 6.54321, atol=1e-4), (
                f"cannot determine endianness of {path}"
            )
        assert self.nelo == self.nelgo, (
            "multi-file .f sets are not supported (nekRS writes single "
            "files)"
        )
        self.nxyz = self.nx * self.ny * self.nz
        self.data_start = 136 + 4 * self.nelo

        # (code, ncomp) in file order
        self.fields = []
        code = self.rdcode.upper()
        i = 0
        while i < len(code):
            c = code[i]
            if c == "X":
                self.fields.append(("X", 3))
            elif c == "U":
                self.fields.append(("U", 3))
            elif c == "P":
                self.fields.append(("P", 1))
            elif c == "T":
                self.fields.append(("T", 1))
            elif c == "S":
                ns = int(code[i + 1 : i + 3])
                for s in range(ns):
                    self.fields.append((f"S{s:02d}", 1))
                i += 2
            i += 1

    def field_offset_bytes(self, name):
        off = self.data_start
        for code, ncomp in self.fields:
            if code == name:
                return off, ncomp
            off += self.nelo * ncomp * self.nxyz * self.wdsiz
        raise KeyError(f"field {name} not in rdcode {self.rdcode}")

    def read_field(self, path, name, e0, e1):
        """Elements [e0, e1) of a field -> ((e1-e0)*nxyz, ncomp) float64."""
        off, ncomp = self.field_offset_bytes(name)
        dtype = np.float32 if self.wdsiz == 4 else np.float64
        nel = e1 - e0
        raw = np.fromfile(
            path,
            dtype=dtype,
            count=nel * ncomp * self.nxyz,
            offset=off + e0 * ncomp * self.nxyz * self.wdsiz,
        )
        if self.swap:
            raw = raw.byteswap()
        arr = raw.reshape(nel, ncomp, self.nxyz).transpose(0, 2, 1)
        return np.ascontiguousarray(
            arr.reshape(nel * self.nxyz, ncomp), dtype=np.float64
        )

    def read_element_map(self, path):
        m = np.fromfile(path, dtype=np.int32, count=self.nelo, offset=136)
        return m.byteswap() if self.swap else m


def _mpi_min():
    from mpi4py import MPI

    return MPI.MIN


def _mpi_max():
    from mpi4py import MPI

    return MPI.MAX


def _min_gll_spacing(pos, template, np_pts, comm):
    ne = pos.shape[0] // np_pts
    if ne == 0:
        local = np.inf
    else:
        offs = np.arange(ne, dtype=np.int64) * np_pts
        ei = (template[None, :, :] + offs[:, None, None]).reshape(-1, 2)
        d = np.linalg.norm(pos[ei[:, 0]] - pos[ei[:, 1]], axis=1)
        local = float(d.min())
    return comm.allreduce(local, op=min)


def assign_gids(
    pos, template, np_pts, comm, bin_factor=4.0, periodic=(False,) * 3
):
    """Assign positive int64 gids to coincidence classes of coordinates.

    Quantizes coordinates to bins of size (min GLL spacing / bin_factor):
    coincident copies (equal up to roundoff) fall in the same bin, distinct
    GLL points never do. Bin keys are rendezvous-hashed; each home rank
    numbers its unique keys with a global exclusive scan.

    periodic: per-axis flags. On a periodic axis, nodes sitting on the
    domain max face are folded onto the min face before quantization, so
    periodic images join the same coincidence class -- matching the
    topology-aware global numbering nekRS builds from periodic BCs. (This is
    what makes coordinate-only .f reconstruction equivalent to
    mesh->globalIds for periodic boxes.)
    """
    size = comm.Get_size()
    h = _min_gll_spacing(pos, template, np_pts, comm) / bin_factor

    if any(periodic):
        pos = pos.copy()
        lo = np.empty(3)
        hi = np.empty(3)
        comm.Allreduce(pos.min(axis=0), lo, op=_mpi_min())
        comm.Allreduce(pos.max(axis=0), hi, op=_mpi_max())
        for ax in range(3):
            if periodic[ax]:
                on_max = np.abs(pos[:, ax] - hi[ax]) < h
                pos[on_max, ax] = lo[ax]

    key = np.round(pos / h).astype(np.int64)  # (n, 3)

    hashed = (
        key[:, 0] * np.int64(73856093)
        ^ key[:, 1] * np.int64(19349663)
        ^ key[:, 2] * np.int64(83492791)
    )
    home = (hashed % size + size) % size

    order = np.argsort(home, kind="stable")
    scounts = np.bincount(home, minlength=size).astype(np.int64)
    rcounts = np.empty(size, dtype=np.int64)
    comm.Alltoall(scounts, rcounts)
    recv = alltoallv_2d(key[order], scounts, rcounts, comm)

    uniq, inverse = np.unique(recv, axis=0, return_inverse=True)
    n_uniq = np.int64(uniq.shape[0])
    offset = np.zeros_like(n_uniq)
    comm.Exscan(n_uniq, offset)
    gid_back = alltoallv(
        1 + offset + inverse.astype(np.int64), rcounts, scounts, comm
    )
    gids = np.empty(pos.shape[0], dtype=np.int64)
    gids[order] = gid_back
    return gids


class FldSource:
    """Graph + field source built from one or more .f files.

    mesh_file: the .f file containing the mesh (rdcode with X); usually the
    first checkpoint. All files must share nx and element count/order.
    """

    def __init__(self, mesh_file, periodic=(False,) * 3):
        self.mesh_file = mesh_file
        self.periodic = periodic
        self.hdr = FldHeader(mesh_file)
        self.Np = self.hdr.nxyz
        assert self.hdr.nx == self.hdr.ny == self.hdr.nz, (
            "only hex elements with equal order per direction are supported"
        )
        self.nq = self.hdr.nx
        self.n_elements_total = self.hdr.nelo
        self.src_size = None  # not partition-based

    def my_ordinal_range(self, comm):
        rank, size = comm.Get_rank(), comm.Get_size()
        n = self.n_elements_total
        return rank * n // size, (rank + 1) * n // size

    def read_elements(self, comm):
        o0, o1 = self.my_ordinal_range(comm)
        pos = self.hdr.read_field(self.mesh_file, "X", o0, o1)
        template = stencil_template(self.nq)
        gids = assign_gids(
            pos, template, self.Np, comm, periodic=self.periodic
        )
        elems = LocalElements(
            Np=self.Np,
            ordinals=np.arange(o0, o1, dtype=np.int64),
            pos=pos,
            gids=gids,
        )
        return elems, template

    def read_node_field(self, comm, spec, ncols=None):
        """spec: (path, field_name) of a .f file field, e.g. (f, "U")."""
        path, name = spec
        hdr = FldHeader(path)
        assert hdr.nxyz == self.Np and hdr.nelo == self.n_elements_total, (
            f"{path} is not compatible with the mesh file"
        )
        o0, o1 = self.my_ordinal_range(comm)
        return hdr.read_field(path, name, o0, o1)
