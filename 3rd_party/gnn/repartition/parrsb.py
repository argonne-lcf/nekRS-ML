"""ctypes wrapper for parRSB's parrsb_part_mesh (connectivity-aware,
fully distributed partitioning — no rank-0 gather).

Requires libparrsb_shim.so, built by build_parrsb_shim.sh against the
libparRSB.a/libgs.a of a nekRS install. Search order: $PARRSB_SHIM_LIB,
next to this file, $NEKRS_HOME/lib, $NEKRS_HOME/nek5000/3rd_party/parRSB/lib.

parRSB partitions the element connectivity graph induced by shared corner
vertices, so it sees the true mesh topology — including periodic faces,
which coordinate-based RCB always cuts. All options can also be overridden
at run time via PARRSB_* environment variables (see parRSB.h).
"""

import ctypes
import os

import numpy as np

# parRSB.h: 0 - RSB, 1 - RCB, 2 - RIB
PARTITIONER_RSB = 0
PARTITIONER_RCB = 1
PARTITIONER_RIB = 2

_LIB_NAME = "libparrsb_shim.so"
_lib = None


def _candidates():
    env = os.environ.get("PARRSB_SHIM_LIB")
    if env:
        yield env
    here = os.path.dirname(os.path.abspath(__file__))
    yield os.path.join(here, _LIB_NAME)
    nekrs_home = os.environ.get("NEKRS_HOME")
    if nekrs_home:
        yield os.path.join(nekrs_home, "lib", _LIB_NAME)
        yield os.path.join(
            nekrs_home, "nek5000", "3rd_party", "parRSB", "lib", _LIB_NAME
        )


def find_library():
    """Path of the shim library, or None if not built/installed."""
    for path in _candidates():
        if os.path.isfile(path):
            return path
    return None


def available():
    return find_library() is not None


def _load():
    global _lib
    if _lib is None:
        path = find_library()
        if path is None:
            raise RuntimeError(
                "libparrsb_shim.so not found (searched: "
                + ", ".join(_candidates())
                + "). Build it with repartition/build_parrsb_shim.sh."
            )
        lib = ctypes.CDLL(path)
        fn = lib.repartition_parrsb_part_mesh
        fn.restype = ctypes.c_int
        fn.argtypes = [
            ctypes.POINTER(ctypes.c_int),       # part (out)
            ctypes.POINTER(ctypes.c_longlong),  # vtx
            ctypes.POINTER(ctypes.c_double),    # xyz (may be NULL)
            ctypes.c_int,                       # nel
            ctypes.c_int,                       # nv
            ctypes.c_int,                       # partitioner
            ctypes.c_int,                       # verbose_level
            ctypes.c_int,                       # MPI_Fint fcomm
        ]
        _lib = lib
    return _lib


def partition_parrsb(
    corner_gids, corner_xyz, comm, partitioner=PARTITIONER_RSB, verbose=0
):
    """Destination rank per local element via parRSB.

    corner_gids: (nel, nv) int64 — global ids of the element corner
      vertices (nv=8 for hex). Coincident corners (including periodic
      images) must share a gid; this defines the connectivity graph.
    corner_xyz: (nel, nv, 3) float64 corner coordinates, or None to
      disable parRSB's geometric pre-partition (rsb_pre).
    comm: mpi4py communicator over ALL participating ranks (elements may
      be empty on some ranks). Result ranks are in [0, comm.size).
    """
    vtx = np.ascontiguousarray(corner_gids, dtype=np.int64)
    if vtx.ndim != 2:
        raise ValueError(f"corner_gids must be (nel, nv), got {vtx.shape}")
    nel, nv = vtx.shape

    if corner_xyz is not None:
        xyz = np.ascontiguousarray(corner_xyz, dtype=np.float64)
        if xyz.shape != (nel, nv, 3):
            raise ValueError(
                f"corner_xyz must be ({nel}, {nv}, 3), got {xyz.shape}"
            )
        xyz_ptr = xyz.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    else:
        xyz_ptr = None

    part = np.zeros(max(nel, 1), dtype=np.int32)
    lib = _load()
    rc = lib.repartition_parrsb_part_mesh(
        part.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        vtx.ctypes.data_as(ctypes.POINTER(ctypes.c_longlong)),
        xyz_ptr,
        nel,
        nv,
        partitioner,
        verbose,
        comm.py2f(),
    )
    if rc != 0:
        raise RuntimeError(f"parrsb_part_mesh failed with code {rc}")

    dest = part[:nel].astype(np.int64)
    if nel and (dest.min() < 0 or dest.max() >= comm.Get_size()):
        raise RuntimeError(
            "parrsb_part_mesh returned out-of-range destination ranks"
        )
    return dest


__all__ = [
    "PARTITIONER_RCB",
    "PARTITIONER_RIB",
    "PARTITIONER_RSB",
    "available",
    "find_library",
    "partition_parrsb",
]
