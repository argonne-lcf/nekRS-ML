import os
import sys
import socket
from typing import Optional
from argparse import ArgumentParser
from time import perf_counter
import math
from pprint import pprint

import torch
import torch.distributed as dist
import torch.distributed.nn as distnn

TORCH_DTYPE = torch.float32
TORCH_ITEMSIZE = torch.empty(0, dtype=TORCH_DTYPE).element_size()
MB_SIZE = 1000 * 1000

from mpi4py import MPI

SIZE = MPI.COMM_WORLD.Get_size()
RANK = MPI.COMM_WORLD.Get_rank()
COMM = MPI.COMM_WORLD
LOCAL_RANK = int(os.getenv("PALS_LOCAL_RANKID", default=RANK % SIZE))

try:
    WITH_CUDA = torch.cuda.is_available()
    if RANK == 0 and WITH_CUDA:
        print("Running on CUDA devices", flush=True)
except:
    WITH_CUDA = False
    pass

try:
    WITH_XPU = torch.xpu.is_available()
    if RANK == 0 and WITH_XPU:
        print("Running on XPU devices", flush=True)
except:
    WITH_XPU = False
    pass

if WITH_CUDA:
    DEVICE = torch.device("cuda")
    N_DEVICES = torch.cuda.device_count()
    DEVICE_ID = LOCAL_RANK if N_DEVICES > 1 else 0
    torch.cuda.set_device(DEVICE_ID)
elif WITH_XPU:
    DEVICE = torch.device("xpu")
    N_DEVICES = torch.xpu.device_count()
    DEVICE_ID = LOCAL_RANK if N_DEVICES > 1 else 0
    torch.xpu.set_device(DEVICE_ID)
else:
    DEVICE = torch.device("cpu")
    DEVICE_ID = "cpu"
    if RANK == 0:
        print("Running on CPU devices", flush=True)


def init_process_group(
    master_addr: Optional[str] = None,
    master_port: Optional[int] = 2345,
    backend: Optional[str] = None,
) -> None:
    os.environ["RANK"] = str(RANK)
    os.environ["WORLD_SIZE"] = str(SIZE)
    os.environ["MASTER_PORT"] = str(master_port)

    if master_addr is not None:
        MASTER_ADDR = str(master_addr) if RANK == 0 else None
    else:
        MASTER_ADDR = socket.gethostname() if RANK == 0 else None
    MASTER_ADDR = MPI.COMM_WORLD.bcast(MASTER_ADDR, root=0)
    os.environ["MASTER_ADDR"] = MASTER_ADDR

    if WITH_CUDA:
        backend = "nccl" if backend is None else str(backend)
    elif WITH_XPU:
        backend = "xccl" if backend is None else str(backend)
    else:
        backend = "gloo" if backend is None else str(backend)

    dist.init_process_group(
        backend,
        rank=int(RANK),
        world_size=int(SIZE),
        init_method="env://",
    )


def cleanup() -> None:
    dist.destroy_process_group()


def rcb_box_neighbors():
    """Neighbor list reproducing the shooting workflow's rcb partitioning.

    The shooting workflow runs turbChannel on a box mesh that
    ``examples/shooting_workflow_adios/nrsrun_aurora`` grows with the node
    count, partitioned by parRSB's rcb with periodic x and z and walls in y.
    This rebuilds that decomposition analytically -- recursively bisect the
    longest physical extent, lower half takes the lower rank ids -- and calls
    two ranks neighbors when their boxes touch on a face, edge or corner,
    which is what shares gll nodes and hence produces a halo.

    The mesh follows nrsrun_aurora exactly: SIM_NODES splits as GX x GZ with
    GZ the largest factor no bigger than sqrt(SIM_NODES), giving
    NX = 42*GX elements over LX = 2*pi*GX and NZ = 26*RANKS_PER_NODE*GZ over
    LZ = pi*RANKS_PER_NODE*GZ. Element counts matter only at the +/-1 level
    here, but they are cheap to carry and keep the cut planes on element
    boundaries the way parRSB places them.

    Returns this rank's neighbors and, for each of them, the number of gll
    nodes the two partitions share. Both are symmetric by construction, since
    both come from the one geometric intersection of the two boxes -- required
    for all_to_all, where a one-sided entry is a buffer size mismatch.
    """
    ranks_per_node = int(os.getenv("PALS_LOCAL_SIZE", 12))
    sim_nodes = max(1, SIZE // ranks_per_node)

    # nrsrun_aurora: GZ = largest factor of SIM_NODES with GZ*GZ <= SIM_NODES
    gz = 1
    f = 1
    while f * f <= sim_nodes:
        if sim_nodes % f == 0:
            gz = f
        f += 1
    gx = sim_nodes // gz

    poly_order = 7
    nx, ny, nz = 42 * gx, 18, 26 * ranks_per_node * gz
    lx, ly, lz = 2 * math.pi * gx, 2.0, math.pi * ranks_per_node * gz
    dx, dy, dz = lx / nx, ly / ny, lz / nz
    n_elem = (nx, ny, nz)
    periodic = (True, False, True)

    # Recursive coordinate bisection over the element index space
    boxes = [None] * SIZE
    next_rank = [0]

    def bisect(box, nparts):
        if nparts == 1:
            boxes[next_rank[0]] = box
            next_rank[0] += 1
            return
        extent = [
            (box[1] - box[0]) * dx,
            (box[3] - box[2]) * dy,
            (box[5] - box[4]) * dz,
        ]
        d = extent.index(max(extent))
        lo_parts = nparts // 2
        a, b = box[2 * d], box[2 * d + 1]
        cut = a + round((b - a) * lo_parts / nparts)
        cut = max(a + 1, min(b - 1, cut))
        lo_box = list(box)
        hi_box = list(box)
        lo_box[2 * d + 1] = cut
        hi_box[2 * d] = cut
        bisect(tuple(lo_box), lo_parts)
        bisect(tuple(hi_box), nparts - lo_parts)

    bisect((0, nx, 0, ny, 0, nz), SIZE)

    def shared_nodes(a, b):
        """Gll nodes boxes a and b share, 0 when they do not touch at all.

        An overlap of k elements spans k*P+1 nodes in that direction, so
        boxes that merely abut still share one plane of nodes (k = 0) and
        boxes that miss each other share none. The product over the three
        directions is the shared surface, and hence the halo, between the
        two ranks -- face contact is large, edge contact smaller, corner
        contact a single node.
        """
        count = 1
        for d in range(3):
            lo_a, hi_a = a[2 * d], a[2 * d + 1]
            lo_b, hi_b = b[2 * d], b[2 * d + 1]
            shifts = [0]
            if periodic[d]:
                shifts += [n_elem[d], -n_elem[d]]
            overlap = max(
                min(hi_a, hi_b + s) - max(lo_a, lo_b + s) for s in shifts
            )
            if overlap < 0:
                return 0
            count *= overlap * poly_order + 1
        return count

    mine = boxes[RANK]
    contact = (
        (r, shared_nodes(mine, boxes[r])) for r in range(SIZE) if r != RANK
    )
    shared = {r: n for r, n in contact if n > 0}
    neighbors = list(shared)

    if RANK == 0:
        print(
            f"rcb_box: {sim_nodes} sim nodes x {ranks_per_node} ranks/node, "
            f"mesh {nx} x {ny} x {nz}, box {lx:.1f} x {ly:.1f} x {lz:.1f}",
            flush=True,
        )
    return neighbors, shared


def get_neighbors(args):
    """Neighbor ranks, and the gll nodes shared with each where known.

    The shared counts are only available from rcb_box, which knows the
    geometry; nearest neighbors carries no notion of contact area, so it
    returns None and every buffer ends up the same size.
    """
    neighbors = []
    shared = None
    if "neighbor" in args.all_to_all_buff:
        if SIZE == 1:
            neighbors = [0]
        else:
            if args.neighbors == "nearest":
                for i in range(args.num_neighbors):
                    left_rank = (RANK - (1 + i)) % SIZE
                    right_rank = (RANK + (1 + i)) % SIZE
                    neighbors.extend([left_rank, right_rank])
            elif args.neighbors == "rcb_box":
                neighbors, shared = rcb_box_neighbors()
        if args.logging == "verbose":
            print(f"[{RANK}] neighbor list: {neighbors}", flush=True)
            COMM.Barrier()
    return neighbors, shared

def buffer_lengths(args, neighbors, shared):
    """Element count for each neighbor's buffer.

    Uniform, unless the neighbor generator also reported how many gll nodes
    each pair of ranks shares, in which case every buffer is scaled by that
    contact area and --buff_size becomes the size of the largest buffer in
    the job. That reproduces the lopsidedness of a real halo, where a rank
    trades hundreds of times more data with the neighbors it meets face on
    than with the ones it only touches along an edge.

    The scale factor is a global max rather than a per-rank one: all_to_all
    pairs rank i's send with rank j's recv, so the two have to agree on the
    size of the buffer between them, and a per-rank normalization would have
    them disagree whenever their busiest neighbors differ.
    """
    n_max = args.buff_size // TORCH_ITEMSIZE
    if shared is None:
        return dict.fromkeys(neighbors, n_max)

    largest = COMM.allreduce(max(shared.values(), default=1), op=MPI.MAX)
    return {i: max(1, round(n_max * shared[i] / largest)) for i in neighbors}


def build_buffers(args, neighbors, shared=None):
    buff_send_sz = [0] * SIZE
    buff_recv_sz = [0] * SIZE

    # --buff_size is the per-buffer payload in bytes; turn it into a length
    n_elements = args.buff_size // TORCH_ITEMSIZE
    lengths = buffer_lengths(args, neighbors, shared)

    if args.all_to_all_buff == "naive":
        buff_send = [torch.empty(0, device=DEVICE)] * SIZE
        buff_recv = [torch.empty(0, device=DEVICE)] * SIZE
        for i in range(SIZE):
            buff_send[i] = torch.empty(
                n_elements,
                dtype=TORCH_DTYPE,
                device=DEVICE,
            )
            buff_send_sz[i] = (
                torch.numel(buff_send[i])
                * buff_send[i].element_size()
                / MB_SIZE
            )
            buff_recv[i] = torch.empty(
                n_elements,
                dtype=TORCH_DTYPE,
                device=DEVICE,
            )
            buff_recv_sz[i] = (
                torch.numel(buff_recv[i])
                * buff_recv[i].element_size()
                / MB_SIZE
            )
    elif args.all_to_all_buff == "neighbor":
        buff_send = [torch.empty(0, device=DEVICE)] * SIZE
        buff_recv = [torch.empty(0, device=DEVICE)] * SIZE
        for i in neighbors:
            buff_send[i] = torch.empty(
                lengths[i],
                dtype=TORCH_DTYPE,
                device=DEVICE,
            )
            buff_send_sz[i] = (
                torch.numel(buff_send[i])
                * buff_send[i].element_size()
                / MB_SIZE
            )
            buff_recv[i] = torch.empty(
                lengths[i],
                dtype=TORCH_DTYPE,
                device=DEVICE,
            )
            buff_recv_sz[i] = (
                torch.numel(buff_recv[i])
                * buff_recv[i].element_size()
                / MB_SIZE
            )
    elif args.all_to_all_buff == "semi-optimized":
        buff_send = [torch.zeros(1, device=DEVICE)] * SIZE
        buff_recv = [torch.zeros(1, device=DEVICE)] * SIZE
        for i in neighbors:
            buff_send[i] = torch.zeros(
                lengths[i],
                dtype=TORCH_DTYPE,
                device=DEVICE,
            )
            buff_send_sz[i] = (
                torch.numel(buff_send[i])
                * buff_send[i].element_size()
                / MB_SIZE
            )
            buff_recv[i] = torch.zeros(
                lengths[i],
                dtype=TORCH_DTYPE,
                device=DEVICE,
            )
            buff_recv_sz[i] = (
                torch.numel(buff_recv[i])
                * buff_recv[i].element_size()
                / MB_SIZE
            )

    # Print information about the buffers
    if args.logging == "verbose":
        print(
            f"[RANK {RANK}]: Send buffers of size [MB]: {buff_send_sz}",
            flush=True,
        )
        COMM.Barrier()

    return [buff_send, buff_recv]


def halo_exchange(args, neighbors, buffers):
    buff_send_safe = buffers[0]
    buff_recv_safe = buffers[1]
    buff_send = buff_send_safe
    buff_recv = buff_recv_safe

    times = []
    for itr in range(args.iterations):
        # initialize the buffers
        for i in range(SIZE):
            buff_send[i] = torch.empty_like(buff_send_safe[i])
            buff_recv[i] = torch.empty_like(buff_recv_safe[i])

        # fill in the non-empty buffers with the rank ID
        if args.all_to_all_buff == "naive":
            for i in range(SIZE):
                buff_send[i].fill_(RANK)
        elif "neighbor" in args.all_to_all_buff:
            for i in neighbors:
                buff_send[i].fill_(RANK)

        # Perform the all_to_all
        tic = perf_counter()
        distnn.all_to_all(buff_recv, buff_send)
        if WITH_CUDA:
            torch.cuda.synchronize()
        elif WITH_XPU:
            torch.xpu.synchronize()
        toc = perf_counter()
        times.append(toc - tic)

        # Check that the received buffers have the expected value
        for i in range(SIZE):
            if buff_recv[i].numel() > 0:
                expected = i
                if not torch.all(buff_recv[i] == expected):
                    print(
                        f"[RANK {RANK}] Error: recv buffer from rank {i} does not match expected value {expected},",
                        f"recv buffer stats: min={torch.min(buff_recv[i])}, max={torch.max(buff_recv[i])}",
                        flush=True,
                    )
                    sys.exit(1)

    # Get timing stats
    # For Aurora, better to throw away first 10 iterations...
    if len(times) > 10:
        times = times[10:]
    avg_time = sum(times) / len(times)
    min_time = min(times)
    return avg_time, min_time


def main() -> None:
    # Parse arguments
    parser = ArgumentParser(
        description="PyTorch distributed nn alltoall benchmark"
    )
    parser.add_argument(
        "--all_to_all_buff",
        default="naive",
        type=str,
        choices=["naive", "neighbor", "semi-optimized"],
        help="Type of all_to_all buffers",
    )
    parser.add_argument(
        "--buff_size",
        default=1_000_000,
        type=int,
        help="Buffer size to the all_to_all in bytes. With --neighbors "
        "rcb_box this is the size of the largest buffer in the job and the "
        "rest scale down with contact area",
    )
    parser.add_argument(
        "--neighbors",
        default="nearest",
        type=str,
        choices=["nearest", "rcb_box"],
        help="Strategy for gathering neighbors. rcb_box reproduces the "
        "shooting workflow's rcb partitioning, sizes each buffer by the "
        "contact area with that neighbor, and ignores --num_neighbors",
    )
    parser.add_argument(
        "--num_neighbors",
        default=1,
        type=int,
        help="Number of neighbors involved in the all_to_all",
    )
    parser.add_argument(
        "--iterations", default=50, type=int, help="Number of iterations to run"
    )
    parser.add_argument(
        "--master_addr",
        default=None,
        type=str,
        help="Master address for torch.distributed",
    )
    parser.add_argument(
        "--master_port",
        default=None,
        type=int,
        help="Master port for torch.distributed",
    )
    parser.add_argument(
        "--logging",
        default="info",
        type=str,
        choices=["info", "verbose"],
        help="Verbosity of logging",
    )
    args = parser.parse_args()

    # Check arguments
    assert args.buff_size >= TORCH_ITEMSIZE, (
        f"Buffer size must be at least one element ({TORCH_ITEMSIZE} bytes)"
    )
    if args.buff_size % TORCH_ITEMSIZE and RANK == 0:
        print(
            f"Warning: --buff_size {args.buff_size} is not a multiple of the "
            f"{TORCH_ITEMSIZE}-byte element size; buffers are truncated to "
            f"{(args.buff_size // TORCH_ITEMSIZE) * TORCH_ITEMSIZE} bytes",
            flush=True,
        )
    if args.neighbors == "nearest" and SIZE > 1:
        assert args.num_neighbors * 2 <= SIZE, (
            "Number of neighbors x 2 must be less than or equal to the number of ranks"
        )

    if RANK == 0:
        print("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("RUNNING WITH INPUTS:")
        pprint(vars(args))
        print(
            "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n", flush=True
        )

    # Init torch distributed
    init_process_group()

    # Get neighbor ranks and build buffers
    neighbors, shared = get_neighbors(args)
    buffers = build_buffers(args, neighbors, shared)
    COMM.Barrier()

    # Run halo exchange
    avg_time, min_time = halo_exchange(args, neighbors, buffers)
    if RANK == 0:
        print("\n\nPerformance Summary:")
        print(f"Average all2all time: {avg_time:>.3e} sec", flush=True)
        print(f"Minimum all2all time: {min_time:>.3e} sec", flush=True)

    # Cleanup
    cleanup()


if __name__ == "__main__":
    main()
