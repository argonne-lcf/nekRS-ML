"""
Utilities for training and inferencing
"""

import sys
from typing import Optional, Union, Callable, Tuple
import time
import logging
import numpy as np
import torch
import torch.distributed as dist

Tensor = torch.Tensor
log = logging.getLogger(__name__)

try:
    import mpi4py

    mpi4py.rc.initialize = False
    from mpi4py import MPI

    if not MPI.Is_initialized():
        MPI.Init()
    COMM = MPI.COMM_WORLD
    RANK = COMM.Get_rank()
    SIZE = COMM.Get_size()
    WITH_DDP = True
except ModuleNotFoundError as e:
    WITH_DDP = False
    pass

try:
    WITH_CUDA = torch.cuda.is_available()
except:
    WITH_CUDA = False
    pass

try:
    WITH_XPU = torch.xpu.is_available()
except:
    WITH_XPU = False
    pass


def init_process_group(
    rank: Union[int, str],
    world_size: Union[int, str],
    backend: Optional[str] = None,
) -> None:
    if WITH_CUDA:
        backend = "nccl" if backend is None else str(backend)
    elif WITH_XPU:
        backend = "xccl" if backend is None else str(backend)
    else:
        backend = "gloo" if backend is None else str(backend)

    dist.init_process_group(
        backend,
        rank=int(rank),
        world_size=int(world_size),
        init_method="env://",
    )


def cleanup():
    dist.destroy_process_group()


def force_abort():
    time.sleep(2)
    if WITH_DDP:
        COMM.Abort()
    else:
        sys.exit("Exiting...")


def metric_average(val: Tensor):
    if WITH_DDP:
        # dist.all_reduce(val, op=dist.ReduceOp.SUM)
        dist.reduce(val, 0, op=dist.ReduceOp.SUM)
        return val / SIZE
    return val


def metric_min(val: Tensor):
    if WITH_DDP:
        dist.all_reduce(val, op=dist.ReduceOp.MIN)
    return val


def metric_max(val: Tensor):
    if WITH_DDP:
        dist.all_reduce(val, op=dist.ReduceOp.MAX)
    return val


def all_gather_tensor(tensor_list: list[Tensor], tensor_local: Tensor):
    if WITH_DDP:
        dist.all_gather(tensor_list, tensor_local)
        return tensor_list
    return [tensor_local]


def mpi_all_gather(local_obj):
    if WITH_DDP:
        obj_list = COMM.allgather(local_obj)
        return obj_list
    return [local_obj]


def trace_handler(p):
    output = p.key_averages().table(sort_by="self_cuda_time", row_limit=20)
    print(output)
    p.export_chrome_trace("/tmp/trace_" + str(p.step_num) + ".json")


def collect_list_times(a_list):
    collected_arr = np.zeros((len(a_list) * SIZE))
    COMM.Gather(np.array(a_list), collected_arr, root=0)
    avg = np.mean(collected_arr)
    std = np.std(collected_arr)
    minn = np.amin(collected_arr)
    min_loc = [minn, 0]
    maxx = np.amax(collected_arr)
    max_loc = [maxx, 0]
    summ = np.sum(collected_arr)
    stats = {
        "avg": avg,
        "std": std,
        "sum": summ,
        "min": [min_loc[0], min_loc[1]],
        "max": [max_loc[0], max_loc[1]],
    }
    return stats


def average_list_times(a_list):
    sum_across_ranks = np.zeros((len(a_list)))
    COMM.Reduce(np.array(a_list), sum_across_ranks, op=MPI.SUM)
    avg = np.mean(sum_across_ranks)
    return avg


def collect_stats(
    n_nodes_local: int, local_time: list, local_throughput: list
) -> dict:
    n_nodes_global = np.array(0)
    gather_time = np.zeros(len(local_time) * SIZE)
    gather_throughput = np.zeros(len(local_throughput) * SIZE)
    COMM.Allreduce(np.array(n_nodes_local), n_nodes_global)
    COMM.Allgather(np.array(local_time), gather_time)
    COMM.Allgather(np.array(local_throughput), gather_throughput)
    global_throughput = np.zeros(len(local_throughput), dtype=np.float32)
    COMM.Allreduce(
        np.array(local_throughput).astype(np.float32),
        global_throughput,
        op=MPI.SUM,
    )
    return {
        "n_nodes": n_nodes_global,
        "time": gather_time,
        "throughput": gather_throughput,
        "glob_throughput": global_throughput,
    }


def collect_online_stats(local_time: list, local_throughput: list) -> dict:
    gather_time = np.zeros(len(local_time) * SIZE)
    gather_time_tot = np.zeros(SIZE)
    gather_throughput = np.zeros(len(local_throughput) * SIZE)
    COMM.Allgather(np.array(local_time), gather_time)
    COMM.Allgather(np.array(sum(local_time)), gather_time_tot)
    COMM.Allgather(np.array(local_throughput), gather_throughput)
    global_throughput = np.zeros(len(local_throughput))
    COMM.Allreduce(np.array(local_throughput), global_throughput, op=MPI.SUM)
    return {
        "time": gather_time,
        "tot_time": gather_time_tot,
        "throughput": gather_throughput,
        "glob_throughput": global_throughput,
    }


def min_max_avg(data: Union[list, np.ndarray]) -> Tuple[float, float, float]:
    if isinstance(data, list):
        min_val = min(data)
        max_val = max(data)
        avg_val = sum(data) / len(data)
    elif isinstance(data, np.ndarray):
        min_val = np.amin(data)
        max_val = np.amax(data)
        avg_val = np.mean(data)
    else:
        min_val = max_val = avg_val = 0.0
    return min_val, max_val, avg_val
