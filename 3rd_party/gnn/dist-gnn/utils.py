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

import mpi4py.rc
mpi4py.rc.initialize = False
from mpi4py import MPI


def collect_list_times(a_list, COMM):
    collected_arr = np.zeros((len(a_list)*COMM.Get_size()))
    COMM.Gather(np.array(a_list),collected_arr,root=0)
    avg = np.mean(collected_arr)
    std = np.std(collected_arr)
    minn = np.amin(collected_arr); min_loc = [minn, 0]
    maxx = np.amax(collected_arr); max_loc = [maxx, 0]
    summ = np.sum(collected_arr)
    stats = {
                "avg": avg,
                "std": std,
                "sum": summ,
                "min": [min_loc[0],min_loc[1]],
                "max": [max_loc[0],max_loc[1]]
    }
    return stats

def average_list_times(a_list, COMM):
    sum_across_ranks = np.zeros((len(a_list)))
    COMM.Reduce(np.array(a_list),sum_across_ranks,op=MPI.SUM)
    avg = np.mean(sum_across_ranks)
    return avg

def collect_stats(COMM,n_nodes_local: int, local_time: list, local_throughput: list) -> dict:
    n_nodes_global = np.array(0)
    gather_time = np.zeros(len(local_time)*COMM.Get_size())
    gather_throughput = np.zeros(len(local_throughput)*COMM.Get_size())
    COMM.Allreduce(np.array(n_nodes_local),n_nodes_global)
    COMM.Allgather(np.array(local_time), gather_time)
    COMM.Allgather(np.array(local_throughput), gather_throughput)
    global_throughput = np.zeros(len(local_throughput), dtype=np.float32)
    COMM.Allreduce(np.array(local_throughput).astype(np.float32), global_throughput, op=MPI.SUM)
    return {'n_nodes':n_nodes_global, 
            'time':gather_time, 
            'throughput':gather_throughput, 
            'glob_throughput':global_throughput
            }

def collect_online_stats(COMM, local_time: list, local_throughput: list) -> dict:
    gather_time = np.zeros(len(local_time)*COMM.Get_size())
    gather_time_tot = np.zeros(COMM.Get_size())
    gather_throughput = np.zeros(len(local_throughput)*COMM.Get_size())
    COMM.Allgather(np.array(local_time), gather_time)
    COMM.Allgather(np.array(sum(local_time)), gather_time_tot)
    COMM.Allgather(np.array(local_throughput), gather_throughput)
    global_throughput = np.zeros(len(local_throughput))
    COMM.Allreduce(np.array(local_throughput), global_throughput, op=MPI.SUM)
    return { 
            'time':gather_time, 
            'tot_time':gather_time_tot,
            'throughput':gather_throughput, 
            'glob_throughput':global_throughput
            }

def min_max_avg(data: Union[list, np.ndarray]) -> Tuple[float,float,float]:
    if isinstance(data,list):
        min_val = min(data)
        max_val = max(data)
        avg_val = sum(data) / len(data)
    elif isinstance(data,np.ndarray):
        min_val = np.amin(data)
        max_val = np.amax(data)
        avg_val = np.mean(data)
    else:
        min_val = max_val = avg_val = 0.
    return min_val, max_val, avg_val


    
