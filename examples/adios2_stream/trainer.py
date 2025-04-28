import os
import numpy as np
from adios2 import Stream, Adios
from time import sleep

from mpi4py import MPI

# MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

color = 3230
app_comm = comm.Split(color, rank)
asize = app_comm.Get_size()
arank = app_comm.Get_rank()

# ADIOS MPI Communicator
adios = Adios(app_comm)

# ADIOS IO
sstIO = adios.declare_io("solutionStream")
sstIO.set_engine("SST")
parameters = {
    'DataTransport': 'WAN', # options: MPI, WAN,  UCX, RDMA
    'OpenTimeoutSecs': '600', # number of seconds SST is to wait for a peer connection on Open()
}
sstIO.set_parameters(parameters)

# Read the graph data 
while True:
    if os.path.exists('./graph.bp'):
        sleep(5)
        break
    else:
        sleep(5)
with Stream('graph.bp', 'r', comm) as stream:
    stream.begin_step()
    
    arr = stream.inquire_variable('N')
    N = stream.read('N', [rank], [1])
    N_list = comm.allgather(N)

    arr = stream.inquire_variable('num_edges')
    num_edges = stream.read('num_edges', [rank], [1])
    num_edges_list = comm.allgather(num_edges)

    arr = stream.inquire_variable('pos_node')
    count = N * 3
    start = sum(N_list[:rank]) * 3
    pos = stream.read('pos_node', [start], [count]).reshape((-1,3),order='F')

    arr = stream.inquire_variable('edge_index')
    count = num_edges * 2
    start = sum(num_edges_list[:rank]) * 2
    edge_index = stream.read('edge_index', [start], [count]).reshape((-1,2),order='F').T
                
    stream.end_step()

comm.Barrier()
if rank == 0: print('Done reading graph data', flush=True)

# Receive training data
workflow_steps = 5
try:
    if rank == 0: print('[ML] Opening stream ... ',flush=True)
    stream = Stream(sstIO, "solutionStream", "r", comm)
    for step in range(workflow_steps):
        sleep(5)
        if rank == 0: print('[ML] Reading solution data for step ',step,flush=True)
        stream.begin_step()
        var = stream.inquire_variable("U")
        count = N * 3
        start = sum(N_list[:rank]) * 3
        train_data = stream.read("U", [start], [count]).reshape((-1,3),order='F')
        stream.end_step()
        comm.Barrier()
        if rank == 0: print('[ML] Done reading solution data for step ',step,flush=True)
    stream.close()
except Exception as e:
    print(e)

MLrun = 0
with Stream('check-run.bp', 'w', comm) as stream:
    if rank == 0:
        stream.write("check-run", np.int32([MLrun]))

comm.Barrier()
if rank == 0: print('Trainer is done!')

