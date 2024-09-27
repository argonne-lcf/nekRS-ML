import os
import numpy as np
from time import sleep

from mpi4py import MPI


# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
print(f"Hello from rank {rank}/{size}",flush=True)

# Pretend to do some calculations for a few time steps
for step in range(5):
    sleep(5)
    maxT = np.random.uniform(low=0., high=1., size=1)[0]
    if rank==0: print(f'\nComputed max Temp = {maxT}\n',flush=True)

    comm.Barrier()

# Exit
if rank==0: print('\nExiting ...',flush=True)


