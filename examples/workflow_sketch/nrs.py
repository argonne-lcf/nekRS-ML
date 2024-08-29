import os
import numpy as np
from time import sleep

from mpi4py import MPI

from smartredis import Client, Dataset

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
print(f"Hello from rank {rank}/{size}",flush=True)

# Initialize SR Client
SSDB = os.getenv('SSDB')
client = Client(address=SSDB,cluster=False)
comm.Barrier()
if rank==0: print('\nAll clients initialized\n',flush=True)

# Get input data from DB
vel_dist = client.get_tensor('vel_dist')
if vel_dist[0]<0.5:
    if rank==0: print(f'\nRead velocity ditribution of uniform type in range {vel_dist[1]} - {vel_dist[2]}\n',flush=True)
inflow_vel = np.random.uniform(low=vel_dist[1], high=vel_dist[2], size=1)[0]

# Pretend to do some calculations for a few time steps
for step in range(5):
    sleep(5)
    maxT = np.random.uniform(low=0., high=1., size=1)[0]
    if rank==0: print(f'\nComputed max Temp = {maxT}\n',flush=True)

    # From rank 0, create a DataSet with training data and append to a list
    if rank==0:
        train_dataset = Dataset(f'train_data_{maxT:>3f}')
        train_dataset.add_tensor('train', np.array([inflow_vel, maxT]))
        client.put_dataset(train_dataset)
        client.append_to_list('training_list',train_dataset)

    comm.Barrier()

# Exit
if rank==0: print('\nExiting ...',flush=True)


