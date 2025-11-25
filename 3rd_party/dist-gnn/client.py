import os
import sys
from typing import Optional, Union, Tuple
import logging
from omegaconf import DictConfig
import numpy as np
from time import sleep, perf_counter

# Import SmartRedis
try:
    from smartredis import Client, Dataset
except ModuleNotFoundError:
    pass

# Import ADIOS2
try:
    from adios2 import Stream, Adios
except ModuleNotFoundError:
    pass

log = logging.getLogger(__name__)

class OnlineClient:
    """Class for the online training client
    """
    def __init__(self, cfg: DictConfig, comm) -> None:
        self.client = None
        self.backend = cfg.client.backend
        self.comm = comm
        self.size = self.comm.Get_size()
        self.rank = self.comm.Get_rank()
        self.local_rank = int(os.getenv("PALS_LOCAL_RANKID"))
        self.local_size = int(os.getenv("PALS_LOCAL_SIZE"))

        # Initialize timers
        self.timers = self.setup_timers()

        # Initialize the client backend
        clients = ['smartredis', 'adios']
        if self.backend not in clients:
            sys.exit(f'Client {self.backend} not implemented. '\
                     f'Available options are: {clients}')
        self.init_client(cfg)

    def setup_timers(self) -> dict:
        """Setup timer dictionary to collect time spent on client ops 
        """
        timers= {}
        timers['init'] = []
        timers['data'] = []
        timers['meta_data'] = []
        return timers

    def init_client(self, cfg: DictConfig) -> None:
        """Initialize the client based on the specified backend
        """
        tic = perf_counter()
        if self.backend == 'smartredis':
            self.db_nodes = cfg.client.db_nodes
            SSDB = os.getenv('SSDB')
            if (self.db_nodes==1):
                self.client = Client(address=SSDB,cluster=False)
            else:
                self.client = Client(address=SSDB,cluster=True)
        elif self.backend == 'adios':
            self.engine = cfg.client.adios_engine
            self.transport = cfg.client.adios_transport
            adios = Adios(self.comm)
            self.client = adios.declare_io('streamIO')
            self.client.set_engine(self.engine)
            parameters = {
                'DataTransport': self.transport, # options: MPI, WAN, UCX, RDMA
                'OpenTimeoutSecs': '600', # number of seconds writer waits on Open() for reader
                'AlwaysProvideLatestTimestep': 'False', # True means reader will see only the newest available step
            }
            self.client.set_parameters(parameters)
            self.solutionStream = None
        self.timers['init'].append(perf_counter()-tic)

    def file_exists(self, file_name: str) -> bool:
        """Check if a file (or key) exists
        """
        tic = perf_counter()
        if self.backend == 'smartredis':
            return self.client.key_exists(file_name)
        self.timers['meta_data'].append(perf_counter()-tic)

    def get_array(self, file_name) -> np.ndarray:
        """Get an array frpm staging area / simulation
        """
        tic = perf_counter()
        if self.backend == 'smartredis':
            if isinstance(file_name, str):
                while True:
                    if self.file_exists(file_name):
                        array = self.client.get_tensor(file_name)
                        break
                    else:
                        sleep(0.5)
                        t_elapsed = perf_counter() - tic
                        if t_elapsed > 300:
                            sys.exit(f'Could not find {file_name} in DB')
            else:
                array = file_name.get_tensor('data')
        if self.backend == 'adios':
            var_name = file_name.split('.')[0]
            with Stream(file_name, 'r', self.comm) as stream:
                stream.begin_step()
                arr = stream.inquire_variable(var_name)
                shape = arr.shape()
                count = int(shape[0] / self.size)
                start = count * self.rank
                if self.rank == self.size - 1:
                    count += shape[0] % self.size
                array = stream.read(var_name, [start], [count])
                stream.end_step()
        self.timers['data'].append(perf_counter()-tic)
        return array
    
    def put_array(self, file_name: str, array: np.ndarray) -> None:
        """Put/send an array to staging area / simulation
        """
        if self.backend == 'smartredis':
            self.client.put_tensor(file_name, array)

    def get_file_list(self, list_name: str) -> list:
        """Get the list of files to read
        """
        tic = perf_counter()
        if self.backend == 'smartredis':
            # Ensure the list of DataSets is available
            while True:
                list_length = self.client.get_list_length(list_name)
                if list_length == 0:
                    sleep(1)
                    continue
                else:
                    break
            
            # Grab list of datasets
            file_list = self.client.get_datasets_from_list(list_name)
        self.timers['meta_data'].append(perf_counter()-tic)
        return file_list
    
    def get_file_list_length(self, list_name: str) -> int:
        """Get the length of the file list
        """
        tic = perf_counter()
        if self.backend == 'smartredis':
            list_length = self.client.get_list_length(list_name)
        self.timers['meta_data'].append(perf_counter()-tic)
        return list_length
        
    def get_graph_data_from_stream(self) -> dict:
        """Get the entire set of graph datasets from a stream
        """
        tic = perf_counter()
        graph_data = {}
        if self.backend == 'adios':
            while True:
                if os.path.exists('./graph.bp'):
                    sleep(1)
                    break
                else:
                    sleep(2)

            #with Stream(self.client, 'graphStream', 'r', self.comm) as stream:
            with Stream('graph.bp', 'r', self.comm) as stream:
                stream.begin_step()
                
                graph_data['Np'] = int(stream.read('Np'))

                arr = stream.inquire_variable('N')
                N = stream.read('N', [self.rank], [1])
                print(f'[RANK {self.rank}] -- N: {N}',flush=True)
                self.N_list = self.comm.allgather(N)

                arr = stream.inquire_variable('num_edges')
                num_edges = stream.read('num_edges', [self.rank], [1])
                self.num_edges_list = self.comm.allgather(num_edges)

                arr = stream.inquire_variable('field_offset')
                field_offset = stream.read('field_offset', [self.rank], [1])
                print(f'[RANK {self.rank}] -- field_offset: {field_offset}',flush=True)
                self.field_offset_list = self.comm.allgather(field_offset)

                arr = stream.inquire_variable('pos_node')
                count = N * 3
                start = sum(self.N_list[:self.rank]) * 3
                graph_data['pos'] = stream.read('pos_node', [start], [count]).reshape((-1,3),order='F')

                arr = stream.inquire_variable('edge_index')
                count = num_edges * 2
                start = sum(self.num_edges_list[:self.rank]) * 2
                graph_data['edge_index'] = stream.read('edge_index', [start], [count]).reshape((-1,2),order='F').T

                arr = stream.inquire_variable('global_ids')
                count = N
                start = sum(self.N_list[:self.rank])
                graph_data['global_ids'] = stream.read('global_ids', [start], [count])

                arr = stream.inquire_variable('local_unique_mask')
                count = N
                start = sum(self.N_list[:self.rank])
                graph_data['local_unique_mask'] = stream.read('local_unique_mask', [start], [count])

                arr = stream.inquire_variable('halo_unique_mask')
                count = N
                start = sum(self.N_list[:self.rank])
                graph_data['halo_unique_mask'] = stream.read('halo_unique_mask', [start], [count])
                
                stream.end_step()
        self.timers['data'].append(perf_counter()-tic)
        return graph_data

    def get_train_data_from_stream(self) -> Tuple[np.ndarray,np.ndarray]:
        """Get the solution from a stream
        """
        self.comm.Barrier()
        tic = perf_counter()
        if self.backend == 'adios':
            if self.solutionStream is None:
                if self.rank == 0: log.info('Opening ADIOS2 solutionStream ...')
                self.solutionStream = Stream(self.client, "solutionStream", "r", self.comm)
                         
            self.solutionStream.begin_step()

            arr = self.solutionStream.inquire_variable('out_u')
            count = self.field_offset_list[self.rank] * 3
            start = sum(self.field_offset_list[:self.rank]) * 3
            # stream.read() gets data now, Mode.Sync is default 
            # see 
            #   - https://github.com/ornladios/ADIOS2/blob/67f771b7a2f88ce59b6808cc4356159d86255f1d/python/adios2/stream.py#L331
            #   - https://github.com/ornladios/ADIOS2/blob/67f771b7a2f88ce59b6808cc4356159d86255f1d/python/adios2/engine.py#L123)
            ticc = perf_counter()
            outputs = self.solutionStream.read('out_u', [start], [count])
            transfer_time = perf_counter() - ticc
            outputs = outputs.reshape((-1,3),order='F')

            arr = self.solutionStream.inquire_variable('in_u')
            ticc = perf_counter()
            inputs = self.solutionStream.read('in_u', [start], [count])
            transfer_time += perf_counter() - ticc
            inputs = inputs.reshape((-1,3),order='F')

            self.solutionStream.end_step()
        self.timers['data'].append(perf_counter()-tic)
        return inputs, outputs, transfer_time

    def stop_nekRS(self) -> None:
        """Communicate to nekRS to stop running and exit cleanly
        """
        MLrun = 0
        tic = perf_counter()
        if self.backend == 'smartredis':
            if self.db_nodes == 1:
                if self.rank % self.local_size == 0:
                    self.put_array('check-run',
                                    np.int32(np.array([MLrun]))
                    )
            else:
                if self.rank == 0:
                    self.put_array('check-run',
                                    np.int32(np.array([MLrun]))
                    )
        elif self.backend == 'adios':
            # Close solution stream
            if self.solutionStream is not None:
                self.solutionStream.close()
            
            # Communicate to nekRS to stop
            with Stream('check-run.bp', 'w', self.comm) as stream:
                if self.rank == 0:
                    stream.write("check-run", np.int32([MLrun]))
        self.timers['meta_data'].append(perf_counter()-tic)
            

