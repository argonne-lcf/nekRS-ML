import os
from parsl.config import Config

from parsl.providers import LocalProvider
from parsl.executors import HighThroughputExecutor
from parsl.launchers import MpiExecLauncher

from parsl.addresses import address_by_interface

# The config will launch workers from this directory
execute_dir = os.getcwd()

nodes_per_job = 1
max_num_jobs = 1

local_config = Config(
    executors=[
        HighThroughputExecutor(
            # Ensures one worker per accelerator
            #available_accelerators=["0","1","2","3"],
            available_accelerators=["0,1,2,3"],
            address=address_by_interface('bond0'),
            # Distributes threads to workers sequentially in reverse order
            cpu_affinity="block-reverse",
            # Increase if you have many more tasks than workers
            prefetch_capacity=0,
            # Needed to avoid interactions between MPI and os.fork
            # start_method="spawn",
            provider=LocalProvider(
                worker_init=f'''source /home/viralss2/ALCF_Hands_on_HPC_Workshop/workflows/parsl/launch_eg/set_run_env;
                            cd {execute_dir}''',
                # Number of nodes per batch job
                nodes_per_block=nodes_per_job,
                # Minimum number of batch jobs running workflow
                min_blocks=0,
                # Maximum number of batch jobs running workflow
                max_blocks=max_num_jobs,
                # Threads per node
                #cores_per_node=64,
            ),
        ),
    ],
    # Retry failed tasks once
    retries=1,
)
