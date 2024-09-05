import os
import parsl
from parsl.config import Config
from parsl import bash_app
from parsl.providers import LocalProvider
from parsl.executors import HighThroughputExecutor
from parsl.launchers import MpiExecLauncher
from parsl.channels import LocalChannel

from parsl.addresses import address_by_interface

# The config will launch workers from this directory                                                                                                        
execute_dir = os.getcwd()

nodes_per_job = 2
max_num_jobs = 1
local_config = Config(
  executors=[
    HighThroughputExecutor(
      # Ensures one worker per accelerator                                                                                                                  
      available_accelerators=["0","1","2","3"],
      #available_accelerators=["0,1,2,3"],                                                                                                                  
      address=address_by_interface('bond0'),
      # Distributes threads to workers sequentially in reverse order                                                                                        
      cpu_affinity="block-reverse",
      cores_per_worker=8,
      provider=LocalProvider(
        #channel=LocalChannel(script_dir='.'),                                                                                                              
        # Number of nodes per batch job                                                                                                                     
        nodes_per_block=nodes_per_job,
        launcher=MpiExecLauncher(bind_cmd="--cpu-bind", overrides="--depth=16"),
        worker_init=f'''module use /soft/modulefiles; module load conda; conda activate balsam''',
        init_blocks=1,
        # Minimum number of batch jobs running workflow                                                                                                     
        #min_blocks=0,                                                                                                                                      
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

parsl.load(local_config)

#@bash_app
#def echo_app(stdout=parsl.AUTO_LOGNAME):
#    return 'hostname; echo $CUDA_VISIBLE_DEVICES; sleep 2'
#
#futures = []
#for i in range(8):
#    futures.append(echo_app())
#
#for t in futures:
#    t.result()
#    with open(t.stdout,'r') as f:
#        print(f.read())


working_directory = os.getcwd()
@bash_app
def hello_affinity(exe_args, wd, k, stdout='hello.stdout', stderr='hello.stderr'):
    return f'cd {wd}/output{k} && cp ../case/* . && mpiexec -n 1 -ppn 1 /home/viralss2/.local/nekrs-23/bin/nekrs {exe_args}'
    #return f'cd {wd}/output{k} && cp ../case/* . && /home/viralss2/.local/nekrs-23/bin/nekrs {exe_args}'

#mpiexec -n 4 -ppn 4 --bind-to {numa} /home/viralss2/.local/nekrs-23/bin/nekrs {exe_args}

# Create futures calling 'hello_affinity', store them in list 'tasks'
tasks = []
for i in range(8):
    exe_arg = "--setup msr.par --backend CUDA"
    tasks.append(hello_affinity(exe_arg, f"{working_directory}", i, stdout=f"{working_directory}/output{i}/nekrs.out",
                                stderr=f"{working_directory}/output{i}/nekrs.err"))

# Wait on futures to return, and print results
for i, t in enumerate(tasks):
    t.result()
    with open(f"{working_directory}/output{i}/nekrs.out", "r") as f:
        print(f.read())

# Workflow complete!
print("Hello tasks completed")
