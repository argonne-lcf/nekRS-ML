import parsl
import os
from parsl import bash_app
from config import local_config

# We will save outputs in the current working directory
working_directory = os.getcwd()

# Load config for polaris
parsl.load(local_config)
#parsl.load()

# Application that reports which worker affinities
@bash_app
def hello_affinity(exe_args, wd, k, stdout='hello.stdout', stderr='hello.stderr'):
    #return f'cd {wd}/output{k} && cp ../case/* . && mpiexec -n 1 -ppn 1 /home/viralss2/.local/nekrs-23/bin/nekrs {exe_args}'
    return f'cd {wd}/output{k} && cp ../case/* . && /home/viralss2/.local/nekrs-23/bin/nekrs {exe_args}'

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
