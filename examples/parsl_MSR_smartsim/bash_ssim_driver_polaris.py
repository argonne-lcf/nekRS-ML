# general imports
import os
import sys 
from argparse import ArgumentParser
import numpy as np

# smartsim and smartredis imports
from smartsim import Experiment
from smartsim.settings import RunSettings, PalsMpiexecSettings
#from smartredis import Client
from smartredis import Client, Dataset


import parsl
from parsl import bash_app
from config import local_config

# We will save outputs in the current working directory
working_directory = os.getcwd()

# Load config for polaris
parsl.load(local_config)

@bash_app
def hello_affinity(exe_args, wd, k, SSDB, stdout='hello.stdout', stderr='hello.stderr'):
    import os
    import sys
    os.environ['SSDB'] = SSDB
    return f'cd {wd}/MSR/nekrs/output{k} && cp ../../../case/* . && /home/viralss2/.local/nekrs-23/bin/nekrs {exe_args}'

# Command to launch nekRS correctly
# mpiexec -n 1 -ppn 1 /home/viralss2/.local/nekrs-23/bin/nekrs --setup msr.par --backend CUDA

## Define function to parse node list
def parseNodeList(fname):
    with open(fname) as file:
        nodelist = file.readlines()
        nodelist = [line.rstrip() for line in nodelist]
        nodelist = [line.split('.')[0] for line in nodelist]
    nNodes = len(nodelist)
    return nodelist, nNodes



## Clustered DB launch
def launch_clDB(args, nodelist, nNodes):
    # Split nodes between the components
    dbNodes_list = None
    if (nodelist is not None):
        simNodes = ','.join(nodelist[0: args.sim_nodes])
        dbNodes = ','.join(nodelist[args.sim_nodes: \
                                args.sim_nodes + args.db_nodes])
        dbNodes_list = nodelist[args.sim_nodes: \
                                args.sim_nodes + args.db_nodes]
        print(f"Database running on {args.db_nodes} nodes:")
        print(dbNodes)
        print(f"Simulatiom running on {args.sim_nodes} nodes:")
        print(simNodes)

    # Set up database and start it
    PORT = 6780
    launcher = 'pals'
    exp = Experiment('MSR', launcher=launcher)
    runArgs = {"np": 1, "ppn": 1, "cpu-bind": "numa"}
    kwargs = {
        'maxclients': 100000,
        'threads_per_queue': 4, # set to 4 for improved performance
        'inter_op_parallelism': 1,
        'intra_op_parallelism': 4,
        'cluster-node-timeout': 30000,
        }
    if (launcher=='pals'): run_command = 'mpiexec'
    db = exp.create_database(port=PORT, 
                             batch=False,
                             db_nodes=args.db_nodes,
                             run_command=run_command,
                             interface=['hsn0','hsn1'], 
                             hosts=dbNodes_list,
                             run_args=runArgs,
                             single_cmd=True,
                             **kwargs
                            )
    exp.generate(db)
    print("\nStarting database ...")
    exp.start(db)
    print("Done\n")

    # Initialize SmartRedis client
    cluster = False if args.db_nodes==1 else True
    client = Client(address=db.get_address()[0], cluster=cluster)

    # Set the distributions of the input variables
    vel_dist_params = np.array([0.0, 1.0, 1.3])
    client.put_tensor('vel_dist',vel_dist_params)

##    vel_dist_type = 'uni'
#    vel_dist_params = np.array([1.0, 1.3])
#    vel_dist_dataset = Dataset('velocity_distribution')
#    vel_dist_dataset.add_tensor('vel_params', vel_dist_params)
##    vel_dist_dataset.add_meta_string('vel_dist_type',vel_dist_type)
#    client.put_dataset(vel_dist_dataset)
    
    # Set up Parsl and launch the NekRS ensemble
    ## Pretend this is Parsl and NekRS
    client.put_tensor('train_data',np.ones((10,2)))

#    client_exe = args.sim_executable
#    if (launcher=='pals'):
#        SSDB = db.get_address()[0]
#        nrs_settings = PalsMpiexecSettings(client_exe,
#                                           exe_args=None,
#                                           run_args=None,
#                                           env_vars={'SSDB' : SSDB})
#        nrs_settings.set_tasks(args.simprocs)
#        nrs_settings.set_tasks_per_node(args.simprocs_pn)
#        nrs_settings.set_hostlist(simNodes)
#        nrs_settings.set_cpu_binding_type(args.sim_cpu_bind)
#        nrs_settings.add_exe_args(args.sim_arguments)
#
##        if (sim_affinity):
##            nrs_settings.set_gpu_affinity_script(sim_affinity,
##                                                 args.simprocs_pn)
#
#    nrs_model = exp.create_model("nekrs", nrs_settings)
#
#    # Start the client model
#    print("Launching the NekRS ...")
#    block = True
#    if len(args.sim_copy_files)>0 or len(args.sim_link_files)>0:
#        nrs_model.attach_generator_files(to_copy=list(args.sim_copy_files), to_symlink=list(args.sim_link_files))
#    exp.generate(nrs_model, overwrite=True)
#    exp.start(nrs_model, summary=False, block=block)
#    print("Done\n")

# Create futures calling 'hello_affinity', store them in list 'tasks'
    tasks = []
    SSDB = db.get_address()[0]
    for i in range(1):
        exe_arg = "--setup msr.par --backend CUDA"
        tasks.append(hello_affinity(exe_arg, f"{working_directory}", i, f"SSDB",
                                    stdout=f"{working_directory}/MSR/nekrs/output{i}/nekrs.out",
                                    stderr=f"{working_directory}/MSR/nekrs/output{i}/nekrs.err"))

    
    for i, t in enumerate(tasks):
        t.result()


    # Gather training data from DB and print
    train_data = client.get_tensor('train_data')
    inputs = train_data[0]
    outputs = train_data[1]
    print(f'Model inputs: {inputs}')
    print(f'Model outputs: {outputs}')

    # Stop database
    print("Stopping the Orchestrator ...")
    exp.stop(db)
    print("Done\n")


## Main function
def main():
    # Parse arguments
#    simargs="--setup msr.par --backend CUDA"

    parser = ArgumentParser(description='Online training from NekRS ensembles')
    parser.add_argument('--sim_nodes', default=1, type=int, help='Number of nodes assigned to the simulations')
    parser.add_argument('--db_nodes', default=1, type=int, help='Number of nodes assigned to the database')
    parser.add_argument('--simprocs_pn', default=4, type=int, help='Number of MPI processes per node for simulation')
    parser.add_argument('--simprocs', default=4, type=int, help='Number of MPI processes for simulation')
    parser.add_argument('--sim_cpu_bind', default="numa", help='CPU binding for simulation')
#    parser.add_argument('--db_nodes', default=1, type=int, help='Number of nodes assigned to the database')
#    parser.add_argument('--sim_arguments', default="${--setup msr.par --backend CUDA --device-id 0}", help='command line arguments to simulation')
    parser.add_argument('--sim_arguments', default="--setup msr.par --backend CUDA", help='command line arguments to simulation')
    parser.add_argument('--sim_executable', default="/home/viralss2/.local/nekrs-ml-msr/bin/nekrs", help='path to simulation executable ')
    parser.add_argument('--sim_affinity', default="", help='GPU affinity script for simulation')
    parser.add_argument('--sim_copy_files', default=["./msr.usr","./msr.par","./msr.udf","./msr.re2","./msr.oudf","msr.co2","./utilities.usr"], help='files to attach by copy to Model sub-directory')
    parser.add_argument('--sim_link_files', default=["./affinity_nrs.sh","restart.fld"], help='files to attach by symlink to Model sub-directory')

    args = parser.parse_args()

    # Get nodes of this allocation (job)
    nodelist = nNodes = None
    launcher = 'pals'
    if (launcher=='pals'):
        hostfile = os.getenv('PBS_NODEFILE')
        nodelist, nNodes = parseNodeList(hostfile)

    launch_clDB(args, nodelist, nNodes)

    # Quit
    print("Quitting")


## Run main
if __name__ == "__main__":
    main()
