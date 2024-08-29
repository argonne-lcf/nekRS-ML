# general imports
import os
import sys 
#from omegaconf import DictConfig, OmegaConf
#import hydra
from argparse import ArgumentParser
import numpy as np

# smartsim and smartredis imports
from smartsim import Experiment
from smartsim.settings import RunSettings, PalsMpiexecSettings
#from smartredis import Client
from smartredis import Client, Dataset

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

    # Define the simulation settings
    if (launcher=='pals'):
        SSDB = db.get_address()[0]
        nrs_settings = PalsMpiexecSettings('python',
                                           exe_args="/eagle/datascience/balin/Nek/nekRS-ML_ConvReac/examples/workflow_sketch/nrs.py",
                                           run_args=None,
                                           env_vars={'SSDB' : SSDB})
        nrs_settings.set_tasks(args.simprocs)
        nrs_settings.set_tasks_per_node(args.simprocs_pn)
        nrs_settings.set_hostlist(simNodes)

    # Launch a few simulations
    # Try:
    #    - using own logic, SmartSim API and mp.Process to manage different models and lauch them on resources
    for isim in range(1):
        nrs_model = exp.create_model(f"nekrs_{isim}", nrs_settings)

        # Start the client model
        print(f"Launching the NekRS simulation number {isim} ...")
        block = False
        #if len(args.sim_copy_files)>0 or len(args.sim_link_files)>0:
        #    nrs_model.attach_generator_files(to_copy=list(args.sim_copy_files), to_symlink=list(args.sim_link_files))
        exp.generate(nrs_model, overwrite=True)
        exp.start(nrs_model, summary=False, block=block)
        print("Done\n")
   
    # Launch a trainer 
    if (launcher=='pals'):
        SSDB = db.get_address()[0]
        train_settings = PalsMpiexecSettings('python',
                                           exe_args="/eagle/datascience/balin/Nek/nekRS-ML_ConvReac/examples/workflow_sketch/trainer.py",
                                           run_args=None,
                                           env_vars={'SSDB' : SSDB})
        train_settings.set_tasks(1)
        train_settings.set_tasks_per_node(1)
        train_settings.set_hostlist(simNodes)
    
    train_model = exp.create_model("trainer", train_settings)
    print(f"Launching the trainer ...")
    block = True
    exp.generate(train_model, overwrite=True)
    exp.start(train_model, summary=False, block=block)
    print("Done\n")

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
