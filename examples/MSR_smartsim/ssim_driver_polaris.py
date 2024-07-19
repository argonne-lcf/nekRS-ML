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
from smartredis import Client

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
    vel_dist_type = 'uniform'
    vel_dist_params = np.array([1.0, 1.3])
    vel_dist_dataset = Dataset('velocity_distribution')
    vel_dist_dataset.add_tensor('vel_params', vel_dist_params)
    vel_dist_dataset.add_meta_string('vel_dist_type',vel_dist_type)
    client.put_dataset(vel_dist_dataset)
    
    # Set up Parsl and launch the NekRS ensemble
    ## Pretend this is Parsl and NekRS
    client.put_tensor('train_data',np.ones((10,2)))
   
    # Gather training data from DB and print
    train_data = client.get_tensor('train_data')
    intputs = train_data[:,0]
    outputs = train_data[:,1]
    print(f'Model inputs: {inputs}')
    print(f'Model outputs: {outputs}')

    # Stop database
    print("Stopping the Orchestrator ...")
    exp.stop(db)
    print("Done\n")


## Main function
def main():
    # Parse arguments
    parser = ArgumentParser(description='Online training from NekRS ensembles')
    parser.add_argument('--sim_nodes', default=1, type=int, help='Number of nodes assigned to the simulations')
    parser.add_argument('--db_nodes', default=1, type=int, help='Number of nodes assigned to the database')
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
