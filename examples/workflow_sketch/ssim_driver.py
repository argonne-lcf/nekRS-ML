# general imports
import os
import sys 
#from omegaconf import DictConfig, OmegaConf
#import hydra
from argparse import ArgumentParser
import numpy as np
import multiprocessing as mp
from time import sleep
import psutil

# smartsim and smartredis imports
#import smartsim
from smartsim import Experiment
from smartsim.settings import RunSettings, PalsMpiexecSettings
from smartsim._core.launcher.taskManager import Task
from smartredis import Client, Dataset

## Define function to parse node list
def parseNodeList(fname):
    with open(fname) as file:
        nodelist = file.readlines()
        nodelist = [line.rstrip() for line in nodelist]
        nodelist = [line.split('.')[0] for line in nodelist]
    nNodes = len(nodelist)
    return nodelist, nNodes


## Launch a SmartSim Model for NekRS
def launch_nrs(q, launch_id, node_list, gpu_list, run_settings, experiment):
    # Launch the first model
    run_id = 0
    run_settings.set_hostlist(','.join(node_list))
    #run_settings.update_env({'MPICH_OFI_CXI_PID_BASE':str(launch_id)})
    nrs_model = experiment.create_model(f"nekrs_{launch_id}_{run_id}", run_settings)
    print(f"Launching a new NekRS simulation from launcher process {launch_id} on nodes {node_list} and GPUs {gpu_list}",flush=True)
    experiment.generate(nrs_model, overwrite=True)
    task_id = experiment.start(nrs_model, summary=False, block=False)
    task = Task(psutil.Process(int(task_id)))
 
    # Check status and launch another one when finished
    while True:
        sleep(5)

        # Read from queue whether to quit simulation
        try:
            run_check = q.get(block=False)
            if run_check=='stop':
                experiment.stop(nrs_model)
                break
        except Exception as e:
            pass

        # Get simulation status and relaunch if done
        #ssim_status = experiment.get_status(nrs_model)[0] # this always prints STATUS_NEW because manager process doesn't update it
        #print(f'Task status: {status}',flush=True) 
        #if status==smartsim.status.SmartSimStatus.STATUS_NEW:
        #    print(f'nekrs_{launch_id}_{run_id} status: New',flush=True)
        #elif status==smartsim.status.SmartSimStatus.STATUS_RUNNING:
        #    print(f'nekrs_{launch_id}_{run_id} status: Running',flush=True)
        #elif status==smartsim.status.SmartSimStatus.STATUS_COMPLETED:
        
        status = task.status
        if status!='zombie':
            print(f'nekrs_{launch_id}_{run_id} status: Running',flush=True)
        else:
            # kill current task
            task.terminate(timeout=1)
            
            # launch a new model
            run_id+=1
            print(f"Launching a new NekRS simulation from launcher process {launch_id} on nodes {node_list} and GPUs {gpu_list}",flush=True)
            nrs_model = experiment.create_model(f"nekrs_{launch_id}_{run_id}", run_settings)
            experiment.generate(nrs_model, overwrite=True)
            task_id = experiment.start(nrs_model, summary=False, block=False)
            task = Task(psutil.Process(int(task_id)))


## Clustered DB launch
def launch_clDB(args, nodelist, nNodes):
    # Split nodes between the components
    dbNodes_list = None
    if (nodelist is not None):
        #simNodes = ','.join(nodelist[0: args.sim_nodes])
        #simNodes_list = nodelist[0: args.sim_nodes]
        #dbNodes = ','.join(nodelist[args.sim_nodes: \
        #                        args.sim_nodes + args.db_nodes])
        #dbNodes_list = nodelist[args.sim_nodes: \
        #                        args.sim_nodes + args.db_nodes]
        
        dbNodes = ','.join(nodelist[0: args.db_nodes])
        dbNodes_list = nodelist[0: args.db_nodes]
        simNodes = ','.join(nodelist[args.db_nodes: \
                                args.db_nodes + args.sim_nodes])
        simNodes_list = nodelist[args.db_nodes: \
                                args.db_nodes + args.sim_nodes]
        
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
    _ = exp.start(db)
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
#        nrs_settings = PalsMpiexecSettings('python',
#                                           exe_args="/eagle/datascience/balin/Nek/nekRS-ML_ConvReac/examples/workflow_sketch/nrs.py",
#                                           run_args=None,
#                                           env_vars={'SSDB' : SSDB})
        nrs_settings = PalsMpiexecSettings('/eagle/ConvReac/viralss2/nekRS-ML_MSR/examples/workflow_sketch/a.out',
                                           exe_args=None,
                                           run_args=None,
                                           env_vars={'SSDB' : SSDB})
        nrs_settings.set_tasks(args.simprocs)
        nrs_settings.set_tasks_per_node(args.simprocs_pn)

    # Launch the NekRS launcher processes
    n_gpu_pn = 4
    n_launchers = int((args.sim_nodes*n_gpu_pn)/args.simprocs)
    n_nodes_nrs_runs = float(args.simprocs/n_gpu_pn)
    #print(f'\n{n_nodes_nrs_runs=}')
    processes = []
    queues = []
    print(f'\nSetting up {n_launchers} NekRS launcher processes',flush=True)
    for launch_id in range(n_launchers):
        if args.sim_nodes==1:
            process_node_list = list(simNodes_list)
        else:
            node_id_start = int((n_nodes_nrs_runs*launch_id)) #//args.sim_nodes)
            node_id_end = node_id_start+int(n_nodes_nrs_runs)
            print(f'{launch_id}: launching on node indices {node_id_start} - {node_id_end}')
            process_node_list = simNodes_list[node_id_start:node_id_end] if node_id_start!=node_id_end else simNodes_list[node_id_start]
        if args.simprocs_pn==n_gpu_pn:
            gpu_list = [i for i in range(n_gpu_pn)]
        else:
            gpu_list = [int(i+n_nodes_nrs_runs*n_gpu_pn*launch_id)%n_gpu_pn for i in range(args.simprocs_pn)]
            #print(f'{launch_id}: gpu {gpu_list}')
        
        q = mp.Queue()
        queues.append(q)
        p = mp.Process(target=launch_nrs, args=(q, launch_id, process_node_list, gpu_list, nrs_settings, exp))
        processes.append(p)

    print(f'Starting the NekRS launcher processes',flush=True)
    for p in processes:
        p.start()
    print("Done\n")


    # Launch a trainer 
    if (launcher=='pals'):
        SSDB = db.get_address()[0]
        train_settings = PalsMpiexecSettings('python',
                                           exe_args="/eagle/ConvReac/viralss2/nekRS-ML_MSR/examples/workflow_sketch/trainer.py",
                                           run_args=None,
                                           env_vars={'SSDB' : SSDB})
        train_settings.set_tasks(1)
        train_settings.set_tasks_per_node(1)
        train_settings.set_hostlist(simNodes)
    
    train_model = exp.create_model("trainer", train_settings)
    print(f"Launching the trainer ...")
    exp.generate(train_model, overwrite=True)
    _ = exp.start(train_model, summary=False, block=True)
    print("Done\n")

    # Tell the launchers to stop the simulations
    print("Stopping the simulations",flush=True)
    for q in queues:
        q.put("stop")
    print("Done\n",flush=True)
    
    # Join the launcher processes
    print("Joining the launcher processes",flush=True)
    for p in processes:
        p.join()
    print("Done\n",flush=True)

    # Stop database
    print("Bringing down the database ...")
    exp.stop(db)
    print("Done\n")


## Main function
def main():
    # Parse arguments
#    simargs="--setup msr.par --backend CUDA"

    parser = ArgumentParser(description='Online training from NekRS ensembles')
    parser.add_argument('--sim_nodes', default=1, type=int, help='Number of nodes assigned to the simulations')
    parser.add_argument('--db_nodes', default=1, type=int, help='Number of nodes assigned to the database')
    parser.add_argument('--simprocs_pn', default=1, type=int, help='Number of MPI processes per node for simulation')
    parser.add_argument('--simprocs', default=1, type=int, help='Number of MPI processes for simulation')
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
