# Solution shooting workflow with a GNN surrogate model using SmartSim

This example builds off of the [online training of time dependent GNN surrogate](../tgv_gnn_traj_online/README.md), however it adds the step of performing inference with the GNN surrogate after online training has concluded. 
The flow problem is based on the turbulence channel flow LES, for which the details are in the [turbChannel example](../turbChannel/README.md).

The main differences between this example and simple online training of time dependent GNN surrogate are in the `driver.py` workflow driver script. 
Specifically, the workflow runner alternates between fine-tuning of the GNN and deploying the model for inference.
During fine-tuning, both nekRS and GNN training are running concurrently.
During inference, only the GNN is run advanding the velocity field in time.
As usual, the plugins are called from  `UDF_Setup()` and `UDF_ExecuteStep()`. 
Note that at the end of the run, nekRS writes a checkpoint to the SmartSim database, which is used as initial condition to GNN inference.

## Building nekRS

To build nekRS with the GNN plugin, simply execute the build script [BuildMeOnAurora](../../BuildMeOnAurora) from the top directory
```bash
source BuildMeOnAurora
```

NOTE: you must enable building SmartRedis for this example since it is performing online training with the SmartRedis backend. 

## Runnig the example

Scripts are provided to conveniently generate run scripts and config files for the workflow on the different ALCF systems.
Note that a virtual environment with SmartSim and PyTorch Geometric is needed for the GNN.
If you don't specify a virtual environment path, the script will create one for you.
From an interactive session on the compute nodes, first execute
```bash
./gen_run_script
```

taking notice of some of the variables to set. 
Specifically, make sure to set 

```
SYSTEM # the ALCF system to run on (aurora, polaris)
DEPLOYMENT # the deployment strategy for the workflow (colocated, clustered)
NEKRS_HOME # path to the nekRS install directory
VENV_PATH # path to the Python venv activate script
PROJ_ID # project name for the allocation
QUEUE # name of the queue to run on
```

The script generates the run script, which is executed with
```bash
./run.sh
```

The `run.sh` script is composed of two steps:

- First nekRS is run by itself with the `--build-only` flag. This is done such that the `.cache` directory can be built beforehand instead of during online training.
- The online training workflow driver `driver.py` is executed with Python, setting up the SmartSim Orchestrator (the database), followed by fine-tuning involving nekRS and the GNN trainer, followed by inference once fine-tuning is over.

The outputs of the nekRS, trainer and inference will be within the `./nekRS` directory created at runtime.

## Known Issues and Tips
- The clustered deployment requires at least 3 nodes for training and 2 nodes for inference since it deploys each component on a distinct set of nodes. Also note that the SmartSim database can be run on a single node or sharded across 3 or more nodes (sharding across 2 nodes is not allowed).
