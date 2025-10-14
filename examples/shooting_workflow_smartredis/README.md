# Solution shooting workflow with a GNN surrogate model using SmartSim

This example builds off of the [online training of time dependent GNN surrogate](../tgv_gnn_traj_online/README.md), however it adds the step of performing inference with the GNN surrogate after online training has concluded. 
The flow problem is based on the turbulence channel flow LES, for which the details are in the [turbChannel example](../turbChannel/README.md).

The main differences between this example and simple online training of time dependent GNN surrogate are in the `driver.py` workflow driver script. 
Specifically, the workflow runner alternates between fine-tuning of the GNN and deploying the model for inference.
During fine-tuning, both nekRS and GNN training are running concurrently.
During inference, only the GNN is run advancing the velocity field in time.
As usual, the plugins are called from  `UDF_Setup()` and `UDF_ExecuteStep()`. 
Note that at the end of the run, nekRS writes a checkpoint to the SmartSim database, which is used as initial condition to GNN inference.

## Building nekRS

Requirements:
* Linux, Mac OS X (Microsoft WSL and Windows is not supported) 
* GNU/oneAPI/NVHPC/ROCm compilers (C++17/C99 compatible)
* MPI-3.1 or later
* CMake version 3.21 or later 
* PyTorch and PyTorch Geometric (for the examples using the GNN)
* SmartSim and SmartRedis (for the examples using SmartSim as a workflow driver)

To build nekRS and the required dependencies, first clone our GitHub repository:

```sh
https://github.com/argonne-lcf/nekRS-ML.git
```

Then, simply execute one of the build scripts contained in the repository. 
The HPC systems currently supported are:
* [Polaris](https://docs.alcf.anl.gov/polaris/) (Argonne LCF)
* [Aurora](https://docs.alcf.anl.gov/aurora/) (Argonne LCF) 
* [Crux](https://docs.alcf.anl.gov/crux/) (Argonne LCF)

For example, to build nekRS-ML on Aurora, execute from a compute node

```sh
ENABLE_SMARTREDIS=ON ./BuildMeOnAurora
```

## Running the example

Scripts are provided to conveniently generate run scripts and config files for the workflow on the different ALCF systems.
Note that a virtual environment with PyTorch Geometric and SmartSim/SmartRedis is needed to train the GNN online.

**From a compute node** execute:
```sh
./gen_run_script <system_name> </path/to/nekRS>
```
or
```sh
./gen_run_script <system_name> </path/to/nekRS> --venv_path </path/to/venv/bin/activate>
```
if you have the necessary packages already installed in a Python virtual environment.

You can specify SmartSim database nodes, simulation nodes and train nodes using `-dn`,
`-sn` and `-tn` respectively.
```sh
./gen_run_script <system_name> </path/to/nekRS> -dn 3 -sn 1 -tn 3
```

For more information on how to use `gen_run_script`, use `--help`:
```sh
./gen_run_script --help
```

The script will produce a `run.sh` script specifically tailored to the desired system and using the desired nekRS install directory. 

Finally, simply execute the run script **from the compute nodes** with

```bash
./run.sh
```

The `run.sh` script is composed of two steps:

- First nekRS is run by itself with the `--build-only` flag. This is done such that the `.cache` directory can be built beforehand instead of during online training.
- The online training workflow driver `driver.py` is executed with Python, setting up the SmartSim Orchestrator (the database), followed by fine-tuning involving nekRS and the GNN trainer, followed by inference once fine-tuning is over.

The outputs of the nekRS, trainer and inference will be within the `./nekRS` directory created at runtime.

## Known Issues and Tips
- The clustered deployment requires at least 3 nodes for training and 2 nodes for inference since it deploys each component on a distinct set of nodes. Also note that the SmartSim database can be run on a single node or sharded across 3 or more nodes (sharding across 2 nodes is not allowed).
