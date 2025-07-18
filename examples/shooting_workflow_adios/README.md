# Solution shooting workflow with a GNN surrogate model using ADIOS2

This example combines online training of a GNN surrogate model with inference of the trained model to shoot the solution forward and thus accelerate a long simulation compaign. 
The workflow leverages the following assumption -- for a sufficiently accurate, robust and fast surrogate model, trained to advance the solution with a time step size larger than that required by the CFL constraints of nekRS, the simulation campaign can be acelerated by replacing nekRS with the surrogate and shoot the solution forward in time, where it can then be picked back up by nekRS for more model fine-tuning. 
The flow problem is based on the turbulence channel flow LES, for which the details are in the [turbChannel example](../turbChannel/README.md).

The workflow is composed of the following two stages:

* Online fine-tuning of the GNN surrogate. This step runs concurrently a high-fidelity nekRS simulation and GNN distributed training, streaming training data from the simulation to the trainer at a constant interval.
* Shooting the solution forward. This step deploys the GNN surrogate for inference, feeding the GNN predictions bask as inputs for the next step in order to advance the solution state in time.

The workflow is set up using ADIOS2 to share data between nekRS, the GNN training module, and the GNN inference module. 
Specifically, the following information is shared between components:

* The data structures needed to build the GNN grpah from the nekRS mesh. This information is shared through the file system because it is needed by the inferencing step as well, thus it needs to be persistent beyond the nekRS run. These data structures are computed once before the nekRS time step loop and written in `graph.bp` with ADIOS2.
* The pair of solution snapshots at every mesh node which represent the input and output data to the GNN model. These are streamed (data is shared through the high-speed network, not through the file system) from nekRS to the GNN trainer with the ADIOS2 SST engine. nekRS is configured to share these snapshot at a predetermined frequency set in the `turbChannel.udf` file.
* A small file called `check-run.bp` used to tell nekRS to exit cleanly when the GNN trainer reaches a stopping point (e.g., a preset maximum  number of iterations or a tolerance on the training loss).
* A solution checkpoint for the last nekRS time step saved as the simulation exit cleanly. This is stored in `checkpoint.bp` and is loaded by the inference module as an initial condition to then advance the solution state with the GNN.

The workflow makes use of new plugins added to the nekRS code. 
The plugin API are called from the `turbChannel.udf` file, specifically within `UDF_Setup()` for initialization and `UDF_ExecuteStep()` to execute tasks every simulation time step.

* `adiosStreamer.hpp`: plugin that creates client with a few ADIOS2 IO objects to enable streaming and I/O of data from other plugins. This plugin also checks when the trainer signals nekRS to quit running and is used to write a checkpoint file to disk at the end of the run. 
* `gnn.hpp`: plugin that computes the graph data structures and communicates them to the trainer (or simply writes them to disk).
* `trajGen.hpp`: plugin that generates a trajectory of training data for the GNN surrogate. The trajectory consists of two solution snapshots from two successive time steps.


## Building nekRS

Requirements:
* Linux, Mac OS X (Microsoft WSL and Windows is not supported) 
* GNU/oneAPI/NVHPC/ROCm compilers (C++17/C99 compatible)
* MPI-3.1 or later
* CMake version 3.21 or later 
* PyTorch and PyTorch Geometric (for the examples using the GNN)

To build nekRS and the required dependencoes, first clone our GitHub repository:

```sh
https://github.com/argonne-lcf/nekRS-ML.git
```

Then, simply execute one of the build scripts contained in the reposotory. 
The HPC systems currently supported are:
* [Polaris](https://docs.alcf.anl.gov/polaris/) (Argonne LCF)
* [Aurora](https://docs.alcf.anl.gov/aurora/) (Argonne LCF) 
* [Crux](https://docs.alcf.anl.gov/crux/) (Argonne LCF)

For example, to build nekRS-ML on Aurora, execute from a compute node

```sh
./BuildMeOnAurora
```

## Runnig the example

Scripts are provided to conveniently generate run scripts and config files for the workflow on the different ALCF systems.
Note that a virtual environment with PyTorch Geometric is needed to train the GNN online.

**From a compute node** execute

```sh
./gen_run_script <system_name> </path/to/nekRS>
```

or

```sh
./gen_run_script <system_name> </path/to/nekRS> </path/to/venv/bin/activate>
```

if you have the necessary packages already installed in a Python virtual environment. 

The script will produce a `run.sh` script specifically tailored to the desired system and using the desired nekRS install directory. 

Finally, simply execute the run script **from the compute nodes** with

```bash
./run.sh
```

The `run.sh` script is composed of two steps:

- First nekRS is run by itself with the `--build-only` flag. This is done such that the `.cache` directory can be built beforehand instead of during online training. This step can be run only once and is helpful to not hang the progress of the simulation and GNN training while the cache is built.
- Execution of the workflow driver script `driver.py` with Python, which takes in the setting in the `config.yaml` file and launches fine tuning (nekRS + GNN training) followed by GNN inference on the requested resources. 

The outputs logs of the nekRS, trainer and inference will be within the `./logs` directory created at runtime.
