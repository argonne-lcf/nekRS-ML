# Offline training of a time dependent GNN surrogate model

This example demonstrates how the `gnn` and the `trajGen` plugins can be used to create a distributed graph from the nekRS mesh and train a GNN surrogate from a saved solution trajectory.
It is based off of the [Taylor-Green-Vortex flow](../tgv/README.md), however on a slightly larger mesh. 
In this example, the model takes as inputs the three components of velocity at a particular time step and learns to predict the velocity components at every graph (mesh) node at a later time step. 
It is a time dependent modeling task, since the training data consists of a pair of solution fields and the model learns to predict future solutions. 

Specifically, in `UDF_Setup()`, the `graph` class is instantiated from the mesh, followed by calls to `graph->gnnSetup();` and `graph->gnnWrite();` to setup and write the GNN input files to disk, respectively. 
The GNN input files are written in a directory called `./gnn_outputs_poly_7`, where the `7` marks the fact that 7th order polynomials are used in this case.
The `trajGen` class is also instantiated, followed by a call to `tgen->trajGenSetup()`.
When initializing the trajectory class, note the definition of `dt_factor`. 
This is the delta factor used to create the trajectory for training.
If `dt_factor=1`, the GNN learns to predict 1 time step into the future, i.e., the next nekRS time step. 
If `dt_factor=10`, the GNN learns to predict the 10 time steps into the future. 
In `UDF_ExecuteStep()`, `tgen->trajGenWrite()` is called to write the trajectory files to disk.
The files are written to `./traj_poly_7`, under a sub-directory tagged with the initial time and the delta factor used for the trajectory, as well as separated into directories for each MPI rank of the nekRS simulation. 


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
Note that a virtual environment with PyTorch Geometric is needed to train the GNN.

**From a compute node** execute

```sh
../setup_offline.sh <system_name> </path/to/nekRS>
```

or

```sh
./setup_offline.sh <system_name> </path/to/nekRS> </path/to/venv> <number_of_nodes> <time> <proj_id>
```

if you have the necessary packages already installed in a Python virtual environment. 

The script will produce a `run.sh` script specifically tailored to the desired system and using the desired nekRS install directory. 

Finally, simply execute the run script **from the compute nodes** with

```bash
./run.sh
```

The `run.sh` script is composed of five steps:

- The nekRS simulation to generate the GNN input files and the trajectory. This step produces the graph and training data in `./gnn_outputs_poly_7` and `./traj_poly_7`, respectively.
- An auxiliary Python script to create additional data structures needed to enforce consistency in the GNN. This step produces some additional files in `./gnn_outputs_poly_7` needed during GNN training.
- A Python script to check the accuracy of the graph data generated. This script compares the results in `./ref` with those created in `./gnn_outputs_poly_7`.
- A second check with the same Python script to ensure the accuracy of the trajectory data generated. This script compares the results in `./ref` with those created in `./traj_poly_7`.
- GNN training. This step trains the GNN for 100 iterations based on the data provided in `./gnn_outputs_poly_7` and `./traj_poly_7`.
- The case is run with 4 MPI ranks for simplicity, however the users can set the desired number of ranks. Note to comment out the accuracy checks as they will fail in this case. 

