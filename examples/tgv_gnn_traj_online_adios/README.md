# Online training of a time dependent GNN surrogate model

This example demonstrates how the `gnn`, `trajGen`, and `smartRedis` plugins can be used to create a distributed graph from the nekRS mesh and online train a GNN from a solution trajectory.
The online workflow is set up using ADIOS2.
The flow problem is based off of the [Taylor-Green-Vortex flow](../tgv/README.md), however on a slightly smaller mesh. 
In this example, the model takes as inputs the three components of velocity at a given time step and learns to predict the velocity field at a future step, thus advancing the solution forward and replacing the solver.
It is a time dependent modeling task, the model learns how to predict a future time step.

Specifically, in `UDF_Setup()`, the `graph` class is instantiated from the mesh, followed by calls to `graph->gnnSetup();` and `graph->gnnWriteADIOS();` to setup and write the GNN input files, respectively. Here, the ADIOS2 client and the trajectory generation class are also initialized.
In `UDF_ExecuteStep()`, the `trajGenWriteADIOS()` method of the trajectory generation class is used to stream the training data to the GNN trainer. By default, the SST engine with RDMA transport and a synchronous data transfer patter is used, but these configurations can be changed in the `.par` file.
For simplicity and reproducibility, nekRS is set up to send training data every 10 time steps for 5 consicutive times only (up to time step 50), but `UDF_ExecuteStep()` can be changed to send as many time steps as desired.

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

- First nekRS is run by itself with the `--build-only` flag. This is done such that the `.cache` directory can be built beforehand instead of during online training.
- The online training workflow driver `driver.py` is executed with Python, setting up nekRS and the GNN training.

The outputs of the nekRS and trainer will be within the `./logs` directory created at runtime.
