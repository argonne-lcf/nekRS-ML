# Offline training of Diffusion Graph Network (DGN) model

This example demonstrates the pipeline for training and deploying the diffusion graph network (DGN) model on nekRS field data. 
It is a simplified version of the flow around a cylinder ...


## Building nekRS

Requirements:
* Linux, Mac OS X (Microsoft WSL and Windows is not supported)
* GNU/oneAPI/NVHPC/ROCm compilers (C++17/C99 compatible)
* MPI-3.1 or later
* CMake version 3.21 or later
* PyTorch, PyTorch Geometric and PyTorch Cluster
* Pymech (for reaking nekRS files from Python)

To build nekRS and the required dependencies, first clone our GitHub repository:

```sh
https://github.com/argonne-lcf/nekRS-ML.git
```

Then, simply execute one of the build scripts contained in the repository.
The HPC systems currently supported are for this example are:
* [Polaris](https://docs.alcf.anl.gov/polaris/) (Argonne LCF)
* [Aurora](https://docs.alcf.anl.gov/aurora/) (Argonne LCF)

For example, to build nekRS-ML on Aurora, execute from a compute node

```sh
./BuildMeOnAurora
```

## Running the example

Scripts are provided to conveniently generate run scripts and config files for the workflow on the different ALCF systems.
Note that a virtual environment with PyTorch Geometric and other dependencies is needed to train the SR-GNN.

**From a compute node** execute

```sh
./gen_run_script <system_name> </path/to/nekRS>
```

or

```sh
./gen_run_script <system_name> </path/to/nekRS> -v </path/to/venv>
```
if you have the necessary packages already installed in a Python virtual environment.

The script will produce a `run.sh` script specifically tailored to the desired system and using the desired nekRS install directory.

Finally, simply execute the run script **from the compute nodes** with

```bash
./run.sh
```

The `run.sh` script is composed of X steps:

- ...


