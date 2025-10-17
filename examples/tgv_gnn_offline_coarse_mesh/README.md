# Offline training of a time independent GNN surrogate model on a coarser mesh

This example demonstrates how the `gnn` plugin can be used to create a distributed graph from a coarsened nekRS mesh and train a GNN from a series of saved solution fields.
It is based off of the [Taylor-Green-Vortex flow](../tgv/README.md), however on a slightly smaller mesh.
In this example, the model takes as inputs the three components of velocity and learns to predict the pressure field at every graph (mesh) node.
It is a time independent modeling task, since no information regarding the time dependency of the solution snapshots is given to the GNN.
The key difference between this example and the [tgv_gnn_offline](../tgv_gnn_offline/) example is that in this case the GNN is trained on a coarser mesh than what nekRS is using to integrate the solution. In the .par file, the nekRS simulation is run with a higher-order polynomial degree (see the `polynomialOrder` parameter) than the GNN one (see the `gnnPolynomialOrder` parameter).

Specifically, in `UDF_Setup()`, the `graph` class is instantiated from the mesh, followed by calls to `graph->gnnSetup();` and `graph->gnnWrite();` to setup and write the GNN input files to disk, respectively.
The files are written in a directory called `./gnn_outputs_poly_3`, where the `3` marks the fact that 3rd order polynomials are used for the GNN mesh.
In `UDF_ExecuteStep()`, first the pressure and velocity fields are interpolated to the coarser GNN mesh with `graph->interpolateField()` and then the `writeToFileBinaryF()` routine is called to write the velocity and pressure fields to disk.
These files are tagged with the time stamp, rank ID, and job size, and are also located in `./gnn_outputs_poly_3`.
For simplicity and reproducibility, nekRS is set up to run for a single time step, thus only printing the velocity and pressure for the initial condition, but `UDF_ExecuteStep()` can be changed to print as many time steps as desired.

## Building nekRS

Requirements:
* Linux, Mac OS X (Microsoft WSL and Windows is not supported)
* GNU/oneAPI/NVHPC/ROCm compilers (C++17/C99 compatible)
* MPI-3.1 or later
* CMake version 3.21 or later
* PyTorch and PyTorch Geometric (for the examples using the GNN)

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
./BuildMeOnAurora
```

## Running the example

Scripts are provided to conveniently generate run scripts and config files for the workflow on the different ALCF systems.
Note that a virtual environment with PyTorch Geometric is needed to train the GNN.

**From a compute node** execute:
```sh
./gen_run_script <system_name> </path/to/nekRS>
```
or
```sh
./gen_run_script <system_name> </path/to/nekRS> --venv_path </path/to/venv>
```
if you have the necessary packages already installed in a Python virtual environment. For more information
on how to use `gen_run_script`, use `--help`

```sh
./gen_run_script --help

The script will produce a `run.sh` script specifically tailored to the desired system and using the desired nekRS install directory.

Finally, simply execute the run script **from the compute nodes** with

```bash
./run.sh
```

The `run.sh` script is composed of four steps:

- The nekRS simulation to generate the GNN input files. This step produces the graph and training data in `./gnn_outputs_poly_3`.
- An auxiliary Python script to create additional data structures needed to enforce consistency in the GNN. This step produces some additional files in `./gnn_outputs_poly_3` needed during GNN training.
- A Python script to check the accuracy of the data generated. This script compares the results in `./ref` with those created in `./gnn_outputs_poly_3`.
- GNN training. This step trains the GNN for 100 iterations based on the data provided in `./gnn_outputs_poly_3`.
- The case is run with 2 MPI ranks for simplicity, however the users can set the desired number of ranks. Note to comment out the accuracy checks as they will fail in this case.

