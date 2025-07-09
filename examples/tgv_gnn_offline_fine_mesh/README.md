# Offline training of a time independent GNN surrogate model on a coarser mesh

This example demonstrates how the `gnn` plugin can be used to create a distributed graph from a coarsened nekRS mesh and train a GNN from a series of saved solution fields.
It is based off of the [Taylor-Green-Vortex flow](../tgv/README.md), however on a slightly smaller mesh. 
In this example, the model takes as inputs the three components of velocity and learns to predict the pressure field at every graph (mesh) node.
It is a time independent modeling task, since no information regarding the time dependency of the solution stepshots is given to the GNN.
The key difference between this example and the [tgv_gnn_offline](../tgv_gnn_offline/) example is that in this case the GNN is trained on a coarser mesh than what nekRS is using to integrate the solution. In the .par file, the nekRS simulation is run with a higher-order polynomial degree (see the `polynomialOrder` parameter) than the GNN one (see the `gnnPolynomialOrder` parameter).

Specifically, in `UDF_Setup()`, the `graph` class is instantiated from the mesh, followed by calls to `graph->gnnSetup();` and `graph->gnnWrite();` to setup and write the GNN input files to disk, respectively. 
The files are written in a directory called `./gnn_outputs_poly_3`, where the `3` marks the fact that 3rd order polynomials are used for the GNN mesh.
In `UDF_ExecuteStep()`, first the pressure and velocity fields are interpolated to the coarser GNN mesh with `graph->interpolateField()` and then the `writeToFileBinaryF()` routine is called to write the velocity and pressure fields to disk. 
These files are tagged with the time stamp, rank ID, and job size, and are also located in `./gnn_outputs_poly_3`.
For simplicity and reproducibility, nekRS is set up to run for a single time step, thus only printing the velocity and pressure for the initial condition, but `UDF_ExecuteStep()` can be changed to print as many time steps as desired.

## Building nekRS

To build nekRS with the GNN plugin, simply execute the build script [BuildMeOnAurora](../../BuildMeOnAurora) from the top directory
```bash
source BuildMeOnAurora
```

NOTE: you can disable building SmartRedis for this example since it is performing offline training from files saved to disk.

## Runnig the example

Scripts are provided to conveniently generate run scripts and config files for the workflow on the different ALCF systems.
Note that a virtual environment with PyTorch Geometric is needed for the GNN.
If you don't specify a virtual environment path, the script will create one for you.
From an interactive session on the compute nodes, first execute
```bash
./gen_run_script
```

taking notice of some of the variables to set. 
Specifically, make sure to set 

```
SYSTEM # the ALCF system to run on (aurora, polaris)
NEKRS_HOME # path to the nekRS install directory
VENV_PATH # path to the Python venv activate script
PROJ_ID # project name for the allocation
QUEUE # name of the queue to run on
```

The script generates the run script, which is executed with
```bash
./run.sh
```

The `run.sh` script is composed of four steps:

- The nekRS simulation to generate the GNN input files. This step produces the graph and training data in `./gnn_outputs_poly_3`.
- An auxiliary Python script to create additional data structures needed to enforce consistency in the GNN. This step produces some additional files in `./gnn_outputs_poly_3` needed during GNN training.
- A Python script to check the accuracy of the data generated. This script compares the results in `./ref` with those created in `./gnn_outputs_poly_3`.
- GNN training. This step trains the GNN for 100 iterations based on the data provided in `./gnn_outputs_poly_3`.
- The case is run with 2 MPI ranks for simplicity, however the users can set the desired number of ranks. Note to comment out the accuracy checks as they will fail in this case. 

