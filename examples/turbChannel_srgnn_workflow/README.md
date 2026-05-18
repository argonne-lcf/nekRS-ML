# SR-GNN Model Training Workflow with EnsembleLauncher

This example extends the [turbChannel_srgnn](../turbChannel_srgnn/) example to train the SR-GNN model using low-polynomial order nekRS data instead of data projected from high to low p-order. After an initial nekRS run saving both high and projected low p-order checkpoints, EnsembleLauncher (EL) is used to automatically run short nekRS simulations from each of the checkpoints and produce more realistic low-p snapshots to be used as the training inputs.
The example still uses the [turbulent channel](../turbChannel/) as the flow case.
The SR-GNN model is based on the [work](https://www.sciencedirect.com/science/article/abs/pii/S0045782525003445) of Prof. Shivam Barwey (U. Notre Dame).

This example combines many of the tools demonstrated in the [turbChannel_srgnn](../turbChannel_srgnn/) and [periodicHill_ensemble](../periodicHill_ensemble/) examples, with the following additional details:

* The example uses two `.par` files. `turbChannel.par` is very similar to the one used by the regular [turbChannel_srgnn](../turbChannel_srgnn/) example and is used to run the initial nekRS simulation writing the `gnn_outputs_poly_*` directories and the `.f` files for the high p-order used by nekRS and the lower p-order specified with the `gnnPolynomialOrder` parameter (in this case set to 2). The second file called `turbChannel_simple.par` is used by the second set of nekRS simulations launched with EL which start from the aforementioned `.f` files and simply advance the simulation for a small number of time steps writing additional `.f` files to be used for training.
* The example also uses two `.udf` files. Similarly to above, `turbChannel.udf` is used for the initial simulation and `turbChannel_simple.udf` is used for the additional simulations launched with EL.
* The example contains a `gen_ensemble_inputs.py` script used to prepare the ensemble of nekRS simulations by creating run directorties with the appropriate files and the JSON configuration for EL all stored in `./run_dir`. The main arguments to the script are the case name, the values of the polynimial orders used to run the additional nekRS simulations, the number of time steps to run. For each additional nekRS simulation, `turbChannel_simple.par` will be copied into the run directory and renamed to `turbChannel.par` with the correct polynomial order and number of time steps. Additionally, the appropriate `.f` checkpoint file from the initial simulation will be linked and renamed to `restart.fld` so the new simulation can restart from the correct snapshot and p-order.
* The training data from the model is collected from the checkpoint files produced by the ensemble of nekRS simulations, with the low p-order simulations contributing the coarse inputs and the high p-order simulations contributing the target outputs. 


## Building nekRS

Requirements:
* Linux, Mac OS X (Microsoft WSL and Windows is not supported)
* GNU/oneAPI/NVHPC/ROCm compilers (C++17/C99 compatible)
* MPI-3.1 or later
* CMake version 3.21 or later
* PyTorch, PyTorch Geometric and PyTorch Cluster
* Pymech (for reaking nekRS files from Python)
* EnsembleLauncher for orchestrating the ensemble of nekRS runs

To build nekRS and the required dependencies, first clone our GitHub repository:

```sh
https://github.com/argonne-lcf/nekRS-ML.git
```

Then, simply execute one of the build scripts contained in the repository.
The HPC systems currently supported for this example are:
* [Aurora](https://docs.alcf.anl.gov/aurora/) (Argonne LCF)

For example, to build nekRS-ML on Aurora, from the login nodes execute 

```sh
./BuildMeOnAurora
```

## Running the example

Scripts are provided to conveniently generate run scripts and config files for the workflow on the different ALCF systems.
Note that a virtual environment with EnsembleLauncher, PyTorch Geometric and PyTorch Cluster is needed to launch the ensemble and training/inference, and by default the `gen_run_script` will create one with the required dependencies.

**From a login node** execute:
```sh
./gen_run_script <system_name> </path/to/nekRS>
```

For more information on how to use `gen_run_script`, use `--help`

```sh
./gen_run_script <system_name> </path/to/nekRS> --help
```

The script will produce a `run.sh` script specifically tailored to the desired system and using the desired nekRS install directory. By default, the script is set up to run on 4 nodes. To change the number of nodes to run on, simply add the number of nodes to the script as follows

```sh
./gen_run_script <system_name> </path/to/nekRS> --nodes 8
```

Finally, to run the example simply submit the run script with

```bash
qsub run.sh
```

The `run.sh` script is composed of six steps:

1. A precompilation step in which nekRS is run with the `--build-only` flag. This is done such that the `.cache` directory can be built beforehand.
2. The initial nekRS simulation to generate the checkpoint `.f` files at the low and high p-orders and the `gnn_outputs_poly_*` directories. The example sets the higher p-order to 7 and the lower one to 2, however these values can be changed in the `run.sh` script.
3. The ensemble of additional nek runs. First, `gen_ensemble_inputs.py` is executed to prepare the EL configuration and run directories, then EL is launched with the CLI command `el start`. 
    * NOTE: The turbulent channel case can easily be run on a single node of Aurora using all PVC 6 GPUs (12 tiles). Therefore, the initial simulation and all other simulations launched by nekRS are set up to run on a single node. With 4 nodes available, this means initially in step 2, 3 nodes are not being utilized, but then all 4 nodes are used to run 4 parallel nekRS simulations during this step. Training also uses all 4 nodes. If more nodes are requested for this example, only the ensemble and training will benefit from these additional resources. Moreover, if this example is to be run on a different case requiring more nodes, simple change the value assigned to `NODES_PER_NEKRS` in the `run.sh` script and allocate sufficient number of nodes to run the workflow.
4. Training data is collected from the ensemble of runs and the files are passed to the SR-GNN utility `nek_to_pt.py` to create the input data to the model in PyTorch format.
5. Training of the SR-GNN model is performed in parallel using all available nodes and GPU.
6. Inference is performed at the end to create `.f` files to be used to visualize the reconstruction of the SR-GNN model. 
