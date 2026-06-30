# Offline training of the Dist-GT model

This example is identical to the [tgv_gnn_offline_traj](../tgv_gnn_offline_traj/) one, with the exception that it demonstrates how to train the Dist-GT model instead of the Dist-GNN model. Since the Dist-GT and Dist-GNN models have many similarities and differ mainly in the internal layers of the arcitecture, the same `main.py` and `trainer.py` scripts can be used for both. The GT architecture, present in [graph_transformer.py](../../3rd_party/gnn/dist-gnn/graph_transformer.py), is used by the Dist-GNN trainer when the `model_name=graph_transformer` argument is passed. This is the only change required to use the Dist-GT model. There are two unique hyperparameters to the GT model, `n_transformer_layers` and `num_heads`, which control the number of transformer layers (similar to the number of message passing layers) and the number of attention heads, respectively.


## Building nekRS

Requirements:
* Linux, Mac OS X (Microsoft WSL and Windows is not supported)
* GNU/oneAPI/NVHPC/ROCm compilers (C++17/C99 compatible)
* MPI-3.1 or later
* CMake version 3.21 or later
* PyTorch and PyTorch Geometric 

To build nekRS and the required dependencies, first clone our GitHub repository:

```sh
https://github.com/argonne-lcf/nekRS-ML.git
```

Then, simply execute one of the build scripts contained in the repository.
The HPC systems currently supported are:
* [Polaris](https://docs.alcf.anl.gov/polaris/) (Argonne LCF)
* [Aurora](https://docs.alcf.anl.gov/aurora/) (Argonne LCF)

For example, to build nekRS-ML on Aurora, execute from a compute node

```sh
./BuildMeOnAurora
```

## Running the example

Scripts are provided to conveniently generate run scripts and config files for the workflow on the different ALCF systems.
Note that a virtual environment with PyTorch Geometric is needed to train the GT on Aurora.

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
```

The script will produce a `run.sh` script specifically tailored to the desired system and using the desired nekRS install directory.

Finally, simply execute the run script **from the compute node** with

```bash
./run.sh
```

The `run.sh` script is composed of five steps:

- The nekRS simulation to generate the GT input files and the trajectory. This step produces the graph and training data in `./gnn_outputs_poly_7` and `./traj_poly_7`, respectively.
- An auxiliary Python script to create additional data structures needed to enforce consistency in the GT. This step produces some additional files in `./gnn_outputs_poly_7` needed during GT training.
- A Python script to check the accuracy of the graph data generated. This script compares the results in `./ref` with those created in `./gnn_outputs_poly_7`.
- A second check with the same Python script to ensure the accuracy of the trajectory data generated. This script compares the results in `./ref` with those created in `./traj_poly_7`.
- GT training. This step trains the GT for 100 iterations based on the data provided in `./gnn_outputs_poly_7` and `./traj_poly_7`.
- The case is run with 4 MPI ranks for simplicity, however the users can set the desired number of ranks. Note to comment out the accuracy checks as they will fail in this case.

