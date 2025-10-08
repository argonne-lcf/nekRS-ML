# Offline training of the SR-GNN model for mesh-based, three-dimensional super-resolution

This example demonstrates the pipeline for training and deploying the SR-GNN model on nekRS field data. It builds upon the [turbulent channel example](../turbChannel/) available with nekRS by modifying the `.par` and `.udf` files and calling on the scripts available in the [SR-GNN repository](../../3rd_party/sr-gnn/).
The following important steps and modifications are highlighted for this example:

* The `.par` file is modified to include the `[ML]` section, under which the polynomial order of the interpolated fields is specified with the `gnnPolynomialOrder` parameter. For this example, it is set to 1 in contrast with the target polynimial order used for the nekRS simulation, which is set to 7.
* The `.udf` file is modified to write both the GNN data structures and the training data at both polynomial orders. In `UDF_Setup()`, we utilize the `gnn` plugin to write the graph data creating the `./gnn_outputs_poly_1` and `./gnn_outputs_poly_7` directories. In `UDF_ExecuteStep()`, we utilize a custom wrapper fuction called `outfld_wrapper()` to write a `p=1` field each time a `p=7` field is written. The `p=1` fields contain the input data to the SR-GNN model and the `p=7` fields contain the target data for training.
* After running neKRS and generating the graph and training data, the training data is generated calling the `nek_to_pt.py` script. This script takes some critical parameters as inputs, including a list of target field and input files to be converted to PyTorch Goemetric Data format. The `--n_element_neighbors` input flat determines how many neighboring elements are used in the super-resolution task, and is set to the default value of 6.
* Training of the SR-GNN model is performed in parallel with PyTorch DDP calling the `main.py` script located in the    `sr-gnn` directory. For the example, training is performed for a few epochs only. Note the backend set to `xccl` for Aurora and the `n_element_neighbors` matching the value used for the preprocessing step. 
* Finally, inference and postprocessing is performed with the `postprocess.py` script, which loads the saved model to produce predicted (super-resolved) and error fields. The script takes as inputs the path to the saved model, the name of the output nek fields, the list of nek fields to load for inference and for measuring the error in the predictions, and the number of neighbors. 


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
./setup_offline.sh <system_name> </path/to/nekRS> </path/to/venv>
```
if you have the necessary packages already installed in a Python virtual environment.

The script will produce a `run.sh` script specifically tailored to the desired system and using the desired nekRS install directory.

Finally, simply execute the run script **from the compute nodes** with

```bash
./run.sh
```

The `run.sh` script is composed of four steps:

- A precompilation step in which nekRS is run with the --build-only flag. This is done such that the `.cache` directory can be built beforehand.
- The nekRS simulation to generate the SR-GNN input files. This step produces the graph data at the higher and lower polynomial orders in `./gnn_outputs_poly_7` and `./gnn_outputs_poly_1`. It also produces the field data to be used for training at the two polynomial orders, which will be in `turbChannel0.f*` and `turbChannel_p10.f*` files.
- A preprocessing step to convert the nekRS field data into PyTorch Geometric Data to be used for training. This step produces the `pt_datasets` directory containing the training and validation datasets.
- SR-GNN training. This step trains the SR-GNN for a few epochs based on the data produced in the previus step. The output of this step is a model checkpoint stored inside `./saved_models`.
- Finally, the SR-GNN is used to perform inference (i.e., postprocessing) and produce a super-resolved solution field and corresponding error field. This step uses the trained model checkpoint and evaluates on the low-order `turbChannel_p10.f*` files.