# Ensemble of periodic hill simulations with varying hill height

This example demonstrates how to launch an ensemble of nekRS jobs in which each member runs the same case at a different geometric configuration.
It is based off of the [periodic hill flow](../periodicHill/README.md), the standard ERCOFTAC test case in which a Cartesian box mesh is procedurally deformed at startup into the well-known periodic hill channel.
Because the mesh deformation is performed at runtime inside `usrdat2()` from a single base `.re2` file, every ensemble member can share one nekRS binary and one `.cache` directory, and per-member geometry is selected purely by a runtime parameter.

The ensemble parameter is a dimensionless `hillScale` multiplier on the hill peak height, exposed through a `[CASEDATA]` block in `periodicHill.par`.
The data flow is `.par` → `.udf` → `.usr`: in `UDF_Setup0()`, `platform->par->extract("casedata", "hillscale", ...)` reads the value from the `.par` file and writes it through `*nek::ptr<double>("hillScale")` into a Fortran common-block scalar registered by `usrdat0()` via `nekrs_registerPtr`.
That value is then consumed in `usrdat2()` as `hmax = 28.0 * hillScale` when the base box mesh is deformed by the `bottomph()` polynomial, scaling the hill peak height (the baseline `hmax = 28.0` of the ERCOFTAC geometry) without changing the underlying hill shape.
A `hillScale` of `1.0` reproduces the original periodic hill case exactly, and both `UDF_Setup0()` and `usrdat2()` log the resolved `hillScale`/`hmax` from rank 0 so each member's logfile records the configuration it actually ran.

In practice, an ensemble driver only needs to (i) build nekRS once with `--build-only`, (ii) for each member, write a templated copy of `periodicHill.par` with a different `[CASEDATA] hillScale = ...` value into a per-member subdirectory and symlink the rest of the case files, and (iii) launch the members concurrently on disjoint subsets of the allocation (the multi-`mpiexec`/`PalsMpiexecSettings` pattern in `examples/shooting_workflow_smartredis/driver.py` is a natural starting point).
Sensible sweep ranges are roughly `hillScale ∈ [0.7, 1.3]` — larger excursions push the deformed elements past `ymax = 3.035` and degrade element quality, so it is worth inspecting the `.fld` output of one member before launching a full sweep.

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

