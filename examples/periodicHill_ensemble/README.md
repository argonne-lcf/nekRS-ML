# Ensemble of periodic hill simulations with varying hill height

This example demonstrates how to launch an ensemble of nekRS jobs sweeping through a geometric configuration.
It is based off of the [periodic hill flow](../periodicHill/README.md), the standard ERCOFTAC test case in which a Cartesian box mesh is deformed at startup into the well-known periodic hill channel.
Because the mesh deformation is performed at runtime inside `usrdat2()` from a single base `.re2` file, every ensemble member can share one nekRS binary and one `.cache` directory, and the hill geometry is modified purely by a runtime parameter in the `.par` file.

The standard periodic hill example was modified to read the parameter `hillScale` under the `[CASEDATA]` section to control the hill height at runtime. From the `.par` file,  `UDF_Setup0()` in the `.udf` file uses `platform->par->extract("casedata", "hillscale", ...)` to read the value and write it through `*nek::ptr<double>("hillScale")` into a Fortran common-block scalar registered by `usrdat0()` in the `.usr` file via `nekrs_registerPtr`. The dafault value of `hillScale = 1.0` recovers the baseline periodic hill, whereas values greater than 1 increase the hill height and values smaller than 1 decrease it.

The ensemble of nekRS cases is run with [EnsembleLauncher](https://github.com/argonne-lcf/ensemble_launcher) (EL), which is a light-weight, scalable task launcher developed at the ALCF. The example first calls `gen_ensemble_inputs.py` to parse user and case specific parameters and then generate the run directory and case files for each of the ensemble members as well as the `.json` configuration files for EL. The run directories, case files, and EL configurations are created using EL utilities available with nekRS-ML located under [3rd_party/ensembleLauncher](../../3rd_party/ensembleLauncher/) and are located under the `./run_dir` directory. Finally, the nekRS runs are launched on the system with the EL CLI command `el start`, with each nekRS case running on a separate node using all GPUs available on that node (e.g., 12 on Aurora). The EL call returns once all the nekRS runs are finished. Once completed, to inspect the files take a look at the individual run directories, where `nekrs.out` will contain the stdout log from the nekRS runs. 

## Building nekRS

Requirements:
* Linux, Mac OS X (Microsoft WSL and Windows is not supported)
* GNU/oneAPI/NVHPC/ROCm compilers (C++17/C99 compatible)
* MPI-3.1 or later
* CMake version 3.21 or later
* EnsembleLauncher

To build nekRS and the required dependencies, first clone our GitHub repository:

```sh
https://github.com/argonne-lcf/nekRS-ML.git
```

Then, simply execute one of the build scripts contained in the repository.
The HPC systems currently supported are for this example are:
* [Aurora](https://docs.alcf.anl.gov/aurora/) (Argonne LCF)

For example, to build nekRS-ML on Aurora, execute from a login node

```sh
./BuildMeOnAurora
```

## Running the example

Scripts are provided to conveniently generate run scripts and config files for the workflow on the different ALCF systems.
Note that a virtual environment with EnsembleLauncher is needed to launch the ensemble, and by default the `gen_run_script` will create one with the required dependencies.

**From a login node** execute:
```sh
./gen_run_script <system_name> </path/to/nekRS>
```

For more information on how to use `gen_run_script`, use `--help`

```sh
./gen_run_script <system_name> </path/to/nekRS> --help
```

The script will produce a `run.sh` script specifically tailored to the desired system and using the desired nekRS install directory. By default, the script is set up to run on 4 nodes, launching one nekRS simulation on each node and each with a different height of the periodic hill. To change the number of nodes to run on (and the ensemble size), simply add the number of nodes to the script as follows

```sh
./gen_run_script <system_name> </path/to/nekRS> --nodes 8
```

Finally, to run the example simply submit the run script with

```bash
qsub run.sh
```

The `run.sh` script is composed of three steps:

- **Build cache:** nekRS is run with the `--build-only` flag to populate `./.cache`. This cache (and the base `.re2`) is then *symlinked* into every ensemble member's run directory, since the periodic hill geometry is generated at runtime in `usrdat2()`.
- **Stage run directories:** `python gen_ensemble_inputs.py` builds one subdirectory per member under `./run_dir/`, copies the small case files (`periodicHill.udf`, `periodicHill.usr`), symlinks `periodicHill.re2` and `.cache`, writes a per-member `periodicHill.par` with a different `[CASEDATA] hillScale = ...`, and emits the three JSON files the [EnsembleLauncher](https://github.com/argonne-lcf/ensemble_launcher) CLI consumes: `./run_dir/config.json` (the ensembles block), `./run_dir/system_config.json` (`SystemConfig`), and `./run_dir/launcher_config.json` (`LauncherConfig`). The sweep is configurable from the command line — see `python gen_ensemble_inputs.py --help` (e.g. `--hillScale 0.8,1.2,4` for a 4-point linspace, or `--hillScale 0.9,1.0,1.1,1.2` for an explicit list).
- **Launch the ensemble:** With EnsembleLauncher's CLI command `el start ...`, the neKRS runs are launched, one on each node using all GPUs on each of the nodes. The CLI blocks until all members finish.


