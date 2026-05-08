# Ensemble of periodic hill simulations with varying hill height

This example demonstrates how to launch an ensemble of nekRS jobs in which each member runs the same case at a different geometric configuration.
It is based off of the [periodic hill flow](../periodicHill/README.md), the standard ERCOFTAC test case in which a Cartesian box mesh is procedurally deformed at startup into the well-known periodic hill channel.
Because the mesh deformation is performed at runtime inside `usrdat2()` from a single base `.re2` file, every ensemble member can share one nekRS binary and one `.cache` directory, and per-member geometry is selected purely by a runtime parameter.

The ensemble parameter `hillScale` in `[CASEDATA]` controls hill height: **larger values yield a taller hill** in visualization—more intuitive than multiplying `hmax` directly in `bottomph(...)`, which stretches the polynomial coordinates and can make factors greater than one look shorter on screen.
The data flow is `.par` → `.udf` → `.usr`: in `UDF_Setup0()`, `platform->par->extract("casedata", "hillscale", ...)` reads the value from the `.par` file and writes it through `*nek::ptr<double>("hillScale")` into a Fortran common-block scalar registered by `usrdat0()` via `nekrs_registerPtr`.
nek invokes `usrdat0()` twice (bootstrap and again at the start of nek setup); `UDF_Setup0` runs between those calls. The `.usr` file therefore registers the pointer only once and avoids resetting `hillScale` on the second `usrdat0()`—otherwise the value written by `UDF_Setup0` would be overwritten with `1.0` before `usrdat2()` runs.
In `usrdat2()`, that value is mapped onto the ERCOFTAC hill polynomial via `hmax = 28.0 / hillScale` before `bottomph(...)` deforms the mesh (`28.0` is the canonical peak scale in the reference geometry). Use **positive** `hillScale` only. With this convention, `hillScale = 1.0` recovers the baseline periodic hill; both `UDF_Setup0()` and `usrdat2()` log `hillScale` and `hmax` from rank 0 so each member's logfile records the configuration it actually ran.

In practice, an ensemble driver only needs to (i) build nekRS once with `--build-only`, (ii) for each member, write a templated copy of `periodicHill.par` with a different `[CASEDATA] hillScale = ...` value into a per-member subdirectory and symlink the rest of the case files, and (iii) launch the members concurrently on disjoint subsets of the allocation (the multi-`mpiexec`/`PalsMpiexecSettings` pattern in `examples/shooting_workflow_smartredis/driver.py` is a natural starting point).
Sweep modestly around `hillScale = 1` (for example `0.8`–`1.5`): very small `hillScale` drives `hmax` large and can crush elements against `ymax = 3.035`, while very large `hillScale` drives `hmax` small and can distort the hill shape—inspect one member's `.fld` before launching a large ensemble.

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

The `run.sh` script is composed of two steps:

- NekRS is run with the `--build-only` flag to create the `.cache` directory. This cache will be used by all ensemble members since the mesh is deformed at runtime within the `usrdat2()` function.
- The ensemble ...

