# Wall-modeled LES of a turbulent channel with an equilibrium wall model

This example performs wall-modeled LES of a turbulent channel flow at a friction Reynolds number of 950. 
It was designed by Vishal Kumar at the Barcelona Supercomputing Center and adapted for this version of nekRS.

The example modifies the `.par` and `.udf` files relative to the plain [turbulent channel example](../turbChannel/) as follows ...


## Building nekRS

Requirements:
* Linux, Mac OS X (Microsoft WSL and Windows is not supported) 
* GNU/oneAPI/NVHPC/ROCm compilers (C++17/C99 compatible)
* MPI-3.1 or later
* CMake version 3.21 or later 
* Python with matplotlib (for the post-processing)

To build nekRS and the required dependencoes, first clone our GitHub repository:

```sh
https://github.com/argonne-lcf/nekRS-ML.git
```

Then, simply execute one of the build scripts contained in the reposotory. 
The HPC systems currently supported are:
* [Polaris](https://docs.alcf.anl.gov/polaris/) (Argonne LCF)
* [Aurora](https://docs.alcf.anl.gov/aurora/) (Argonne LCF) 

For example, to build nekRS-ML on Aurora, execute from a compute node

```sh
./BuildMeOnAurora
```

## Runnig the example

Scripts are provided to conveniently generate run scripts and config files for the workflow on the different ALCF systems.

**From a compute node** execute

```sh
./gen_run_script <system_name> </path/to/nekRS>
```

The script will produce a `run.sh` script specifically tailored to the desired system and using the desired nekRS install directory. 

Finally, simply execute the run script **from the compute nodes** with

```bash
./run.sh
```

The `run.sh` script is composed of two steps:

- The nekRS simulation performing wall-modeled LES of the turbulent channel flow. The simulation produces the `NekAvgData_1D.csv` file containing the time- and plane-averaged flow statistics (velocity and products of velocity).
- A Python post-processing step which generates plots of the nekRS statistics.   

